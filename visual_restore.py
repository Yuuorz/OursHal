"""Fixed pre-L7 visual-token restoration, fitted without answer supervision."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from restorer import BLURS, E, Captured, controller, sanitized, restoration_metrics, to_device
from train import optimize, spectral

HERE = Path(__file__).resolve().parent
SAMPLES = 32


def capture_visual(runtime, inputs):
    start, count = runtime.visual_span(inputs)
    record = []

    def stop(_module, args):
        record.append(args[0][:, start:start+count].detach().float().cpu())
        raise Captured()

    handle = runtime.language_layer(E).register_forward_pre_hook(stop)
    try:
        try:
            runtime.model.model(**inputs, use_cache=False, return_dict=True)
        except Captured:
            pass
    finally:
        handle.remove()
    if len(record) != 1:
        raise ValueError('visual prefix was not captured exactly once')
    return record[0]


def features(values, space):
    projection = space['visual_projection']
    return ((values.float()-projection['mean'])@projection['components'].T)/projection['scale']


class VisualInjector:
    def __init__(self, model=None, space=None, direction=None):
        self.model, self.space, self.direction = model, space, direction

    def reset(self, start, count):
        self.start, self.count, self.injection_count = start, count, 0

    def inject(self, _module, args):
        hidden = args[0]
        if self.injection_count or hidden.shape[1] <= 1:
            return None
        visual = hidden[:, self.start:self.start+self.count]
        direction = self.direction
        if direction is None:
            direction = self.model(features(visual, self.space))@self.space['basis'].T
        if direction.shape != visual.shape:
            raise ValueError('visual residual shape mismatch')
        changed = hidden.clone()
        changed[:, self.start:self.start+self.count] += direction.to(hidden.device, hidden.dtype)
        self.injection_count += 1
        return (changed,) + args[1:]


def generate_visual(runtime, inputs, edit):
    edit.reset(*runtime.visual_span(inputs))
    handle = runtime.language_layer(E).register_forward_pre_hook(edit.inject)
    try:
        answer, _ = runtime.generate(inputs)
        assert edit.injection_count == 1
        return answer
    finally:
        handle.remove()


def extract(args):
    from run import Qwen2Runtime
    rows = json.loads((HERE/f'data/{args.stage}.json').read_text())
    by_image = {r['image']: sanitized(r) for r in rows}
    names = sorted(by_image)[args.shard::args.shards]
    runtime = Qwen2Runtime()
    values = []
    with torch.inference_mode():
        for i, name in enumerate(names):
            records = [capture_visual(runtime, runtime.prepare(args.images, by_image[name], blur))
                       for blur in BLURS]
            assert len({r.shape for r in records}) == 1
            indices = torch.linspace(0, records[0].shape[1]-1, SAMPLES).round().long()
            values.append(torch.stack([r[0, indices] for r in records], dim=1))
            if (i+1) % 10 == 0:
                print('extracted', args.stage, args.shard, i+1, '/', len(names), flush=True)
    torch.save(dict(images=names, values=torch.stack(values), layer=E, samples=SAMPLES,
                    manifest_sha256=hashlib.sha256((HERE/f'data/{args.stage}.json').read_bytes()).hexdigest()),
               args.output/f'{args.stage}{args.shard}.pt')


def load_features(root, stage):
    items = [torch.load(p, map_location='cpu', weights_only=False)
             for p in sorted(root.glob(f'{stage}[0-9]*.pt'))]
    images = [name for item in items for name in item['images']]
    expected = {r['image'] for r in json.loads((HERE/f'data/{stage}.json').read_text())}
    digest = hashlib.sha256((HERE/f'data/{stage}.json').read_bytes()).hexdigest()
    if len(images) != len(set(images)) or set(images) != expected:
        raise ValueError('incomplete or duplicate image cache')
    assert all(i['layer'] == E and i['samples'] == SAMPLES and i['manifest_sha256'] == digest for i in items)
    values = torch.cat([i['values'] for i in items])
    assert values.shape[:3] == (len(images), SAMPLES, 3) and torch.isfinite(values).all()
    return images, values


def train(args):
    images, raw = load_features(args.output, 'train')
    fit = {r['image'] for r in json.loads((HERE/'artifacts/original/fit_manifest.json').read_text())}
    mask = torch.tensor([name in fit for name in images])
    assert int(mask.sum()) == 400
    values = raw[mask].flatten(0, 1).cuda()
    delta = (values[:, :1]-values[:, 1:]).flatten(0, 1)
    _, directions, _, rank = spectral(delta, False)
    basis = directions.T.contiguous()
    scale = (delta.cpu()@basis).square().mean(0).sqrt().clamp_min(1e-6)
    mean, components, _, _ = spectral(values.flatten(0, 1), True)
    projected = (values.flatten(0, 1).cpu()-mean)@components.T
    space = dict(rank=rank, input_dim=len(components), basis=basis*scale, layer=E,
                 visual_projection=dict(mean=mean, components=components,
                                        scale=projected.std(0, unbiased=False).clamp_min(1e-6)))
    del values, delta, projected
    x = features(raw.flatten(0, 2), space)
    target = (raw[:, :, :1]-raw).flatten(0, 2)
    mask = mask.repeat_interleave(SAMPLES*3)
    _, history = optimize(x[mask].cuda(), target[mask].cuda(), space, 40,
                          (x[~mask].cuda(), target[~mask].cuda()))
    winner = min(history, key=lambda m: (m['mse'], m['epoch']))
    selection = dict(epoch=winner['epoch'], history=history, layer=E,
                     fit_images=400, validation_images=100, samples_per_image=SAMPLES,
                     answer_supervision=False, criterion='three-condition full-hidden visual-token MSE')
    (args.output/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    model, _ = optimize(x.cuda(), target.cuda(), space, winner['epoch'])
    torch.save(dict(space=space, state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    epoch=winner['epoch'], amplitude=1., method='visual_token_preL7'), args.output/'controller.pt')
    names, confirmation = load_features(args.output, 'confirmation')
    assert not set(images)&set(names)
    q = confirmation.flatten(0, 1)
    metrics = restoration_metrics(model, features(q.flatten(0, 1), space).cuda(),
                                  (q[:, :1]-q).flatten(0, 1).cuda(), space['basis'].cuda())
    (args.output/'representation_confirmation.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print('selected', winner['epoch'], 'rank', rank, 'input', space['input_dim'], 'confirmation', metrics, flush=True)


def evaluate(args):
    from evaluate import parse_answer
    from restorer import Injector
    from run import Qwen2Runtime, generate
    cp = torch.load(args.output/'controller.pt', map_location='cpu', weights_only=False)
    old = torch.load(HERE/'controller.pt', map_location='cpu', weights_only=False)
    rows = json.loads(args.manifest.read_text())
    images = sorted({r['image'] for r in rows})
    assigned = set(images[args.shard::args.shards])
    rows = [r for r in rows if r['image'] in assigned]
    runtime = Qwen2Runtime()
    visual = VisualInjector(controller(cp['space'], cp['state']).cuda().eval(), to_device(cp['space'], 'cuda'))
    current = Injector(controller(old['space'], old['state']).cuda().eval(), to_device(old['space'], 'cuda'))
    checkpoint_hash = hashlib.sha256((args.output/'controller.pt').read_bytes()).hexdigest()
    with (args.output/f'answers{args.shard}.jsonl').open('x') as stream, torch.inference_mode():
        for i, row in enumerate(rows):
            clean = runtime.prepare(args.images, sanitized(row))
            clean_visual = capture_visual(runtime, clean)
            for blur in BLURS:
                inputs = clean if blur == 0 else runtime.prepare(args.images, sanitized(row), blur)
                delta = clean_visual-(clean_visual if blur == 0 else capture_visual(runtime, inputs))
                values = {}
                for name in ('regular', 'current', 'visual', 'oracle_visual'):
                    if name == 'oracle_visual' and blur == 0:
                        values[name] = dict(values['regular'], reused_zero_edit=True)
                        continue
                    if name in ('regular', 'current'):
                        answer = generate(runtime, inputs, None if name == 'regular' else current)
                    else:
                        edit = visual if name == 'visual' else VisualInjector(direction=delta)
                        answer = generate_visual(runtime, inputs, edit)
                    values[name] = dict(answer=answer, prediction=parse_answer(answer['text']))
                stream.write(json.dumps(dict(**row, blur=blur, values=values, checkpoint_sha256=checkpoint_hash))+'\n')
                stream.flush()
            if (i+1) % 10 == 0:
                print('evaluated', args.shard, i+1, '/', len(rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('extract', 'train', 'evaluate'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('train', 'confirmation'), default='train')
    parser.add_argument('--images', type=Path, default=Path('/root/code/dataset/val2017'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=5)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error('requires 0 <= shard < shards')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == 'extract' and (args.output/f'{args.stage}{args.shard}.pt').exists():
        parser.error('cache already exists')
    if args.command == 'train' and (args.output/'controller.pt').exists():
        parser.error('checkpoint already exists')
    if args.command == 'evaluate' and args.manifest is None:
        parser.error('evaluate requires a labelled evaluation manifest')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
