"""Fit train-only space, select by held-out MSE, then refit the Controller."""
import argparse
import json
from pathlib import Path
import random

import torch

from restorer import E, R, controller, make_features, restoration_metrics, sanitized, tensors

SEED = 20261009
EPOCHS = (0, 1, 5, 10, 20, 40)


def spectral(values,center):
    values=values.double(); mean=values.mean(0) if center else torch.zeros(values.shape[1],device=values.device,dtype=values.dtype)
    _,s,v=torch.linalg.svd(values-mean,full_matrices=False,**({'driver': 'gesvd'} if values.is_cuda else {}))
    cumulative=s.square().cumsum(0)/s.square().sum();rank=int(torch.searchsorted(cumulative,.9))+1
    return mean.cpu().float(),v[:rank].cpu().float(),s.cpu(),rank


def fit_space(values):
    delta = (values['question'][:, 0, None]-values['question'][:, 1:]).reshape(-1, values['question'].shape[-1])
    _, right, singular, rank = spectral(delta, False)
    orthogonal = right.T.contiguous()
    coefficient_scale = (delta.cpu() @ orthogonal).square().mean(0).sqrt().clamp_min(1e-6)
    projections = {}
    for key in ('question', 'visual'):
        x = values[key].flatten(0, 1)
        mean, components, s, _ = spectral(x, True)
        projected = (x.cpu()-mean) @ components.T
        projections[key] = dict(mean=mean, components=components,
                                scale=projected.std(0, unbiased=False).clamp_min(1e-6), singular_values=s)
    raw_features = make_features(values['question'].flatten(0, 1).cpu(), values['visual'].flatten(0, 1).cpu(),
                                 values['statistics'].flatten(0, 1).cpu(), projections['question'], projections['visual'])
    return dict(question_projection=projections['question'], visual_projection=projections['visual'],
                feature_mean=raw_features.mean(0), feature_scale=raw_features.std(0, unbiased=False).clamp_min(1e-6),
                basis=orthogonal*coefficient_scale, orthogonal_basis=orthogonal,
                coefficient_scale=coefficient_scale, singular_values=singular,
                rank=rank, input_dim=raw_features.shape[1], evidence_layer=E, recovery_layer=R, model_family='qwen2_vl')


def optimize(x, target, space, epochs, validation=None):
    torch.manual_seed(SEED)
    model = controller(space).to(x.device).train()
    basis = space['basis'].to(x.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    generator = torch.Generator().manual_seed(SEED)
    history = []
    for epoch in range(epochs+1):
        if validation is not None and epoch in EPOCHS:
            history.append(dict(epoch=epoch, **restoration_metrics(model, *validation, basis)))
            print(history[-1], flush=True)
        if epoch == epochs:
            break
        order = torch.randperm(len(x), generator=generator)
        for indices in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            loss = (model(x[indices]) @ basis.T-target[indices]).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
    return model, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    item = torch.load(args.features, map_location='cpu', weights_only=False)
    rows, raw = item['rows'], item['values']
    assert all(r == sanitized(r) for r in rows), 'features must contain no answer fields'
    assert len(rows) == 3000 and len({r['image'] for r in rows}) == 500
    assert len({r['question_id'] for r in rows}) == len(rows)
    assert all(v.shape[:2] == (len(rows), 3) and torch.isfinite(v).all() for v in raw.values())
    images = sorted({r['image'] for r in rows})
    random.Random(SEED).shuffle(images)
    fit_images = set(images[:400])
    mask = torch.tensor([r['image'] in fit_images for r in rows])
    space = fit_space({k: v[mask].cuda() for k, v in raw.items()})
    x, target = tensors(raw, space)
    mask = mask.repeat_interleave(3)
    _, history = optimize(x[mask].cuda(), target[mask].cuda(), space, max(EPOCHS),
                          (x[~mask].cuda(), target[~mask].cuda()))
    winner = min(history, key=lambda m: (m['mse'], m['epoch']))
    model, _ = optimize(x.cuda(), target.cuda(), space, winner['epoch'])
    torch.save(dict(state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    space=space, epoch=winner['epoch'], amplitude=1.), args.output/'controller.pt')
    (args.output/'selection.json').write_text(json.dumps(dict(epoch=winner['epoch'], history=history), indent=2)+'\n')
    print('selected epoch', winner['epoch'], flush=True)


if __name__ == '__main__':
    main()
