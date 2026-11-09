"""Frozen representation confirmation and COCO POPE evaluation."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch

from restorer import BLURS, Injector, capture, controller, restoration_metrics, tensors, to_device

HERE = Path(__file__).resolve().parent


def parse_answer(text: str) -> str:
    """Parse without labels; explicit answer words precede negation fallback."""
    sentence = re.split(r"[.!?。！？\n]", text.strip(), maxsplit=1)[0]
    answer = re.search(r"\b(yes|no)\b", sentence, flags=re.IGNORECASE)
    if answer:
        return answer.group(1).lower()
    if re.search(r"\bnot\b", sentence, flags=re.IGNORECASE):
        return "no"
    return "other"


def binary_metrics(labels: list[str], predictions: list[str]) -> dict:
    """Measure binary/native outputs, keeping Other errors visible."""
    if len(labels) != len(predictions) or not labels or set(labels) != {"yes", "no"}:
        raise ValueError(
            "metrics require both labels and matching nonempty predictions"
        )
    if set(predictions) - {"yes", "no", "other"}:
        raise ValueError("invalid prediction")
    truth, pred = np.asarray(labels), np.asarray(predictions)
    tp = int(((truth == "yes") & (pred == "yes")).sum())
    tn = int(((truth == "no") & (pred == "no")).sum())
    fp = int(((truth == "no") & (pred == "yes")).sum())
    fn = int(((truth == "yes") & (pred == "no")).sum())
    other_yes = int(((truth == "yes") & (pred == "other")).sum())
    other_no = int(((truth == "no") & (pred == "other")).sum())
    recall, tnr = tp / int((truth == "yes").sum()), tn / int((truth == "no").sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "questions": len(labels),
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": (recall + tnr) / 2,
        "recall": recall,
        "tnr": tnr,
        "precision": precision,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "other_yes": other_yes,
        "other_no": other_no,
        "other": other_yes + other_no,
    }


def paired_comparison(rows):
    metrics={name:binary_metrics([r['label'] for r in rows],[r[name] for r in rows])
             for name in ('regular_prediction','prediction')}
    images=sorted({r['image'] for r in rows});index={name:i for i,name in enumerate(images)}
    counts=np.zeros((2,len(images),6),dtype=np.int64)
    for row in rows:
        for branch,name in enumerate(('regular_prediction','prediction')):
            pred,pos=row[name],row['label']=='yes'
            cell=(0 if pos else 2) if pred=='yes' else (3 if pos else 1) if pred=='no' else (4 if pos else 5)
            counts[branch,index[row['image']],cell]+=1
    sampled=counts[:,np.random.default_rng(20261011).integers(0,len(images),(5000,len(images)))].sum(2)
    tp,tn,fp,fn,oy,on=[sampled[:,:,i] for i in range(6)]
    values=dict(accuracy=(tp+tn)/sampled.sum(-1),f1=2*tp/np.maximum(2*tp+fp+fn+oy,1),fp=fp,fn=fn)
    differences={k:dict(difference=metrics['prediction'][k]-metrics['regular_prediction'][k],
                         ci95=np.quantile(v[1]-v[0],[.025,.975]).tolist()) for k,v in values.items()}
    return dict(regular=metrics['regular_prediction'],restored=metrics['prediction'],versus_regular=differences,
                rescued=sum(r['regular_prediction']!=r['label'] and r['prediction']==r['label'] for r in rows),
                hurt=sum(r['regular_prediction']==r['label'] and r['prediction']!=r['label'] for r in rows))


def confirm(features, checkpoint):
    item = torch.load(features, map_location='cpu', weights_only=False)
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    space = cp['space']
    model = controller(space, cp['state']).eval()
    x, target = tensors(item['values'], space)
    return dict(**restoration_metrics(model, x, target, space['basis']),
                questions=len(item['rows']), images=len({r['image'] for r in item['rows']}),
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                no_reselection=True)


def summarize(answers, splits, checkpoint):
    expected_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    records = [json.loads(line) for path in sorted(answers.glob('*.jsonl'))
               for line in path.read_text().splitlines() if line.strip()]
    by_key = {}
    for row in records:
        if row['checkpoint_sha256'] != expected_hash:
            raise ValueError('answer checkpoint mismatch')
        key = row['image'], row['text'], row['label']
        if key in by_key:
            raise ValueError('duplicate answer')
        for name in ('regular', 'restored'):
            value = row['values'][name]
            if value['prediction'] != parse_answer(value['answer']['text']):
                raise ValueError('answer parser mismatch')
        by_key[key] = row
    official = json.loads(splits.read_text())
    expected = {(r['image'], r['text'], r['label']) for rows in official.values() for r in rows}
    if set(by_key) != expected:
        raise ValueError('incomplete or unexpected answer cohort')
    results = {}
    for split, rows in official.items():
        cells = []
        for row in rows:
            values = by_key[row['image'], row['text'], row['label']]['values']
            cells.append(dict(**row, regular_prediction=values['regular']['prediction'],
                              prediction=values['restored']['prediction']))
        results[split] = paired_comparison(cells)
    macro = {name: {k: float(np.mean([r[name][k] for r in results.values()]))
                    for k in ('accuracy', 'f1')} for name in ('regular', 'restored')}
    return dict(splits=results, macro=macro, checkpoint_sha256=expected_hash)


def generate_answers(images, splits, checkpoint, destination):
    from run import Qwen2Runtime, generate
    official = json.loads(splits.read_text())
    rows = {(r['image'], r['text'], r['label']): r for group in official.values() for r in group}
    destination.mkdir(parents=True, exist_ok=False)
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if cp['amplitude'] != 1.:
        raise ValueError('requires fixed amplitude 1')
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    runtime = Qwen2Runtime()
    edit = Injector(controller(cp['space'], cp['state']).cuda().eval(), to_device(cp['space'], 'cuda'))
    with (destination/'answers.jsonl').open('x') as stream, torch.inference_mode():
        for i, row in enumerate(rows.values()):
            inputs = runtime.prepare(images, row)
            values = {}
            for name, injector in (('regular', None), ('restored', edit)):
                answer = generate(runtime, inputs, injector)
                values[name] = dict(answer=answer, prediction=parse_answer(answer['text']))
            stream.write(json.dumps(dict(**row, values=values, checkpoint_sha256=checkpoint_hash))+'\n')
            stream.flush()
            if (i+1) % 50 == 0:
                print(f'evaluated {i+1}/{len(rows)}', flush=True)
    return summarize(destination, splits, checkpoint)


class OracleInjector:
    """Offline paired-image diagnostic; never used for deployment."""

    def __init__(self, direction):
        self.direction = direction

    def reset(self, *_span):
        self.injection_count = 0

    def capture(self, _module, _args):
        pass

    def inject(self, _module, args):
        hidden = args[0]
        if self.injection_count or hidden.shape[1] <= 1:
            return None
        changed = hidden.clone()
        changed[:, -1] += self.direction.to(device=hidden.device, dtype=hidden.dtype)
        self.injection_count += 1
        return (changed,) + args[1:]


def diagnose(images, splits, checkpoint, destination, shard, shards):
    from run import Qwen2Runtime, generate
    official = json.loads(splits.read_text())
    rows = {(r['image'], r['text'], r['label']): r for group in official.values() for r in group}
    names = sorted({r['image'] for r in rows.values()})
    assigned = set(names[shard::shards])
    rows = [r for r in rows.values() if r['image'] in assigned]
    destination.mkdir(parents=True, exist_ok=True)
    path = destination/f'shard{shard}.jsonl'
    metadata = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    splits_sha256=hashlib.sha256(splits.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    shard=shard, shards=shards, offline_oracle=True)
    with (destination/f'protocol{shard}.json').open('x') as stream:
        stream.write(json.dumps(dict(**metadata, conditions=BLURS, questions=len(rows)), indent=2)+'\n')
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert cp['amplitude'] == 1.
    runtime = Qwen2Runtime()
    space = to_device(cp['space'], 'cuda')
    edit = Injector(controller(cp['space'], cp['state']).cuda().eval(), space)
    basis = space['basis']
    with path.open('x') as stream, torch.inference_mode():
        for i, row in enumerate(rows):
            clean = runtime.prepare(images, row)
            clean_state = capture(runtime, clean)['question'].cuda()
            for blur in BLURS:
                inputs = clean if blur == 0 else runtime.prepare(images, row, blur)
                current = clean_state if blur == 0 else capture(runtime, inputs)['question'].cuda()
                delta = clean_state-current
                projected = (delta@basis/basis.square().sum(0))@basis.T
                values = {}
                for name, injector in (('regular', None), ('restored', edit),
                                       ('oracle_full', OracleInjector(delta)),
                                       ('oracle_projected', OracleInjector(projected))):
                    if blur == 0 and name.startswith('oracle_'):
                        values[name] = dict(values['regular'], reused_zero_edit=True)
                        continue
                    answer = generate(runtime, inputs, injector)
                    values[name] = dict(answer=answer, prediction=parse_answer(answer['text']))
                stream.write(json.dumps(dict(**row, blur=blur, values=values, **metadata))+'\n')
                stream.flush()
            if (i+1) % 10 == 0:
                print(f'diagnosed shard {shard}: {i+1}/{len(rows)}', flush=True)


def summarize_diagnosis(answers, splits, checkpoint):
    official = json.loads(splits.read_text())
    expected_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    split_hash = hashlib.sha256(splits.read_bytes()).hexdigest()
    records = [json.loads(line) for path in sorted(answers.glob('shard*.jsonl'))
               for line in path.read_text().splitlines() if line.strip()]
    expected = {(r['image'], r['text'], r['label'], blur)
                for rows in official.values() for r in rows for blur in BLURS}
    indexed = {}
    for row in records:
        key = row['image'], row['text'], row['label'], row['blur']
        if key in indexed or row['checkpoint_sha256'] != expected_hash or row['splits_sha256'] != split_hash:
            raise ValueError('duplicate diagnostic answer or provenance mismatch')
        if set(row['values']) != {'regular', 'restored', 'oracle_full', 'oracle_projected'}:
            raise ValueError('incomplete diagnostic branches')
        for value in row['values'].values():
            if value['prediction'] != parse_answer(value['answer']['text']):
                raise ValueError('answer parser mismatch')
        indexed[key] = row
    if set(indexed) != expected or len({r['script_sha256'] for r in records}) != 1:
        raise ValueError('incomplete diagnostic cohort or mixed scripts')
    results = {}
    for blur in BLURS:
        results[str(blur)] = {}
        for split, rows in official.items():
            compared = {}
            for method in ('restored', 'oracle_full', 'oracle_projected'):
                cells = []
                for row in rows:
                    values = indexed[row['image'], row['text'], row['label'], blur]['values']
                    cells.append(dict(**row, regular_prediction=values['regular']['prediction'],
                                      prediction=values[method]['prediction']))
                compared[method] = paired_comparison(cells)
            results[str(blur)][split] = compared
    return dict(checkpoint_sha256=expected_hash, splits_sha256=split_hash,
                conditions=results, records=len(records), offline_oracle=True,
                limitations='Previously observed benchmark; frozen diagnostic, no model selection. '
                             'Oracle requires a clean paired image and is not deployable.',
                truncated={method: sum(r['values'][method]['answer']['truncated'] for r in records)
                           for method in ('regular', 'restored', 'oracle_full', 'oracle_projected')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('confirm', 'summarize', 'pope', 'diagnose', 'diagnose-summary'))
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--checkpoint', type=Path, default=HERE/'controller.pt')
    parser.add_argument('--features', type=Path, default=HERE/'data/features/confirmation.pt')
    parser.add_argument('--splits', type=Path, default=HERE/'data/official_splits.json')
    parser.add_argument('--answers', type=Path, default=HERE/'artifacts/original/answers')
    parser.add_argument('--images', type=Path, default=Path('/root/code/dataset/val2014'))
    parser.add_argument('--output', type=Path, required=True, help='new JSON report; for pope, a new directory')
    args = parser.parse_args()
    if args.command == 'diagnose':
        if not 0 <= args.shard < args.shards:
            parser.error('requires 0 <= shard < shards')
        diagnose(args.images, args.splits, args.checkpoint, args.output, args.shard, args.shards)
        return
    if args.output.exists():
        parser.error('output already exists')
    if args.command == 'confirm':
        result = confirm(args.features, args.checkpoint)
    elif args.command == 'summarize':
        result = summarize(args.answers, args.splits, args.checkpoint)
    elif args.command == 'diagnose-summary':
        result = summarize_diagnosis(args.answers, args.splits, args.checkpoint)
    else:
        result = generate_answers(args.images, args.splits, args.checkpoint, args.output)
    output = args.output/'summary.json' if args.command == 'pope' else args.output
    with output.open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(output)


if __name__ == '__main__':
    main()
