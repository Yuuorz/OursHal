"""Clean POPE: L7 visual-logit bias 0.1*p_visual*a_j/(mean_visual(a)+epsilon)."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from evaluate import paired_comparison, parse_answer
from restorer import E
from train import SEED

HERE = Path(__file__).resolve().parent
STRENGTH = .1
EPSILON = 1e-12


class VisualAttentionBias:
    def __init__(self, start, count, strength=STRENGTH):
        self.start, self.count, self.strength = start, count, strength
        self.applications = 0

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        output, weights = sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
        if query.shape[-2] <= 1:
            return output, weights
        assert query.shape[0] == 1 and query.shape[-2] == key.shape[-2]
        assert 0 <= self.start < self.start+self.count < key.shape[-2]
        self.applications += 1
        if self.strength == 0:
            return output, weights
        keys = repeat_kv(key, module.num_key_value_groups)
        values = repeat_kv(value, module.num_key_value_groups)
        scores = (query[:, :, -1:].float()@keys.float().transpose(-2, -1))*kwargs['scaling']
        if attention_mask is not None:
            mask = attention_mask[..., -1:, :key.shape[-2]]
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, -torch.inf)
            else:
                scores = scores+mask.float()
        probability = scores.softmax(-1)[..., self.start:self.start+self.count]
        mass = probability.sum(-1, keepdim=True)
        bias = self.strength*mass*probability/(probability.mean(-1, keepdim=True)+EPSILON)
        offset = bias.amax(-1, keepdim=True)
        base = (-offset).exp()
        gain = (bias-offset).exp()-base
        weighted = probability*gain
        contribution = weighted@values[:, :, self.start:self.start+self.count].float()
        # Shift all exponentials equally to avoid overflow without clipping the requested bias.
        last = (base*output[:, -1:].transpose(1, 2).float()+contribution)/(base+weighted.sum(-1, keepdim=True))
        changed = output.clone()
        changed[:, -1:] = last.transpose(1, 2).to(output.dtype)
        return changed, weights


def generate_biased(runtime, inputs, strength=STRENGTH):
    module = runtime.language_layer(E).self_attn
    assert module.config._attn_implementation == 'sdpa' and module.sliding_window is None
    edit = VisualAttentionBias(*runtime.visual_span(inputs), strength)
    original = module.config
    name = 'ours_visual_logit_bias'
    assert name not in ALL_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS[name] = edit
    module.config = copy.copy(original)
    module.config._attn_implementation = name
    try:
        answer, _ = runtime.generate(inputs)
        assert edit.applications == 1
        return answer
    finally:
        module.config = original
        del ALL_ATTENTION_FUNCTIONS[name]


def run(args):
    from run import Qwen2Runtime
    protocol_hash = hashlib.sha256((args.output/'protocol.json').read_bytes()).hexdigest()
    official = json.loads((HERE/'data/official_splits.json').read_text())
    rows = {(r['image'], r['text'], r['label']): r for group in official.values() for r in group}
    assigned = set(sorted({r['image'] for r in rows.values()})[args.shard::args.shards])
    rows = [r for r in rows.values() if r['image'] in assigned]
    torch.manual_seed(SEED)
    runtime = Qwen2Runtime()
    with (args.output/f'answers{args.shard}.jsonl').open('x') as stream, torch.inference_mode():
        for i, row in enumerate(rows):
            inputs = runtime.prepare(args.images, row)
            if i == 0:
                native, _ = runtime.generate(inputs)
                assert generate_biased(runtime, inputs, 0.) == native, 'zero-bias parity failed'
                generate_biased(runtime, inputs)
                (args.output/f'preflight{args.shard}.json').write_text(json.dumps(dict(zero_bias_exact_parity=True))+'\n')
            values = {}
            order = ('regular', 'biased') if i % 2 == 0 else ('biased', 'regular')
            for method in order:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                answer = runtime.generate(inputs)[0] if method == 'regular' else generate_biased(runtime, inputs)
                torch.cuda.synchronize()
                values[method] = dict(answer=answer, prediction=parse_answer(answer['text']),
                                      seconds=time.perf_counter()-started,
                                      peak_memory_bytes=torch.cuda.max_memory_allocated())
            stream.write(json.dumps(dict(**row, values=values, protocol_sha256=protocol_hash))+'\n')
            stream.flush()
            if (i+1) % 25 == 0:
                print('evaluated', args.shard, i+1, '/', len(rows), flush=True)


def summarize(args):
    official = json.loads((HERE/'data/official_splits.json').read_text())
    records = [json.loads(line) for p in sorted(args.output.glob('answers[0-9]*.jsonl'))
               for line in p.read_text().splitlines() if line.strip()]
    indexed = {(r['image'], r['text'], r['label']): r for r in records}
    expected = {(r['image'], r['text'], r['label']) for rows in official.values() for r in rows}
    digest = hashlib.sha256((args.output/'protocol.json').read_bytes()).hexdigest()
    assert len(records) == len(indexed) == 5127 and set(indexed) == expected
    assert all(r['protocol_sha256'] == digest for r in records)
    for row in records:
        assert set(row['values']) == {'regular', 'biased'}
        for value in row['values'].values():
            assert value['prediction'] == parse_answer(value['answer']['text'])
    splits = {}
    for split, rows in official.items():
        cells = []
        for row in rows:
            values = indexed[row['image'], row['text'], row['label']]['values']
            cells.append(dict(**row, regular_prediction=values['regular']['prediction'], prediction=values['biased']['prediction']))
        splits[split] = paired_comparison(cells)
    accepted = all(v['versus_regular'][k]['ci95'][0] > 0 for v in splits.values() for k in ('accuracy', 'f1'))
    accepted = accepted and all(v['restored']['fp'] <= v['regular']['fp'] for v in splits.values())
    performance = {m: dict(mean_seconds=float(np.mean([r['values'][m]['seconds'] for r in records])),
                           mean_peak_memory_bytes=float(np.mean([r['values'][m]['peak_memory_bytes'] for r in records])),
                           max_peak_memory_bytes=max(r['values'][m]['peak_memory_bytes'] for r in records),
                           truncated=sum(r['values'][m]['answer']['truncated'] for r in records))
                   for m in ('regular', 'biased')}
    macro = {m: {k: float(np.mean([s[field][k] for s in splits.values()])) for k in ('accuracy', 'f1')}
             for m, field in (('regular', 'regular'), ('biased', 'restored'))}
    result = dict(splits=splits, descriptive_macro=macro, performance=performance, accepted=accepted,
                  protocol_sha256=digest, unique_questions=len(records), no_training=True)
    with (args.output/'summary.json').open('x') as stream:
        stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'summarize'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--images', type=Path, default=Path('/root/code/dataset/val2014'))
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=5)
    args = parser.parse_args()
    if not 0 <= args.shard < args.shards:
        parser.error('requires 0 <= shard < shards')
    globals()[args.command](args)


if __name__ == '__main__':
    main()
