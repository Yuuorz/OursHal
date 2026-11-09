"""Native LLaVA-1.5-7B baseline on the existing official COCO POPE splits."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, GenerationConfig, LlavaForConditionalGeneration

from evaluate import binary_metrics, parse_answer
from run import decode_generated

ROOT = Path(__file__).resolve().parent
MODEL = Path('/mnt/code/model/llava-1.5-7b-hf')
IMAGES = Path('/root/code/dataset/val2014')
SPLITS = ROOT/'data/official_splits.json'
OUTPUT = ROOT/'runs/llava15_7b_baseline_v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort():
    splits = json.loads(SPLITS.read_text())
    rows = {(r['image'], r['text'], r['label']): r for group in splits.values() for r in group}
    return splits, rows


def initialize():
    splits, rows = cohort()
    assert all(len(group) == 3000 for group in splits.values()) and len(rows) == 5127
    assert all((IMAGES/r['image']).is_file() for r in rows.values())
    OUTPUT.mkdir(exist_ok=False)
    protocol = dict(model=str(MODEL), images=str(IMAGES), unique_questions=len(rows),
                    split_questions={k: len(v) for k, v in splits.items()}, seed=20261009,
                    dtype='bfloat16', attention='sdpa', prompt_suffix=' Answer only yes or no.',
                    template='checkpoint apply_chat_template; single image then question',
                    preprocessing=json.loads((MODEL/'preprocessor_config.json').read_text()),
                    do_sample=False, num_beams=1, max_new_tokens=64, repetition_penalty=1.,
                    use_cache=True, shards=1, physical_gpus=[4],
                    splits_sha256=digest(SPLITS), script_sha256=digest(Path(__file__)),
                    parser_sha256=digest(ROOT/'evaluate.py'),
                    model_files={p.name: dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                                 for p in MODEL.iterdir() if p.is_file()},
                    config_hashes={p.name: digest(p) for p in MODEL.iterdir()
                                   if p.suffix in ('.json', '.jinja')},
                    torch=torch.__version__, transformers=transformers.__version__)
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')


def worker(shard):
    protocol = json.loads((OUTPUT/'protocol.json').read_text())
    assert protocol['script_sha256'] == digest(Path(__file__))
    assert protocol['splits_sha256'] == digest(SPLITS)
    torch.set_num_threads(4)
    torch.manual_seed(protocol['seed'])
    _, indexed = cohort()
    rows = list(indexed.values())[shard::protocol['shards']]
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, use_fast=False)
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
        local_files_only=True).cuda().eval().requires_grad_(False)
    config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=64,
                              repetition_penalty=1., use_cache=True,
                              eos_token_id=model.generation_config.eos_token_id,
                              pad_token_id=processor.tokenizer.pad_token_id,
                              bos_token_id=model.generation_config.bos_token_id)

    def answer(row):
        messages = [dict(role='user', content=[dict(type='image'),
                    dict(type='text', text=row['text']+protocol['prompt_suffix'])])]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        with Image.open(IMAGES/row['image']) as image:
            inputs = processor(text=prompt, images=image.convert('RGB'), return_tensors='pt')
        inputs = inputs.to('cuda', torch.bfloat16)
        assert int((inputs['input_ids'] == model.config.image_token_index).sum()) == 576
        generated = model.generate(**inputs, generation_config=config)
        tokens = generated[0, inputs['input_ids'].shape[1]:].tolist()
        eos = config.eos_token_id
        return decode_generated(processor.tokenizer, tokens,
                                set(eos if isinstance(eos, list) else [eos]), 64), prompt

    started = time.monotonic()
    with torch.inference_mode(), (OUTPUT/f'answers{shard}.jsonl').open('x') as stream:
        first, prompt = answer(rows[0])
        assert answer(rows[0])[0] == first, 'preflight generation is not reproducible'
        for i, row in enumerate(rows):
            value = first if i == 0 else answer(row)[0]
            stream.write(json.dumps(dict(**row, answer=value, prediction=parse_answer(value['text'])))+'\n')
            stream.flush()
            if (i+1) % 100 == 0:
                print(f'shard {shard}: {i+1}/{len(rows)} elapsed={time.monotonic()-started:.1f}s', flush=True)
        assert answer(rows[0])[0] == first, 'end-of-run generation changed'
    audit = dict(questions=len(rows), repeat_exact=True, backbone_frozen=True,
                 no_parameter_gradients=all(p.grad is None for p in model.parameters()),
                 prompt_example=prompt, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 seconds=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    (OUTPUT/f'audit{shard}.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps(audit), flush=True)


def summarize():
    splits, expected = cohort()
    protocol = json.loads((OUTPUT/'protocol.json').read_text())
    assert protocol['script_sha256'] == digest(Path(__file__))
    assert protocol['splits_sha256'] == digest(SPLITS)
    indexed = {}
    for shard in range(protocol['shards']):
        audit = json.loads((OUTPUT/f'audit{shard}.json').read_text())
        assert audit['repeat_exact'] and audit['no_parameter_gradients']
        for line in (OUTPUT/f'answers{shard}.jsonl').read_text().splitlines():
            row = json.loads(line)
            key = row['image'], row['text'], row['label']
            assert key not in indexed
            assert row['prediction'] == parse_answer(row['answer']['text'])
            indexed[key] = row
    assert indexed.keys() == expected.keys(), 'incomplete or unexpected cohort'
    metrics = {name: binary_metrics([r['label'] for r in rows],
               [indexed[r['image'], r['text'], r['label']]['prediction'] for r in rows])
               for name, rows in splits.items()}
    summary = dict(model=str(MODEL), unique_questions=len(indexed), splits=metrics,
                   truncated=sum(r['answer']['truncated'] for r in indexed.values()),
                   macro={k: sum(m[k] for m in metrics.values())/len(metrics)
                          for k in ('accuracy', 'f1')},
                   checks='Exact coverage, no duplicates, parser parity, frozen model, start/end repeat passed')
    (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'run', 'summarize'])
    parser.add_argument('--shard', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    if args.command == 'init':
        initialize()
    elif args.command == 'run':
        worker(args.shard)
    else:
        summarize()
