"""Original LLaVA code and non-HF weights, official Clean COCO POPE cohort."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

SOURCE = Path('/root/code/ClearSight-main/LLaVA')
sys.path.insert(0, str(SOURCE))
import torch
import transformers
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from evaluate import binary_metrics, parse_answer

ROOT = Path(__file__).resolve().parent
MODEL = Path('/root/code/model/llava-v1.5-7b')
SPLITS = ROOT/'data/official_splits.json'
IMAGES = Path('/root/code/dataset/val2014')
OUTPUT = ROOT/'runs/llava15_7b_native_baseline_v1'
SUFFIX = ' Answer only yes or no.'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cohort():
    splits = json.loads(SPLITS.read_text())
    return splits, {(r['image'], r['text'], r['label']): r
                    for group in splits.values() for r in group}


def initialize():
    splits, rows = cohort()
    assert len(rows) == 5127 and all(len(v) == 3000 for v in splits.values())
    assert all((IMAGES/r['image']).is_file() for r in rows.values())
    OUTPUT.mkdir(exist_ok=False)
    protocol = dict(model=str(MODEL), source=str(SOURCE), shards=2, physical_gpus=[2, 4],
                    dtype='float16', attention='original transformers 4.31 eager attention',
                    conv_mode='llava_v1', image_aspect_ratio='pad', seed=20261009,
                    suffix=SUFFIX, max_new_tokens=64, do_sample=False,
                    num_beams=1, repetition_penalty=1., use_cache=True,
                    script_sha256=digest(Path(__file__)), splits_sha256=digest(SPLITS),
                    parser_sha256=digest(ROOT/'evaluate.py'), torch=torch.__version__,
                    transformers=transformers.__version__,
                    source_hashes={str(p.relative_to(SOURCE)): digest(p)
                                   for p in (SOURCE/'llava').rglob('*.py')},
                    model_files={p.name: dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                                 for p in MODEL.iterdir() if p.is_file()})
    (OUTPUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')


def worker(shard):
    protocol = json.loads((OUTPUT/'protocol.json').read_text())
    assert digest(Path(__file__)) == protocol['script_sha256']
    assert digest(SPLITS) == protocol['splits_sha256']
    torch.set_num_threads(4)
    torch.manual_seed(protocol['seed'])
    disable_torch_init()
    tokenizer, model, processor, _ = load_pretrained_model(
        str(MODEL), None, 'llava-v1.5-7b', device_map={'': 0})
    model.eval().requires_grad_(False)
    assert model.config.image_aspect_ratio == 'pad'
    _, indexed = cohort()
    rows = list(indexed.values())[shard::protocol['shards']]

    def answer(row):
        conv = conv_templates['llava_v1'].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN+'\n'+row['text']+protocol['suffix'])
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        assert int((ids == IMAGE_TOKEN_INDEX).sum()) == 1
        with Image.open(IMAGES/row['image']) as image:
            pixels = process_images([image.convert('RGB')], processor, model.config).half().cuda()
        assert tuple(pixels.shape) == (1, 3, 336, 336)
        output = model.generate(ids, images=pixels, do_sample=False, num_beams=1,
                                max_new_tokens=64, repetition_penalty=1., use_cache=True,
                                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        assert torch.equal(output[:, :ids.shape[1]], ids)
        tokens = output[0, ids.shape[1]:].tolist()
        stop = next((i for i, t in enumerate(tokens) if t == tokenizer.eos_token_id), len(tokens))
        content = tokens[:stop]
        return dict(text=tokenizer.decode(content, skip_special_tokens=True).strip(), token_ids=content,
                    stopped=stop < len(tokens), truncated=stop == len(tokens) and len(tokens) >= 64), prompt

    started = time.monotonic()
    with torch.inference_mode(), (OUTPUT/f'answers{shard}.jsonl').open('x') as stream:
        first, prompt = answer(rows[0])
        assert answer(rows[0])[0] == first
        print(f'preflight shard {shard} passed: {first}', flush=True)
        for i, row in enumerate(rows):
            value = first if i == 0 else answer(row)[0]
            stream.write(json.dumps(dict(**row, answer=value, prediction=parse_answer(value['text'])))+'\n')
            stream.flush()
            if (i+1) % 100 == 0:
                print(f'shard {shard}: {i+1}/{len(rows)} elapsed={time.monotonic()-started:.1f}s', flush=True)
        assert answer(rows[0])[0] == first
    audit = dict(questions=len(rows), repeat_exact=True, frozen=all(not p.requires_grad for p in model.parameters()),
                 no_parameter_gradients=all(p.grad is None for p in model.parameters()), prompt_example=prompt,
                 physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'], seconds=time.monotonic()-started,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(), processor=processor.to_dict())
    (OUTPUT/f'audit{shard}.json').write_text(json.dumps(audit, indent=2)+'\n')


def summarize():
    splits, expected = cohort()
    protocol = json.loads((OUTPUT/'protocol.json').read_text())
    assert digest(Path(__file__)) == protocol['script_sha256']
    assert digest(SPLITS) == protocol['splits_sha256']
    assert digest(ROOT/'evaluate.py') == protocol['parser_sha256']
    for name, value in protocol['source_hashes'].items():
        assert digest(SOURCE/name) == value, name
    indexed = {}
    for shard in range(protocol['shards']):
        audit = json.loads((OUTPUT/f'audit{shard}.json').read_text())
        assert audit['repeat_exact'] and audit['frozen'] and audit['no_parameter_gradients']
        lines = (OUTPUT/f'answers{shard}.jsonl').read_text().splitlines()
        assert len(lines) == audit['questions']
        for line in lines:
            row = json.loads(line)
            key = row['image'], row['text'], row['label']
            assert key not in indexed and row['prediction'] == parse_answer(row['answer']['text'])
            indexed[key] = row
    assert indexed.keys() == expected.keys()
    metrics = {name: binary_metrics([r['label'] for r in rows],
               [indexed[r['image'], r['text'], r['label']]['prediction'] for r in rows])
               for name, rows in splits.items()}
    summary = dict(model=str(MODEL), unique_questions=len(indexed), splits=metrics,
                   truncated=sum(r['answer']['truncated'] for r in indexed.values()),
                   macro={k: sum(m[k] for m in metrics.values())/len(metrics) for k in ('accuracy', 'f1')})
    (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init', 'run', 'summarize'])
    parser.add_argument('--shard', type=int, choices=[0, 1], default=0)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--suffix', default=SUFFIX)
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    SUFFIX = args.suffix
    if args.command == 'init':
        initialize()
    elif args.command == 'run':
        worker(args.shard)
    else:
        summarize()
