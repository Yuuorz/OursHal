"""Experiment A: native LLaVA final-prompt attention versus COCO boxes."""
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from llava_native_baseline import (ROOT, MODEL, SOURCE, IMAGES, digest, torch,
                                  load_pretrained_model, disable_torch_init,
                                  conv_templates, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                                  tokenizer_image_token, process_images, parse_answer)

OUT = ROOT/'runs/llava15_attention_gt_v1'
BASE = ROOT/'runs/llava15_7b_native_please_yesno_v1'
SET = ROOT/'data/llava15_native_please_yesno_fp50_fn50_v1.json'
ANNOTATIONS = Path('/root/code/dataset/annotations/instances_val2014.json')


def union_area(boxes):
    xs = sorted({x for b in boxes for x in (b[0], b[2])})
    total = 0.
    for left, right in zip(xs, xs[1:]):
        spans = sorted((b[1], b[3]) for b in boxes if b[0] < right and b[2] > left)
        length, end = 0., -math.inf
        for lo, hi in spans:
            length += max(0., hi-max(lo, end))
            end = max(end, hi)
        total += (right-left)*length
    return total


def grid_mask(boxes, width, height):
    side = max(width, height)
    ox, oy = (side-width)//2, (side-height)//2
    x, y = np.meshgrid((np.arange(24)+.5)*side/24-ox,
                       (np.arange(24)+.5)*side/24-oy)
    mask = np.zeros((24, 24), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        mask |= (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    support = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    return mask, support


def random_masks(mask, support, seed):
    yy, xx = np.where(mask)
    candidates = []
    if len(xx):
        for dy in range(-int(yy.min()), 24-int(yy.max())):
            for dx in range(-int(xx.min()), 24-int(xx.max())):
                shifted = np.zeros_like(mask)
                shifted[yy+dy, xx+dx] = True
                if not np.any(shifted & mask) and not np.any(shifted & ~support):
                    candidates.append((dy, dx))
    random.Random(seed).shuffle(candidates)
    controls = []
    for dy, dx in candidates[:20]:
        shifted = np.zeros_like(mask)
        shifted[yy+dy, xx+dx] = True
        assert shifted.sum() == mask.sum() and not (shifted & mask).any()
        controls.append(shifted.flatten())
    return np.asarray(controls, dtype=bool).reshape(-1, 576), candidates[:20], len(candidates)


def prepare():
    coco = json.loads(ANNOTATIONS.read_text())
    images = {i['id']: i for i in coco['images']}
    categories = {c['name']: c['id'] for c in coco['categories']}
    annotations = defaultdict(list)
    for ann in coco['annotations']:
        annotations[ann['image_id'], ann['category_id']].append(ann)

    def enrich(row):
        category = row['text'].split(' in the image?')[0].split(' ', 3)[3]
        iid = int(Path(row['image']).stem.split('_')[-1])
        image = images.get(iid)
        boxes, aids = [], []
        if image:
            for ann in annotations[iid, categories[category]]:
                x, y, w, h = ann['bbox']
                box = [max(0., x), max(0., y), min(image['width'], x+w), min(image['height'], y+h)]
                if box[2] > box[0] and box[3] > box[1]:
                    boxes.append(box)
                    aids.append(ann['id'])
        result = dict(row, target=category, coco_image_id=iid, boxes_xyxy=boxes, annotation_ids=aids)
        if image:
            result.update(width=image['width'], height=image['height'],
                          area_fraction=union_area(boxes)/(image['width']*image['height']))
        return result

    selected = json.loads(SET.read_text())
    fns = [enrich(r) for r in selected if r['error_type'] == 'FN']
    excluded = [dict(r, exclusion='image_not_in_val2014_annotations' if 'width' not in r
                     else 'no_target_category_bbox') for r in fns if not r['boxes_xyxy']]
    valid = [r for r in fns if r['boxes_xyxy']]
    baseline = [json.loads(line) for p in sorted(BASE.glob('answers*.jsonl')) for line in p.read_text().splitlines()]
    fn_images = {r['image'] for r in fns}
    targets = {r['target'] for r in valid}
    tps = [enrich(r) for r in baseline if r['label'] == r['prediction'] == 'yes'
           and r['image'] not in fn_images and r['text'].split(' in the image?')[0].split(' ', 3)[3] in targets]
    tps = [r for r in tps if r['boxes_xyxy']]
    used, rows = set(), []
    for pair, fn in enumerate(sorted(valid, key=lambda r: (sum(t['target'] == r['target'] for t in tps), r['question_id']))):
        choices = [t for t in tps if t['target'] == fn['target'] and t['image'] not in used]
        assert choices, fn['target']
        tp = min(choices, key=lambda t: (abs(math.log(t['area_fraction']/fn['area_fraction'])), t['image'], t['text']))
        used.add(tp['image'])
        for group, row in [('FN', fn), ('TP', tp)]:
            rows.append(dict(row, group=group, pair_id=pair, sample_id=len(rows),
                             match_log_area_gap=abs(math.log(tp['area_fraction']/fn['area_fraction']))))
    assert len(valid) + len(excluded) == 50 and len(rows) == 2*len(valid)
    OUT.mkdir(exist_ok=False)
    (OUT/'cohort.json').write_text(json.dumps(rows, indent=2)+'\n')
    (OUT/'excluded.json').write_text(json.dumps(excluded, indent=2)+'\n')
    protocol = dict(query='last token of complete prompt; user confirmed', layers=32, heads=32,
                    grid=[24, 24], mask='patch center inside clipped union of ALL target-category bboxes',
                    bbox_transform='native integer-centered square padding, scale to336; patch center mapping',
                    mass='mean-head attention inside GT / mean-head attention over ALL576 visual tokens',
                    auroc='tie-aware Mann-Whitney, GT-center masks versus all other visual patches; undefined for zero/full masks',
                    controls='up to20 uniform legal integer-patch translations of the entire box-union mask, without replacement; disjoint GT, same exact patch area/shape, no padded patch centers',
                    control_unavailable='record NA, do not fabricate random regions or discard sample from GT analysis',
                    matching='unique TP images outside all50 FN images; exact target category, greedy nearest log union-bbox area/image area; rare categories first',
                    fn=len(valid), tp=len(valid), excluded_fn=len(excluded), fp='not used: absent target has no GT localization metric',
                    seed=20261009, gpus=[2, 4], source_run=str(BASE), script_sha256=digest(Path(__file__)),
                    cohort_sha256=digest(OUT/'cohort.json'), annotation_sha256=digest(ANNOTATIONS),
                    set_sha256=digest(SET), baseline_protocol=json.loads((BASE/'protocol.json').read_text()))
    (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps(dict(fn=len(valid), tp=len(valid), excluded=len(excluded))))


def extract(shard):
    from PIL import Image
    protocol = json.loads((OUT/'protocol.json').read_text())
    assert digest(Path(__file__)) == protocol['script_sha256']
    assert digest(OUT/'cohort.json') == protocol['cohort_sha256']
    torch.set_num_threads(4)
    torch.manual_seed(protocol['seed'])
    disable_torch_init()
    tokenizer, model, processor, _ = load_pretrained_model(str(MODEL), None, 'llava-v1.5-7b', device_map={'': 0})
    model.eval().requires_grad_(False)
    rows = json.loads((OUT/'cohort.json').read_text())[shard::2]
    results = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            conv = conv_templates['llava_v1'].copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN+'\n'+row['text']+' Please answer yes or no.')
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            start = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero().item())
            with Image.open(IMAGES/row['image']) as image:
                assert image.size == (row['width'], row['height'])
                pixels = process_images([image.convert('RGB')], processor, model.config).half().cuda()
            plain = model(ids, images=pixels, use_cache=False).logits[:, -1].clone()
            outputs = model(ids, images=pixels, use_cache=False, output_attentions=True)
            assert torch.equal(plain, outputs.logits[:, -1]), 'attention capture changed logits'
            assert len(outputs.attentions) == 32
            raw = []
            for attention in outputs.attentions:
                assert attention.shape == (1, 32, ids.shape[1]+575, ids.shape[1]+575)
                assert torch.allclose(attention[0, :, -1].float().sum(-1), torch.ones(32, device='cuda'), atol=.002, rtol=0)
                raw.append(attention[0, :, -1, start:start+576].float().cpu().numpy())
            raw = np.stack(raw)
            assert np.isfinite(raw).all() and (raw >= 0).all()
            mask, support = grid_mask(row['boxes_xyxy'], row['width'], row['height'])
            controls, shifts, count = random_masks(mask, support, protocol['seed']+row['sample_id'])
            np.savez_compressed(OUT/f"sample{row['sample_id']:03d}.npz", per_head=raw,
                                attention=raw.mean(1), gt_mask=mask.flatten(), controls=controls)
            generated = model.generate(ids, images=pixels, do_sample=False, num_beams=1, max_new_tokens=64,
                                       repetition_penalty=1., use_cache=True, eos_token_id=tokenizer.eos_token_id,
                                       pad_token_id=tokenizer.eos_token_id)
            assert torch.equal(generated[:, :ids.shape[1]], ids)
            answer = tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True).strip()
            assert answer == row['answer']['text'] and parse_answer(answer) == row['prediction'], 'baseline answer drift'
            if index == 0:
                repeat = model(ids, images=pixels, use_cache=False, output_attentions=True)
                for layer, a in enumerate(repeat.attentions):
                    assert np.array_equal(a[0, :, -1, start:start+576].float().cpu().numpy(), raw[layer])
                del repeat
            results.append(dict(sample_id=row['sample_id'], query_token_id=int(ids[0, -1]),
                                query_token=tokenizer.convert_ids_to_tokens(int(ids[0, -1])), visual_start=start,
                                query_position=ids.shape[1]+574, gt_patches=int(mask.sum()),
                                random_controls=len(controls), legal_random_translations=count, shifts=shifts,
                                capture_logits_exact=True, answer_exact=True))
            del outputs, plain
            print(f"shard{shard}: {index+1}/{len(rows)} sample{row['sample_id']} GT={mask.sum()} random={len(controls)}", flush=True)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    (OUT/f'audit{shard}.json').write_text(json.dumps(dict(rows=results, frozen=True, repeat_exact=True), indent=2)+'\n')


def auroc(scores, mask):
    n = int(mask.sum())
    if n == 0 or n == len(mask):
        return np.full(scores.shape[0], np.nan)
    result = []
    for values in scores:
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        ranks = np.cumsum(counts)-(counts-1)/2
        result.append((ranks[inverse][mask].sum()-n*(n+1)/2)/(n*(len(mask)-n)))
    return np.array(result)


def analyze():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = json.loads((OUT/'cohort.json').read_text())
    audits = [json.loads((OUT/f'audit{s}.json').read_text()) for s in range(2)]
    assert {r['sample_id'] for a in audits for r in a['rows']} == {r['sample_id'] for r in rows}
    assert all(a['frozen'] and a['repeat_exact'] for a in audits)
    assert auroc(np.array([[0., 0., 1., 1.], [1., 1., 0., 0.], [1., 1., 1., 1.]]), np.array([0, 0, 1, 1], bool)).tolist() == [1., 0., .5]
    tables, values = [], defaultdict(list)
    for row in rows:
        data = np.load(OUT/f"sample{row['sample_id']:03d}.npz")
        attention, mask, controls = data['attention'].astype(float), data['gt_mask'], data['controls']
        sums = attention.sum(1)
        assert (sums > 0).all() and np.allclose(attention, data['per_head'].mean(1))
        mass = attention[:, mask].sum(1)/sums
        roc = auroc(attention, mask)
        random_mass = np.mean([attention[:, c].sum(1)/sums for c in controls], axis=0) if len(controls) else np.full(32, np.nan)
        random_roc = np.mean([auroc(attention, c) for c in controls], axis=0) if len(controls) else np.full(32, np.nan)
        entry = np.stack([mass, roc, random_mass, random_roc, mass-random_mass, roc-random_roc], axis=1)
        values[row['group']].append(entry)
        for layer, cell in enumerate(entry):
            tables.append(dict(sample_id=row['sample_id'], pair_id=row['pair_id'], group=row['group'], layer=layer,
                               gt_patches=int(mask.sum()), controls=len(controls),
                               **{k: (float(v) if np.isfinite(v) else '') for k, v in zip(
                                   ['gt_mass', 'auroc', 'random_mass', 'random_auroc', 'mass_advantage', 'auroc_advantage'], cell)}))
    with (OUT/'per_sample_layer.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(tables[0]))
        writer.writeheader()
        writer.writerows(tables)
    summary = {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for group, entries in values.items():
        array = np.stack(entries)
        mean = np.array([[np.mean(array[np.isfinite(array[:, l, k]), l, k]) if np.isfinite(array[:, l, k]).any() else np.nan
                          for k in range(6)] for l in range(32)])
        summary[group] = dict(samples=len(entries), valid_auroc=int(np.isfinite(array[:, 0, 1]).sum()),
                              random_eligible=int(np.isfinite(array[:, 0, 2]).sum()), layer_means=mean.tolist())
        eligible = np.isfinite(array[:, 0, 2])
        paired = array[eligible].mean(0)
        summary[group]['random_matched_layer_means'] = paired.tolist()
        label = f'{group} (n={int(eligible.sum())})'
        axes[0].plot(range(32), paired[:, 4], label=label)
        axes[1].plot(range(32), paired[:, 1], label=label+' GT')
        axes[1].plot(range(32), paired[:, 3], linestyle='--', label=label+' random')
    axes[0].axhline(0, color='gray', linewidth=.8)
    axes[1].axhline(.5, color='gray', linewidth=.8)
    for ax, title, ylabel in zip(axes, ['GT mass minus matched random mass', 'Attention AUROC'], ['Visual attention fraction difference', 'AUROC']):
        ax.set(xlabel='Decoder layer (zero-based)', ylabel=ylabel, title=title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT/'layer_comparison.png', dpi=180)
    plt.close(fig)
    summary['columns'] = ['gt_mass', 'auroc', 'random_mass', 'random_auroc', 'mass_advantage', 'auroc_advantage']
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: {n: v for n, v in r.items() if n != 'layer_means'} for k, r in summary.items() if k != 'columns'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'extract', 'analyze'])
    parser.add_argument('--shard', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare()
    elif args.command == 'extract':
        extract(args.shard)
    else:
        analyze()
