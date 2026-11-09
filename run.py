"""Model adapter and the two online operations: extract features or generate."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageFilter
from transformers import AutoProcessor, GenerationConfig, Qwen2VLForConditionalGeneration

from restorer import BLURS, E, R, Injector, capture, controller, sanitized, to_device

MODEL = Path('/root/code/model/Qwen2-VL-7B-Instruct')
HERE = Path(__file__).resolve().parent


def full_gaussian(image, level):
    return image if level == 0 else image.filter(ImageFilter.GaussianBlur(float(level)*min(image.size)))


def decode_generated(tokenizer, tokens: list[int], eos: set[int], limit: int) -> dict:
    """Remove stop/padding tokens and record whether generation hit the limit."""
    stop = next((i for i, token in enumerate(tokens) if token in eos), len(tokens))
    content = tokens[:stop]
    return {
        "text": tokenizer.decode(content, skip_special_tokens=True).strip(),
        "token_ids": content,
        "stopped": stop < len(tokens),
        "truncated": stop == len(tokens) and len(tokens) >= limit,
    }


def image_span(input_ids, token_id):
    positions=(input_ids[0]==token_id).nonzero().flatten()
    if len(positions)<2 or not torch.equal(positions,torch.arange(positions[0],positions[-1]+1,device=positions.device)):
        raise ValueError('requires one contiguous model-native image span')
    return int(positions[0]),len(positions)


class Qwen2Runtime:
    family='qwen2_vl'

    def __init__(self):
        torch.set_num_threads(4)
        self.processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,use_fast=False)
        self.tokenizer=self.processor.tokenizer
        self.model=Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL,torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval().requires_grad_(False)
        assert len(self.model.model.language_model.layers)==28

    def language_layer(self,index):
        return self.model.model.language_model.layers[index]

    def visual_span(self,inputs):
        return image_span(inputs['input_ids'],self.model.config.image_token_id)

    def prepare(self,root,row,blur=0.):
        with Image.open(root/row['image']) as source:
            image=full_gaussian(source.convert('RGB'),blur)
        messages=[dict(role='user',content=[dict(type='image'),
                  dict(type='text',text=row['text']+' Answer only yes or no.')])]
        prompt=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[prompt],images=[image],return_tensors='pt',padding=False).to('cuda')
        start,count=self.visual_span(inputs)
        grid=inputs['image_grid_thw'][0]
        assert int(grid.prod())//4==count and start+count<inputs['input_ids'].shape[1]
        return dict(inputs)

    def forward(self,inputs):
        return self.model(**inputs,use_cache=False,return_dict=True).logits

    def generation_config(self):
        return GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=64,use_cache=True,
            repetition_penalty=1.,eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,bos_token_id=self.model.generation_config.bos_token_id)

    def generate(self,inputs):
        result=self.model.generate(**inputs,generation_config=self.generation_config(),
                                   return_dict_in_generate=True,output_scores=True)
        count=len(result.scores)
        assert 0<count<=64 and result.sequences.shape[0]==1
        tokens=result.sequences[0,-count:].tolist()
        eos=self.model.generation_config.eos_token_id
        return decode_generated(self.tokenizer,tokens,set(eos if isinstance(eos,list) else [eos]),64),tokens[0]


def generate(runtime, inputs, injector=None):
    handles = []
    if injector is not None:
        injector.reset(*runtime.visual_span(inputs))
        handles = [runtime.language_layer(E).register_forward_pre_hook(injector.capture),
                   runtime.language_layer(R).register_forward_pre_hook(injector.inject)]
    try:
        answer, _ = runtime.generate(inputs)
        if injector is not None:
            assert injector.injection_count == 1
        return answer
    finally:
        for handle in handles:
            handle.remove()


def extract_features(runtime, manifest, images, output):
    rows = [sanitized(r) for r in json.loads(manifest.read_text())]
    # Match the original five-shard concatenation order without five worker scripts.
    names = sorted({r['image'] for r in rows})
    groups = [set(names[i::5]) for i in range(5)]
    rows = [r for group in groups for r in rows if r['image'] in group]
    values = {k: [] for k in ('question', 'visual', 'statistics')}
    for i, row in enumerate(rows):
        records = [capture(runtime, runtime.prepare(images, row, blur)) for blur in BLURS]
        for key in values:
            values[key].append(torch.cat([record[key] for record in records]))
        if (i+1) % 25 == 0:
            print(f'extracted {i+1}/{len(rows)}', flush=True)
    torch.save(dict(rows=rows, values={k: torch.stack(v) for k, v in values.items()}), output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    extract_parser = commands.add_parser('extract')
    extract_parser.add_argument('--manifest', type=Path, required=True)
    extract_parser.add_argument('--images', type=Path, required=True)
    extract_parser.add_argument('--output', type=Path, required=True)
    infer_parser = commands.add_parser('infer')
    infer_parser.add_argument('--image', type=Path, required=True)
    infer_parser.add_argument('--question', required=True)
    infer_parser.add_argument('--checkpoint', type=Path, default=HERE/'controller.pt')
    infer_parser.add_argument('--native', action='store_true')
    args = parser.parse_args()
    if args.command == 'extract' and args.output.exists():
        parser.error('output already exists; choose a new feature file')
    runtime = Qwen2Runtime()
    with torch.inference_mode():
        if args.command == 'extract':
            extract_features(runtime, args.manifest, args.images, args.output)
        else:
            injector = None
            if not args.native:
                cp = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
                assert cp['amplitude'] == 1.
                injector = Injector(controller(cp['space'], cp['state']).cuda().eval(),
                                    to_device(cp['space'], 'cuda'))
            inputs = runtime.prepare(args.image.parent, dict(image=args.image.name, text=args.question))
            print(json.dumps(generate(runtime, inputs, injector), ensure_ascii=False))


if __name__ == '__main__':
    main()
