#!/usr/bin/env python3
"""Build Wan2.2-5B Manual Copy GDN weights without loading the entire DiT."""

import sys
from pathlib import Path

# Support running this entry point directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from contextlib import ExitStack
import gc
import json
import os
from types import SimpleNamespace
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_blocks(value):
    blocks = [int(x) for x in value.split(',')]
    if not blocks or len(set(blocks)) != len(blocks) or any(x < 0 or x >= 30 for x in blocks):
        raise argparse.ArgumentTypeError('Expected unique block indices in [0, 29]')
    return blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--blocks', type=parse_blocks, default=parse_blocks('0,5,10,15,20,25'))
    parser.add_argument('--num-threads', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.num_threads < 1:
        parser.error('--num-threads must be positive')
    torch.set_num_threads(args.num_threads)
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / 'examples/wanvideo/model_training'))
    from train_single_layer_weight_mapping import MappedGDNWeightTransform
    from diffsynth.models.wan_video_dit_linear import SelfAttention

    shards = sorted(args.model_dir.glob('diffusion_pytorch_model*.safetensors'))
    if not shards:
        raise FileNotFoundError(f'No DiT shards under {args.model_dir}')
    config = json.loads((args.model_dir / 'config.json').read_text())
    if (config['dim'], config['num_heads'], config['num_layers']) != (3072, 24, 30):
        raise ValueError('Expected Wan2.2-5B: dim=3072, heads=24, layers=30')
    provenance = json.dumps({
        'format': 'wan22_5b_manual_copy_repeat_init_v1', 'blocks': args.blocks,
        'seed': args.seed, 'expand_v': 2.0, 'head_dim': 256,
        'teachers': [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in shards],
    }, sort_keys=True)
    metadata = {'provenance': provenance}
    reuse = args.output_path.exists()
    if reuse:
        with safe_open(str(args.output_path), framework='pt', device='cpu') as existing:
            if existing.metadata() != metadata:
                raise ValueError('Existing init provenance differs; choose another output path')
    reference = SelfAttention(dim=3072, num_heads=24, eps=1e-6, expand_v=2.0, head_dim=256).to(dtype=torch.bfloat16)
    specs = {name: (tuple(value.shape), value.dtype) for name, value in reference.state_dict().items()}
    del reference
    output = {}
    expected_keys = set()
    with torch.no_grad(), ExitStack() as stack:
        handles = [stack.enter_context(safe_open(str(p), framework='pt', device='cpu')) for p in shards]
        lookup = {}
        for handle in handles:
            for key in handle.keys():
                if key in lookup:
                    raise ValueError(f'Duplicate teacher tensor: {key}')
                lookup[key] = handle
        for block in args.blocks:
            projections = {}
            for name in ('q', 'k', 'v', 'o'):
                suffix = f'blocks.{block}.self_attn.{name}.weight'
                matches = [key for key in lookup if key.endswith(suffix)]
                if len(matches) != 1:
                    raise KeyError(f'Expected exactly one teacher projection: {suffix}')
                key = matches[0]
                value = lookup[key].get_tensor(key).to(dtype=torch.bfloat16)
                if value.shape != (3072, 3072):
                    raise ValueError(f'Unexpected teacher projection shape: {key}')
                projections[name] = SimpleNamespace(weight=value)
            source = SimpleNamespace(dim=3072, num_heads=24, norm_q=SimpleNamespace(eps=1e-6), **projections)
            mapper = MappedGDNWeightTransform(
                source_attention=source, block_index=block, training_block_indices=args.blocks,
                mixing_init='manual_copy', low_rank=0, full_rank_residual=False,
                gdn_expand_v=2.0, gdn_head_dim=256, shape_adapter='repeat',
                gdn_use_short_conv=True, identity_short_conv=True,
                map_auxiliary_parameters=True, seed=args.seed,
            )
            values = {f'attn.{name}': value for name, value in mapper.materialized_parameters().items()}
            if set(values) != set(specs):
                raise ValueError(f'Incomplete training parameter coverage at block {block}')
            for name, value in values.items():
                if (tuple(value.shape), value.dtype) != specs[name] or not torch.isfinite(value).all():
                    raise ValueError(f'Invalid manual parameter at block {block}: {name}')
                key = f'blocks.{block}.self_attn.{name}'
                expected_keys.add(key)
                if not (reuse or args.validate_only):
                    output[key] = value.detach().cpu().contiguous()
            if reuse:
                with safe_open(str(args.output_path), framework='pt', device='cpu') as existing:
                    for name, value in values.items():
                        if not torch.equal(existing.get_tensor(f'blocks.{block}.self_attn.{name}'), value):
                            raise ValueError(f'Existing init differs: block={block} name={name}')
            print(f'Validated Wan2.2 manual-copy block {block}: {len(values)} tensors', flush=True)
            del values, mapper, source, projections
            gc.collect()
    if reuse:
        with safe_open(str(args.output_path), framework='pt', device='cpu') as existing:
            if set(existing.keys()) != expected_keys:
                raise ValueError('Existing init contains unexpected tensors')
        print(f'Reusing validated init: {args.output_path}')
    elif not args.validate_only:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_path.with_name(f'.{args.output_path.name}.{os.getpid()}.tmp')
        try:
            save_file(output, str(temporary), metadata=metadata)
            os.replace(temporary, args.output_path)
        finally:
            temporary.unlink(missing_ok=True)
        args.output_path.with_suffix('.json').write_text(json.dumps(json.loads(provenance), indent=2) + '\n')
        print(f'Saved Wan2.2 manual init: {args.output_path}')


if __name__ == '__main__':
    main()
