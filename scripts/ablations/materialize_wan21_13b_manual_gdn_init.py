#!/usr/bin/env python3
"""Materialize the existing exact-large Manual Copy evaluation baseline."""

import sys
from pathlib import Path

# Support running this entry point directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import argparse
import gc
import json
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from linmorph.materialization import SourceAttention, parse_blocks, source_weights_for_block


def main():
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--blocks", type=parse_blocks, default=parse_blocks("0,4,8,12,16,20,24"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.num_threads < 1:
        parser.error("--num-threads must be positive")
    torch.set_num_threads(args.num_threads)
    for directory in (repo, repo / "examples/wanvideo/model_training"):
        sys.path.insert(0, str(directory))
    os.environ.setdefault("DIFFSYNTH_STANDARD_WAN_TEACHER", "1")
    os.environ.setdefault("MAPPING_SAFE_GATED_NORM", "0")
    os.environ.setdefault("MAPPING_SAFE_GDN_RULE", "0")
    from train_single_layer_linear_adapter import MappedGDNWeightTransform
    from diffsynth.models.wan_video_dit_linear import SelfAttention

    if not args.teacher_checkpoint.is_file():
        raise FileNotFoundError(args.teacher_checkpoint)
    metadata = {
        "format": "wan21_13b_manual_copy_init_v1",
        "source_teacher": str(args.teacher_checkpoint.resolve()),
        "source_teacher_size": args.teacher_checkpoint.stat().st_size,
        "source_teacher_mtime_ns": args.teacher_checkpoint.stat().st_mtime_ns,
        "blocks": ",".join(map(str, args.blocks)),
        "seed": str(args.seed),
        "mapping_scheme": "manual_copy/subheadmix",
    }
    reuse = args.output_path.exists()
    if reuse:
        with safe_open(str(args.output_path), framework="pt", device="cpu") as existing:
            if existing.metadata() != {key: str(value) for key, value in metadata.items()}:
                raise ValueError("Existing manual init provenance differs; choose another output path")

    output = {}
    expected_keys = set()
    # Validate against the real training module, including fp32 recurrence parameters.
    reference = SelfAttention(
        dim=1536, num_heads=12, eps=1e-6, expand_v=2.0, head_dim=256,
        gdn_hidden_size=3072, gdn_num_heads=24,
    ).to(dtype=torch.bfloat16)
    specs = {name: (tuple(value.shape), value.dtype) for name, value in reference.state_dict().items()}
    del reference
    with torch.no_grad(), safe_open(str(args.teacher_checkpoint), framework="pt", device="cpu") as teacher:
        for block in args.blocks:
            mapper = MappedGDNWeightTransform(
                source_attention=SourceAttention(source_weights_for_block(teacher, block, torch.bfloat16)),
                block_index=block, training_block_indices=args.blocks,
                mixing_init="manual_copy", mix_granularity="full", mix_order="post",
                low_rank=0, full_residual=False, post_full_residual=False,
                source_specific_expansion=False, shape_adapter="subheadmix",
                gdn_hidden_size=3072, gdn_num_heads=24, gdn_head_dim=256,
                gdn_expand_v=2.0, gdn_use_short_conv=True, identity_short_conv=True,
                map_auxiliary_parameters=True, seed=args.seed,
            )
            values = {
                "hidden_input_adapter": mapper.hidden_input_adapter,
                "hidden_output_adapter": mapper.hidden_output_adapter,
                **{f"attn.{name}": value for name, value in mapper.materialized_parameters().items()},
            }
            if set(values) != set(specs):
                raise ValueError(f"Incomplete manual init coverage at block {block}")
            for name, value in values.items():
                if (tuple(value.shape), value.dtype) != specs[name] or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid manual init tensor: block={block} name={name}")
                key = f"blocks.{block}.self_attn.{name}"
                expected_keys.add(key)
                if not (reuse or args.validate_only):
                    output[key] = value.detach().cpu().contiguous()
            if reuse:
                with safe_open(str(args.output_path), framework="pt", device="cpu") as existing:
                    for name, value in values.items():
                        key = f"blocks.{block}.self_attn.{name}"
                        if not torch.equal(existing.get_tensor(key), value):
                            raise ValueError(f"Existing manual init differs: {key}")
            print(f"Validated manual-copy block {block}: {len(values)} tensors", flush=True)
            del values, mapper
            gc.collect()
    if reuse:
        with safe_open(str(args.output_path), framework="pt", device="cpu") as existing:
            if set(existing.keys()) != expected_keys:
                raise ValueError("Existing manual init contains unexpected tensors")
        print(f"Reusing validated manual init: {args.output_path}")
    elif not args.validate_only:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output_path.with_name(f".{args.output_path.name}.{os.getpid()}.tmp")
        try:
            save_file(output, str(temporary), metadata={key: str(value) for key, value in metadata.items()})
            os.replace(temporary, args.output_path)
        finally:
            temporary.unlink(missing_ok=True)
        args.output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Saved manual init: {args.output_path}")


if __name__ == "__main__":
    main()
