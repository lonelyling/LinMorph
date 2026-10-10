"""Materialize the format-v9 FullMix mapping into Wan2.1-1.3B GDN blocks.

The mapping keeps an exact 3072-hidden/24-head GatedDeltaNet internally and
learns 1536<->3072 boundary projections.  This script reads only the four
self-attention projection matrices needed from each source block, avoiding a
full Wan pipeline/model load on CPU."""
from __future__ import annotations
import sys
from pathlib import Path

# Support running this entry point directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import gc
import json
import os
import time
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from linmorph.materialization import DEFAULT_BLOCKS, parse_blocks, normalize_mapping_state, validate_fullmix_checkpoint, SourceAttention, source_weights_for_block, sha256_file, gibibytes, safetensors_dtype

def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=repo_root
        / "model_weight/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
    )
    parser.add_argument("--mapping-checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--from-congyue-dir",
        type=Path,
        default=repo_root / "examples/wanvideo/model_training",
    )
    parser.add_argument(
        "--blocks",
        type=parse_blocks,
        default=list(DEFAULT_BLOCKS),
    )
    parser.add_argument("--num-threads", type=int, default=32)
    parser.add_argument(
        "--torch-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Compute/output dtype. Mapping training used bfloat16.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate checkpoints/configuration and one mapper without writing weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    training_script_dir = repo_root / "examples/wanvideo/model_training"
    if str(training_script_dir) not in sys.path:
        sys.path.insert(0, str(training_script_dir))
    if args.num_threads < 1:
        raise ValueError("--num-threads must be positive")
    torch.set_num_threads(args.num_threads)
    for path, label in (
        (args.teacher_checkpoint, "teacher checkpoint"),
        (args.mapping_checkpoint, "mapping checkpoint"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    module_path = args.from_congyue_dir / "train_single_layer_linear_adapter.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"mapping implementation does not exist: {module_path}")
    if args.output_path.exists() and not (args.force or args.validate_only):
        raise FileExistsError(
            f"refusing to overwrite existing output without --force: {args.output_path}"
        )

    os.environ.setdefault("DIFFSYNTH_STANDARD_WAN_TEACHER", "1")
    os.environ.setdefault("MAPPING_SAFE_GATED_NORM", "0")
    os.environ.setdefault("MAPPING_SAFE_GDN_RULE", "0")
    sys.path.insert(0, str(args.from_congyue_dir.resolve()))
    from train_single_layer_linear_adapter import MappedGDNWeightTransform

    started = time.time()
    print(f"Loading mapping checkpoint: {args.mapping_checkpoint}", flush=True)
    mapping_state = normalize_mapping_state(load_file(str(args.mapping_checkpoint)))
    training_blocks = validate_fullmix_checkpoint(mapping_state)
    materialize_dtype = (
        torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32
    )
    print(
        "Validated format-v9 FullMix/PostFullResidual mapping: "
        "hidden=3072 heads=24 head_dim=256 expand_v=2 auxiliary=full",
        flush=True,
    )

    output: dict[str, torch.Tensor] = {}
    tensor_names_per_block: set[str] | None = None
    with safe_open(
        str(args.teacher_checkpoint), framework="pt", device="cpu"
    ) as teacher:
        for position, block_index in enumerate(args.blocks, start=1):
            block_started = time.time()
            source_attention = SourceAttention(
                source_weights_for_block(teacher, block_index, materialize_dtype)
            )
            mapper = MappedGDNWeightTransform(
                source_attention=source_attention,
                block_index=block_index,
                training_block_indices=training_blocks,
                low_rank=0,
                full_residual=False,
                post_full_residual=True,
                postmix=False,
                mix_granularity="full",
                mix_order="post",
                source_specific_expansion=False,
                mixing_init="manual_copy",
                gdn_expand_v=2.0,
                gdn_head_dim=256,
                gdn_hidden_size=3072,
                gdn_num_heads=24,
                shape_adapter="subheadmix",
                gdn_use_short_conv=True,
                identity_short_conv=True,
                map_auxiliary_parameters=True,
                seed=42,
            )
            load_result = mapper.load_state_dict(mapping_state, strict=False)
            unexpected = list(load_result.unexpected_keys)
            disallowed_missing = [
                name
                for name in load_result.missing_keys
                if not name.startswith("gdn.")
            ]
            if unexpected or disallowed_missing:
                raise RuntimeError(
                    "mapping checkpoint/module mismatch: "
                    f"unexpected={unexpected[:8]} "
                    f"disallowed_missing={disallowed_missing[:8]}"
                )

            materialized = mapper.materialized_parameters()
            expected_gdn_names = set(dict(mapper.gdn.named_parameters()))
            if set(materialized) != expected_gdn_names:
                raise RuntimeError(
                    f"materialized GDN coverage mismatch at block {block_index}: "
                    f"missing={sorted(expected_gdn_names - set(materialized))} "
                    f"unexpected={sorted(set(materialized) - expected_gdn_names)}"
                )
            block_values = {
                "hidden_input_adapter": mapper.hidden_input_adapter,
                "hidden_output_adapter": mapper.hidden_output_adapter,
                **{f"attn.{name}": value for name, value in materialized.items()},
            }
            if tensor_names_per_block is None:
                tensor_names_per_block = set(block_values)
            elif set(block_values) != tensor_names_per_block:
                raise RuntimeError(
                    f"inconsistent materialized names at block {block_index}"
                )
            for name, value in block_values.items():
                key = f"blocks.{block_index}.self_attn.{name}"
                output[key] = value.detach().cpu().contiguous()

            print(
                f"[{position:02d}/{len(args.blocks):02d}] materialized block "
                f"{block_index:02d}: tensors={len(block_values)} "
                f"bank={gibibytes(output):.2f} GiB "
                f"elapsed={time.time() - block_started:.1f}s",
                flush=True,
            )
            del materialized, mapper, source_attention
            gc.collect()
            if args.validate_only:
                if args.output_path.is_file():
                    expected_keys = {
                        f"blocks.{expected_block}.self_attn.{name}"
                        for expected_block in args.blocks
                        for name in block_values
                    }
                    with safe_open(
                        str(args.output_path), framework="pt", device="cpu"
                    ) as existing:
                        existing_metadata = existing.metadata() or {}
                        actual_keys = set(existing.keys())
                        if actual_keys != expected_keys:
                            raise ValueError(
                                "existing materialized bank key mismatch: "
                                f"missing={sorted(expected_keys - actual_keys)[:8]} "
                                f"unexpected={sorted(actual_keys - expected_keys)[:8]}"
                            )
                        for key in actual_keys:
                            name = key.split(".self_attn.", 1)[1]
                            reference = block_values[name]
                            tensor_slice = existing.get_slice(key)
                            if (
                                tuple(tensor_slice.get_shape())
                                != tuple(reference.shape)
                                or tensor_slice.get_dtype()
                                != safetensors_dtype(reference.dtype)
                            ):
                                raise ValueError(
                                    f"existing materialized bank spec mismatch for {key}"
                                )
                    current_mapping_sha256 = sha256_file(args.mapping_checkpoint)
                    if existing_metadata.get("source_mapping_sha256") != current_mapping_sha256:
                        raise ValueError(
                            "existing materialized bank was produced from a "
                            "different mapping checkpoint: "
                            f"stored={existing_metadata.get('source_mapping_sha256')} "
                            f"current={current_mapping_sha256}"
                        )
                    print(
                        "Validated existing materialized bank: "
                        f"blocks={len(args.blocks)} tensors={len(expected_keys)}"
                    )
                print("Validation-only mode completed after the first block.")
                return

    expected_tensor_count = len(args.blocks) * len(tensor_names_per_block or ())
    if len(output) != expected_tensor_count:
        raise RuntimeError(
            f"output tensor count mismatch: {len(output)} != {expected_tensor_count}"
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output_path.with_name(
        f".{args.output_path.name}.{os.getpid()}.tmp"
    )
    mapping_sha256 = sha256_file(args.mapping_checkpoint)
    metadata = {
        "format": "wan21_13b_projected_gdn_init_v1",
        "source_teacher": str(args.teacher_checkpoint.resolve()),
        "source_teacher_size": args.teacher_checkpoint.stat().st_size,
        "source_mapping": str(args.mapping_checkpoint.resolve()),
        "source_mapping_sha256": mapping_sha256,
        "blocks": args.blocks,
        "gdn_hidden_size": 3072,
        "gdn_num_heads": 24,
        "gdn_head_dim": 256,
        "gdn_expand_v": 2.0,
        "torch_dtype": args.torch_dtype,
        "mapping_scheme": "FullMix/PostMix/PostFullResidual",
        "tensor_count": len(output),
        "tensors_per_block": len(tensor_names_per_block or ()),
        "size_gib": gibibytes(output),
        "elapsed_seconds_before_save": time.time() - started,
    }
    try:
        save_file(
            output,
            str(temporary_path),
            metadata={
                "format": metadata["format"],
                "blocks": ",".join(map(str, args.blocks)),
                "source_mapping_sha256": mapping_sha256,
                "mapping_scheme": metadata["mapping_scheme"],
            },
        )
        os.replace(temporary_path, args.output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    metadata["elapsed_seconds_total"] = time.time() - started
    metadata_path = args.output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Saved {len(args.blocks)}-block FullMix GDN bank: {args.output_path}",
        flush=True,
    )
    print(
        f"Tensors={len(output)} size={gibibytes(output):.2f} GiB "
        f"metadata={metadata_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
