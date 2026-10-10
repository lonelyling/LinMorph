#!/usr/bin/env python3
"""Materialize a compact shared mapping checkpoint into layer-specific GDN weights."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

from diffsynth.core import load_state_dict
from diffsynth.pipelines.pipeline_cross import ModelConfig, WanVideoPipeline

from train_single_layer_weight_mapping import (
    SHAPE_ADAPTER_NAMES,
    TARGET_NAMES,
    MappedGDNWeightTransform,
)


DEFAULT_24_BLOCKS = (
    0, 1, 2, 3,
    5, 6, 7, 8,
    10, 11, 12, 13,
    15, 16, 17, 18,
    20, 21, 22, 23,
    25, 26, 27, 28,
)
MAPPING_PREFIXES = ("mapped_attention.", "module.mapped_attention.")


def parse_bool_or_auto(value):
    normalized = str(value).strip().lower()
    if normalized == "auto":
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected auto/true/false, got {value!r}")


def parse_block_indices(value):
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices or len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError(
            f"Expected unique comma-separated block indices, got {value!r}"
        )
    invalid = [index for index in indices if index < 0]
    if invalid:
        raise argparse.ArgumentTypeError(f"Negative block indices: {invalid}")
    return indices


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit_path", action="append", required=True)
    parser.add_argument("--mapping_checkpoint", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument(
        "--gdn_block_indices",
        type=parse_block_indices,
        default=list(DEFAULT_24_BLOCKS),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--torch_dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--mapping_seed", type=int, default=42)
    parser.add_argument("--low_rank", type=int, default=-1)
    parser.add_argument(
        "--gdn_expand_v",
        type=float,
        default=None,
        help="Target GDN value expansion. Defaults to checkpoint metadata.",
    )
    parser.add_argument(
        "--gdn_head_dim",
        type=int,
        default=None,
        help="Target GDN head dimension. Defaults to checkpoint metadata.",
    )
    parser.add_argument(
        "--mapping_shape_adapter",
        choices=("auto",) + SHAPE_ADAPTER_NAMES,
        default="auto",
    )
    parser.add_argument(
        "--map_auxiliary_parameters",
        type=parse_bool_or_auto,
        default=None,
    )
    return parser.parse_args()


def normalize_mapping_state(state):
    normalized = {}
    for name, value in state.items():
        normalized_name = name
        for prefix in MAPPING_PREFIXES:
            if normalized_name.startswith(prefix):
                normalized_name = normalized_name[len(prefix) :]
                break
        normalized[normalized_name] = value
    return normalized


def compact_mapping_tensors(state):
    return {
        name: value
        for name, value in state.items()
        if name == "mixing"
        or name.startswith("auxiliary_mixing.")
        or name.startswith("low_rank_u.")
        or name.startswith("low_rank_v.")
        or name.startswith("full_rank_residual.")
    }


def infer_low_rank(state):
    metadata = state.get("mapping_low_rank")
    if metadata is not None:
        return int(metadata.item())
    low_rank_tensor = next(
        (value for name, value in state.items() if name.startswith("low_rank_u.")),
        None,
    )
    return int(low_rank_tensor.shape[1]) if low_rank_tensor is not None else 0


def infer_full_rank_residual(state):
    tensor_names = {
        name for name in state if name.startswith("full_rank_residual.")
    }
    has_tensors = bool(tensor_names)
    metadata = state.get("mapping_full_rank_residual")
    if metadata is not None:
        enabled = bool(int(metadata.item()))
        if enabled != has_tensors:
            raise ValueError(
                "Full-rank residual metadata and checkpoint tensors disagree"
            )
    else:
        enabled = has_tensors
    if enabled:
        expected = {f"full_rank_residual.{name}" for name in TARGET_NAMES}
        if tensor_names != expected:
            missing = sorted(expected - tensor_names)
            unexpected = sorted(tensor_names - expected)
            raise ValueError(
                "Incomplete full-rank residual checkpoint: "
                f"missing={missing}, unexpected={unexpected}"
            )
    return enabled


def infer_auxiliary_enabled(state):
    metadata = state.get("mapping_auxiliary_enabled")
    if metadata is not None:
        return bool(int(metadata.item()))
    auxiliary = [
        value
        for name, value in state.items()
        if name.startswith("auxiliary_mixing.")
    ]
    return bool(auxiliary) and any(torch.count_nonzero(value).item() for value in auxiliary)


def checkpoint_training_blocks(state):
    values = state.get("selected_block_indices")
    if values is not None:
        return [int(value) for value in values.reshape(-1).tolist()]
    value = state.get("selected_block_index")
    return [int(value.item())] if value is not None else []


def main():
    args = parse_args()
    if not args.mapping_checkpoint.is_file():
        raise FileNotFoundError(args.mapping_checkpoint)
    dtype = torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float32
    state = normalize_mapping_state(
        load_state_dict(
            str(args.mapping_checkpoint), torch_dtype=None, device="cpu"
        )
    )
    if "mixing" not in state:
        raise KeyError(f"Checkpoint has no compact mapping tensor: {args.mapping_checkpoint}")
    if any(name.startswith("gdn.") for name in state):
        raise ValueError(
            "The checkpoint contains layer-specific materialized GDN parameters. "
            "Use a compact model-step checkpoint, or a format-v3 shared model-final checkpoint."
        )
    version = int(state.get("mapping_format_version", torch.tensor(1)).item())
    if version > 5:
        raise ValueError(f"Unsupported mapping checkpoint format: {version}")
    checkpoint_expand_v = float(
        state.get("mapping_gdn_expand_v", torch.tensor(1.0)).item()
    )
    checkpoint_head_dim = int(
        state.get("mapping_gdn_head_dim", torch.tensor(0)).item()
    )
    checkpoint_adapter_id = int(
        state.get("mapping_shape_adapter", torch.tensor(0)).item()
    )
    if checkpoint_adapter_id < 0 or checkpoint_adapter_id >= len(SHAPE_ADAPTER_NAMES):
        raise ValueError(
            f"Unsupported mapping shape adapter id: {checkpoint_adapter_id}"
        )
    checkpoint_adapter = SHAPE_ADAPTER_NAMES[checkpoint_adapter_id]
    gdn_expand_v = (
        checkpoint_expand_v if args.gdn_expand_v is None else args.gdn_expand_v
    )
    gdn_head_dim = (
        checkpoint_head_dim if args.gdn_head_dim is None else args.gdn_head_dim
    )
    shape_adapter = (
        checkpoint_adapter
        if args.mapping_shape_adapter == "auto"
        else args.mapping_shape_adapter
    )
    requested_architecture = (float(gdn_expand_v), int(gdn_head_dim), shape_adapter)
    checkpoint_architecture = (
        checkpoint_expand_v,
        checkpoint_head_dim,
        checkpoint_adapter,
    )
    if version >= 4 and requested_architecture != checkpoint_architecture:
        raise ValueError(
            "Requested GDN mapping architecture does not match checkpoint metadata: "
            f"requested={requested_architecture}, checkpoint={checkpoint_architecture}"
        )

    checkpoint_rank = infer_low_rank(state)
    low_rank = checkpoint_rank if args.low_rank < 0 else args.low_rank
    if low_rank != checkpoint_rank:
        raise ValueError(
            f"Requested low_rank={low_rank}, checkpoint contains rank {checkpoint_rank}"
        )
    full_rank_residual = infer_full_rank_residual(state)
    if low_rank > 0 and full_rank_residual:
        raise ValueError(
            "Mapping checkpoint cannot contain both low-rank and full-rank residuals"
        )
    inferred_auxiliary = infer_auxiliary_enabled(state)
    map_auxiliary = (
        inferred_auxiliary
        if args.map_auxiliary_parameters is None
        else args.map_auxiliary_parameters
    )
    mapping_tensors = compact_mapping_tensors(state)

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=args.device,
        model_configs=[ModelConfig(path=args.dit_path)],
        tokenizer_config=None,
    )
    if pipe.dit is None:
        raise RuntimeError("The supplied paths did not load a Wan DiT")
    num_layers = len(pipe.dit.blocks)
    invalid = [index for index in args.gdn_block_indices if index >= num_layers]
    if invalid:
        raise ValueError(f"Target blocks outside [0, {num_layers - 1}]: {invalid}")

    output_state = {}
    expected_parameter_names = None
    for position, block_index in enumerate(args.gdn_block_indices, start=1):
        mapper = MappedGDNWeightTransform(
            source_attention=pipe.dit.blocks[block_index].self_attn,
            block_index=block_index,
            training_block_indices=checkpoint_training_blocks(state),
            low_rank=low_rank,
            full_rank_residual=full_rank_residual,
            gdn_expand_v=gdn_expand_v,
            gdn_head_dim=gdn_head_dim,
            shape_adapter=shape_adapter,
            map_auxiliary_parameters=map_auxiliary,
            capture_random_initialization=False,
            seed=args.mapping_seed,
        )
        load_result = mapper.load_state_dict(mapping_tensors, strict=False)
        if load_result.unexpected_keys:
            raise ValueError(
                f"Unsupported mapping tensors: {load_result.unexpected_keys[:8]}"
            )
        with torch.no_grad():
            materialized = mapper.materialized_parameters()
        parameter_names = set(materialized)
        if expected_parameter_names is None:
            expected_parameter_names = parameter_names
        elif parameter_names != expected_parameter_names:
            raise RuntimeError(
                f"Inconsistent materialized parameters at block {block_index}"
            )
        for parameter_name, value in materialized.items():
            output_name = (
                f"blocks.{block_index}.self_attn.attn.{parameter_name}"
            )
            output_state[output_name] = value.detach().to("cpu").contiguous()
        print(
            f"[{position}/{len(args.gdn_block_indices)}] materialized block "
            f"{block_index}: {len(materialized)} GDN tensors",
            flush=True,
        )
        del mapper, materialized

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(output_state, str(args.output_path))
    metadata = {
        "mapping_checkpoint": str(args.mapping_checkpoint.resolve()),
        "mapping_format_version": version,
        "mapping_training_blocks": checkpoint_training_blocks(state),
        "target_gdn_blocks": args.gdn_block_indices,
        "low_rank": low_rank,
        "full_rank_residual": full_rank_residual,
        "gdn_expand_v": gdn_expand_v,
        "gdn_head_dim": gdn_head_dim,
        "mapping_shape_adapter": shape_adapter,
        "map_auxiliary_parameters": map_auxiliary,
        "tensor_count": len(output_state),
        "parameter_names_per_block": sorted(expected_parameter_names or []),
        "output_path": str(args.output_path.resolve()),
    }
    metadata_path = args.output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Saved {len(args.gdn_block_indices)}-layer mapped GDN initialization: "
        f"{args.output_path}"
    )
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
