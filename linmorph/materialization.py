"""Shared materialization implementation; command-line entry points live in scripts/."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace
import torch




DEFAULT_BLOCKS = (
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,
    16, 17, 18, 20, 21, 22, 24, 25, 26,
)


PROJECTION_NAMES = ("q", "k", "v", "o")


MAPPING_PREFIXES = ("mapped_attention.", "module.mapped_attention.")


def parse_blocks(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("block list cannot be empty")
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError(f"duplicate block indices: {result}")
    invalid = [item for item in result if item < 0 or item >= 30]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Wan2.1-1.3B block indices must be in [0, 29]: {invalid}"
        )
    return result


def normalize_mapping_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for name, value in state.items():
        normalized_name = name
        for prefix in MAPPING_PREFIXES:
            if normalized_name.startswith(prefix):
                normalized_name = normalized_name[len(prefix) :]
                break
        if normalized_name in normalized:
            raise ValueError(f"duplicate normalized mapping key: {normalized_name}")
        normalized[normalized_name] = value
    return normalized


def scalar(state: dict[str, torch.Tensor], name: str) -> int | float:
    if name not in state or state[name].numel() != 1:
        raise ValueError(f"missing scalar mapping metadata: {name}")
    return state[name].item()


def validate_fullmix_checkpoint(state: dict[str, torch.Tensor]) -> list[int]:
    expected = {
        "mapping_format_version": 9,
        "mapping_gdn_hidden_size": 3072,
        "mapping_gdn_num_heads": 24,
        "mapping_gdn_head_dim": 256,
        "mapping_mix_granularity": 2,  # full
        "mapping_mix_order": 1,  # post
        "mapping_postmix_enabled": 1,
        "mapping_auxiliary_enabled": 1,
        "mapping_auxiliary_mix_granularity": 2,
        "mapping_low_rank": 0,
        "mapping_full_residual_enabled": 0,
        "mapping_post_full_residual_enabled": 1,
        "mapping_source_specific_expansion": 0,
        "mapping_shape_adapter": 2,  # subheadmix
    }
    mismatches = []
    for name, wanted in expected.items():
        actual = scalar(state, name)
        if actual != wanted:
            mismatches.append((name, actual, wanted))
    expand_v = float(scalar(state, "mapping_gdn_expand_v"))
    if abs(expand_v - 2.0) > 1e-6:
        mismatches.append(("mapping_gdn_expand_v", expand_v, 2.0))
    if mismatches:
        raise ValueError(
            "The supplied checkpoint is not the expected exact-large "
            f"FullMix/PostFullResidual mapping: {mismatches}"
        )
    if "mixing" not in state:
        raise ValueError("mapping checkpoint has no mixing tensor")
    selected = state.get("selected_block_indices")
    if selected is None:
        raise ValueError("mapping checkpoint has no selected_block_indices")
    return [int(value) for value in selected.reshape(-1).tolist()]


class FrozenProjection(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)


class SourceAttention(torch.nn.Module):
    def __init__(self, weights: dict[str, torch.Tensor]):
        super().__init__()
        shapes = {tuple(weight.shape) for weight in weights.values()}
        if shapes != {(1536, 1536)}:
            raise ValueError(f"expected four 1536x1536 projections, got {shapes}")
        dtypes = {weight.dtype for weight in weights.values()}
        if len(dtypes) != 1:
            raise ValueError(f"source projection dtypes disagree: {dtypes}")
        self.dim = 1536
        self.num_heads = 12
        self.norm_q = SimpleNamespace(eps=1e-6)
        for name in PROJECTION_NAMES:
            setattr(self, name, FrozenProjection(weights[name]))


def source_weights_for_block(
    handle, block_index: int, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    available = set(handle.keys())
    result = {}
    for projection in PROJECTION_NAMES:
        suffix = f"blocks.{block_index}.self_attn.{projection}.weight"
        candidates = [name for name in available if name.endswith(suffix)]
        if len(candidates) != 1:
            raise KeyError(
                f"expected exactly one teacher tensor ending in {suffix!r}, "
                f"found {candidates}"
            )
        result[projection] = handle.get_tensor(candidates[0]).to(dtype=dtype)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def gibibytes(tensors: dict[str, torch.Tensor]) -> float:
    size = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    return size / 1024**3


def safetensors_dtype(dtype: torch.dtype) -> str:
    names = {torch.bfloat16: "BF16", torch.float32: "F32"}
    if dtype not in names:
        raise ValueError(f"unsupported output dtype for validation: {dtype}")
    return names[dtype]

