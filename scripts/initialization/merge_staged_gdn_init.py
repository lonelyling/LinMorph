#!/usr/bin/env python3
"""Subset a mapped GDN bank and optionally overlay previously trained blocks."""

from __future__ import annotations
import sys
from pathlib import Path

# Support running this entry point directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import argparse
from contextlib import ExitStack
import json
import os
import re
from safetensors import safe_open
from safetensors.torch import save_file


GDN_KEY_PATTERN = re.compile(r"^blocks\.(\d+)\.self_attn\.(.+)$")


def parse_block_indices(value: str) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise argparse.ArgumentTypeError("block list cannot be empty")
    if len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError(f"block list contains duplicates: {indices}")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a staged GDN initialization by overlaying trained source "
            "blocks onto a complete mapped target initialization."
        )
    )
    parser.add_argument("--base-init", type=Path, required=True)
    parser.add_argument("--overlay-checkpoint", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--target-blocks", type=parse_block_indices, required=True)
    parser.add_argument("--overlay-blocks", type=parse_block_indices)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate tensor coverage/shapes without writing the output file.",
    )
    return parser.parse_args()


def checkpoint_keys(path: Path) -> list[str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return list(handle.keys())


def keys_by_block(
    keys: list[str], checkpoint_name: str, *, ignore_other_keys: bool = False
) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    invalid = []
    for key in keys:
        match = GDN_KEY_PATTERN.match(key)
        if match is None:
            invalid.append(key)
            continue
        block_index = int(match.group(1))
        result.setdefault(block_index, set()).add(match.group(2))
    if invalid and not ignore_other_keys:
        raise ValueError(
            f"{checkpoint_name} contains non-GDN keys: "
            f"count={len(invalid)} sample={invalid[:8]}"
        )
    return result


def validate_block_coverage(
    keys: list[str], expected_blocks: list[int], checkpoint_name: str,
    *, allow_superset: bool = False, ignore_other_keys: bool = False,
) -> dict[int, set[str]]:
    grouped = keys_by_block(
        keys, checkpoint_name, ignore_other_keys=ignore_other_keys
    )
    actual_blocks = set(grouped)
    expected = set(expected_blocks)
    coverage_ok = expected.issubset(actual_blocks) if allow_superset else actual_blocks == expected
    if not coverage_ok:
        raise ValueError(
            f"{checkpoint_name} block coverage mismatch: "
            f"missing={sorted(expected - actual_blocks)} "
            f"unexpected={sorted(actual_blocks - expected)}"
        )
    reference_block = expected_blocks[0]
    reference_names = grouped[reference_block]
    for block_index in expected_blocks[1:]:
        if grouped[block_index] != reference_names:
            raise ValueError(
                f"{checkpoint_name} has inconsistent tensors at block {block_index}: "
                f"missing={sorted(reference_names - grouped[block_index])} "
                f"unexpected={sorted(grouped[block_index] - reference_names)}"
            )
    return grouped


def main() -> None:
    args = parse_args()
    if (args.overlay_checkpoint is None) != (args.overlay_blocks is None):
        raise ValueError(
            "--overlay-checkpoint and --overlay-blocks must be supplied together"
        )
    for path, label in ((args.base_init, "base initialization"),):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if args.overlay_checkpoint is not None and not args.overlay_checkpoint.is_file():
        raise FileNotFoundError(
            f"overlay checkpoint does not exist: {args.overlay_checkpoint}"
        )
    overlay_blocks = args.overlay_blocks or []
    if not set(overlay_blocks).issubset(args.target_blocks):
        raise ValueError(
            "overlay blocks must be a subset of target blocks: "
            f"overlay={overlay_blocks} target={args.target_blocks}"
        )
    if args.output_path.exists() and not args.validate_only:
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_path}")

    base_keys = checkpoint_keys(args.base_init)
    overlay_keys = (
        checkpoint_keys(args.overlay_checkpoint)
        if args.overlay_checkpoint is not None
        else []
    )
    base_grouped = validate_block_coverage(
        base_keys,
        args.target_blocks,
        "base initialization",
        allow_superset=True,
    )
    overlay_grouped = (
        validate_block_coverage(
            overlay_keys,
            overlay_blocks,
            "overlay checkpoint",
            allow_superset=True,
            ignore_other_keys=True,
        )
        if overlay_blocks
        else {}
    )
    for block_index in overlay_blocks:
        if overlay_grouped[block_index] != base_grouped[block_index]:
            raise ValueError(
                f"overlay tensor names do not match base block {block_index}: "
                f"missing={sorted(base_grouped[block_index] - overlay_grouped[block_index])} "
                f"unexpected={sorted(overlay_grouped[block_index] - base_grouped[block_index])}"
            )

    target_block_set = set(args.target_blocks)
    overlay_block_set = set(overlay_blocks)
    selected_base_keys = [
        key
        for key in base_keys
        if (match := GDN_KEY_PATTERN.match(key)) is not None
        and int(match.group(1)) in target_block_set
    ]
    selected_overlay_keys = [
        key
        for key in overlay_keys
        if (match := GDN_KEY_PATTERN.match(key)) is not None
        and int(match.group(1)) in overlay_block_set
    ]
    if args.validate_only:
        output_tensor_count = None
        if args.output_path.is_file():
            output_keys = checkpoint_keys(args.output_path)
            output_grouped = validate_block_coverage(
                output_keys,
                args.target_blocks,
                "existing staged output",
            )
            for block_index in args.target_blocks:
                if output_grouped[block_index] != base_grouped[block_index]:
                    raise ValueError(
                        "existing staged output tensor names do not match "
                        f"base block {block_index}"
                    )
            with ExitStack() as stack:
                base_handle = stack.enter_context(
                    safe_open(str(args.base_init), framework="pt", device="cpu")
                )
                output_handle = stack.enter_context(
                    safe_open(str(args.output_path), framework="pt", device="cpu")
                )
                overlay_handle = (
                    stack.enter_context(
                        safe_open(
                            str(args.overlay_checkpoint),
                            framework="pt",
                            device="cpu",
                        )
                    )
                    if args.overlay_checkpoint is not None
                    else None
                )
                for key in output_keys:
                    match = GDN_KEY_PATTERN.match(key)
                    block_index = int(match.group(1))
                    source_handle = (
                        overlay_handle
                        if block_index in overlay_block_set
                        else base_handle
                    )
                    source_slice = source_handle.get_slice(key)
                    output_slice = output_handle.get_slice(key)
                    if (
                        source_slice.get_shape() != output_slice.get_shape()
                        or source_slice.get_dtype() != output_slice.get_dtype()
                    ):
                        raise ValueError(
                            f"existing staged output spec mismatch for {key}: "
                            f"source=({source_slice.get_shape()}, {source_slice.get_dtype()}) "
                            f"output=({output_slice.get_shape()}, {output_slice.get_dtype()})"
                        )
            output_tensor_count = len(output_keys)
        print(
            f"Validated staged {len(args.target_blocks)}-GDN initialization: "
            f"base_tensors={len(selected_base_keys)} "
            f"overlay_tensors={len(selected_overlay_keys)} "
            f"existing_output_tensors={output_tensor_count or '<not written>'}"
        )
        return

    with safe_open(str(args.base_init), framework="pt", device="cpu") as handle:
        merged = {key: handle.get_tensor(key) for key in selected_base_keys}
    if args.overlay_checkpoint is not None:
        with safe_open(
            str(args.overlay_checkpoint), framework="pt", device="cpu"
        ) as handle:
            for key in selected_overlay_keys:
                overlay_tensor = handle.get_tensor(key)
                base_tensor = merged[key]
                if overlay_tensor.shape != base_tensor.shape:
                    raise ValueError(
                        f"shape mismatch for {key}: "
                        f"base={tuple(base_tensor.shape)} overlay={tuple(overlay_tensor.shape)}"
                    )
                if overlay_tensor.dtype != base_tensor.dtype:
                    raise ValueError(
                        f"dtype mismatch for {key}: "
                        f"base={base_tensor.dtype} overlay={overlay_tensor.dtype}"
                    )
                merged[key] = overlay_tensor

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output_path.with_name(
        f".{args.output_path.name}.{os.getpid()}.tmp"
    )
    try:
        save_file(
            merged,
            str(temporary_path),
            metadata={
                "base_init": str(args.base_init.resolve()),
                "overlay_checkpoint": (
                    str(args.overlay_checkpoint.resolve())
                    if args.overlay_checkpoint is not None
                    else ""
                ),
                "target_blocks": ",".join(map(str, args.target_blocks)),
                "overlay_blocks": ",".join(map(str, overlay_blocks)),
            },
        )
        os.replace(temporary_path, args.output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    metadata = {
        "base_init": str(args.base_init.resolve()),
        "overlay_checkpoint": (
            str(args.overlay_checkpoint.resolve())
            if args.overlay_checkpoint is not None
            else None
        ),
        "output_path": str(args.output_path.resolve()),
        "target_blocks": args.target_blocks,
        "overlay_blocks": overlay_blocks,
        "mapped_blocks": sorted(set(args.target_blocks) - set(overlay_blocks)),
        "tensor_count": len(merged),
        "overlay_tensor_count": len(selected_overlay_keys),
    }
    metadata_path = args.output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Saved staged {len(args.target_blocks)}-GDN initialization: "
        f"{args.output_path}"
    )
    print(
        f"Overlayed {len(overlay_blocks)} trained blocks "
        f"({len(selected_overlay_keys)} tensors); retained mapping initialization for "
        f"{len(args.target_blocks) - len(overlay_blocks)} new blocks."
    )
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
