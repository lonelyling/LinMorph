"""Shared vbench implementation; command-line entry points live in scripts/."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from .sampling import DEFAULT_UNIPC_SOURCE, DEFAULT_UNIPC_DEPS, DEFAULT_NEGATIVE_PROMPT, load_unipc_class, DiffSynthUniPCAdapter

SCRIPT_DIR = Path(__file__).resolve().parent


PROJECT_ROOT = SCRIPT_DIR.parent


DEFAULT_MODEL_DIR = Path(os.environ.get("LINMORPH_MODEL_DIR", str(PROJECT_ROOT / "model_weight" / "Wan2.1-T2V-1.3B")))


DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "output/train/wan21_13B_warmup500_fullstate_8gpu_ga2/stage2_mapping_full_finetune"
)


DEFAULT_OUTPUT = PROJECT_ROOT / "output/vbench_standard/wan21_13b_stage2/raw_videos"


DEFAULT_FULL_INFO = (
    PROJECT_ROOT
    / ".deps/VBench/vbench/VBench_full_info.json"
)


DEFAULT_AUGMENTED_PROMPTS = (
    PROJECT_ROOT
    / ".cache/vbench/official_protocol/all_dimension_aug_wanx_seed42.txt"
)






ALL_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "appearance_style",
    "temporal_style",
    "overall_consistency",
)


DEFAULT_GDN_BLOCKS = "0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26"




@dataclass(frozen=True)
class PromptRecord:
    original_prompt: str
    inference_prompt: str
    dimensions: tuple[str, ...]
    metadata_indices: tuple[int, ...]


@dataclass(frozen=True)
class GenerationTask:
    ordinal: int
    original_prompt: str
    inference_prompt: str
    dimensions: tuple[str, ...]
    sample_index: int
    seed: int
    output_path: str


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def load_prompt_records(args: argparse.Namespace) -> list[PromptRecord]:
    full_info_path = resolve(args.full_info)
    with full_info_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, list) or len(metadata) != 946:
        raise ValueError(f"Expected the official 946-record VBench metadata: {full_info_path}")

    if args.prompt_source == "wan-enhanced":
        augmented_path = resolve(args.augmented_prompts)
        if not augmented_path.is_file():
            raise FileNotFoundError(
                f"Official Wan enhanced prompts are missing: {augmented_path}\n"
                "Run: bash scripts/evaluation/vbench/setup_vbench_standard.sh"
            )
        augmented = augmented_path.read_text(encoding="utf-8").splitlines()
        if len(augmented) != len(metadata):
            raise ValueError(
                f"Enhanced prompt count {len(augmented)} does not match metadata count {len(metadata)}"
            )
    else:
        augmented = [str(row["prompt_en"]) for row in metadata]

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, (row, inference_prompt) in enumerate(zip(metadata, augmented)):
        original = str(row["prompt_en"])
        if "/" in original or "\x00" in original:
            raise ValueError(f"Official prompt cannot be used as a filename: {original!r}")
        dimensions = tuple(str(value) for value in row["dimension"])
        if original not in merged:
            merged[original] = {
                "inference_prompt": inference_prompt,
                "dimensions": list(dimensions),
                "indices": [index],
            }
            order.append(original)
        else:
            previous = merged[original]
            if previous["inference_prompt"] != inference_prompt:
                raise ValueError(f"Duplicate official prompt has different enhanced text: {original}")
            previous["dimensions"].extend(
                value for value in dimensions if value not in previous["dimensions"]
            )
            previous["indices"].append(index)

    selected = set(args.dimensions)
    records = []
    for original in order:
        row = merged[original]
        dimensions = tuple(row["dimensions"])
        if not selected.intersection(dimensions):
            continue
        records.append(
            PromptRecord(
                original_prompt=original,
                inference_prompt=str(row["inference_prompt"]),
                dimensions=dimensions,
                metadata_indices=tuple(row["indices"]),
            )
        )
    return records


def deterministic_seed(base_seed: int, prompt: str, sample_index: int) -> int:
    digest = hashlib.sha256(
        f"vbench-standard-v1\0{base_seed}\0{prompt}\0{sample_index}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def select_diagnostic_records(
    records: list[PromptRecord],
    dimensions: list[str],
    prompts_per_dimension: int,
    diagnostic_prompt_count: int = 0,
) -> list[PromptRecord]:
    """Deterministically cover every requested dimension with few official prompts.

    Some VBench prompts belong to several dimensions. Selecting the candidate that
    covers the most still-underrepresented dimensions keeps the diagnostic small,
    while original metadata order provides a stable tie-breaker.
    """
    if prompts_per_dimension == 0:
        if diagnostic_prompt_count:
            raise ValueError(
                "--diagnostic-prompt-count requires --prompts-per-dimension > 0"
            )
        return records
    if prompts_per_dimension < 0:
        raise ValueError("--prompts-per-dimension must be non-negative")
    if diagnostic_prompt_count < 0:
        raise ValueError("--diagnostic-prompt-count must be non-negative")

    requested = tuple(dict.fromkeys(dimensions))
    counts = {dimension: 0 for dimension in requested}
    remaining = list(records)
    selected: list[PromptRecord] = []
    while any(count < prompts_per_dimension for count in counts.values()):
        best_index = -1
        best_gain = 0
        for index, record in enumerate(remaining):
            gain = sum(
                dimension in record.dimensions and counts[dimension] < prompts_per_dimension
                for dimension in requested
            )
            if gain > best_gain:
                best_index = index
                best_gain = gain
        if best_index < 0:
            missing = {
                dimension: prompts_per_dimension - count
                for dimension, count in counts.items()
                if count < prompts_per_dimension
            }
            raise ValueError(f"Official metadata cannot satisfy diagnostic coverage: {missing}")
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        for dimension in requested:
            if dimension in chosen.dimensions and counts[dimension] < prompts_per_dimension:
                counts[dimension] += 1

    if diagnostic_prompt_count:
        if diagnostic_prompt_count < len(selected):
            raise ValueError(
                "--diagnostic-prompt-count is smaller than the minimum official-prompt "
                f"cover: requested={diagnostic_prompt_count}, minimum={len(selected)}"
            )
        if diagnostic_prompt_count > len(records):
            raise ValueError(
                "--diagnostic-prompt-count exceeds the available official prompts: "
                f"requested={diagnostic_prompt_count}, available={len(records)}"
            )
        selected.extend(remaining[: diagnostic_prompt_count - len(selected)])
    return selected


def write_subset_full_info(
    source_full_info: Path, records: list[PromptRecord], destination: Path
) -> None:
    with source_full_info.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    selected_indices = {
        index for record in records for index in record.metadata_indices
    }
    subset = [row for index, row in enumerate(metadata) if index in selected_indices]
    if not subset:
        raise ValueError("Refusing to write an empty VBench diagnostic subset")
    write_json(destination, subset)


def build_tasks(args: argparse.Namespace, records: list[PromptRecord], output: Path) -> list[GenerationTask]:
    diagnostic_subset = args.prompts_per_dimension > 0
    if diagnostic_subset:
        if args.samples_per_prompt < 1:
            raise ValueError("Diagnostic subsets require --samples-per-prompt >= 1")
        if args.temporal_flicker_candidates < 1:
            raise ValueError("Diagnostic subsets require --temporal-flicker-candidates >= 1")
    else:
        if args.samples_per_prompt != 5:
            raise ValueError("Official VBench standard requires --samples-per-prompt 5")
        if args.temporal_flicker_candidates < 5:
            raise ValueError("--temporal-flicker-candidates must be at least 5")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")

    all_tasks: list[GenerationTask] = []
    ordinal = 0
    for record in records:
        sample_count = (
            args.temporal_flicker_candidates
            if "temporal_flickering" in record.dimensions
            and "temporal_flickering" in args.dimensions
            else args.samples_per_prompt
        )
        for sample_index in range(sample_count):
            path = output / f"{record.original_prompt}-{sample_index}.mp4"
            all_tasks.append(
                GenerationTask(
                    ordinal=ordinal,
                    original_prompt=record.original_prompt,
                    inference_prompt=record.inference_prompt,
                    dimensions=record.dimensions,
                    sample_index=sample_index,
                    seed=deterministic_seed(args.base_seed, record.original_prompt, sample_index),
                    output_path=str(path),
                )
            )
            ordinal += 1

    if args.min_sample_index < 0:
        raise ValueError("--min-sample-index must be nonnegative")
    shard = [
        task for task in all_tasks
        if task.ordinal % args.num_shards == args.shard_index
        and task.sample_index >= args.min_sample_index
    ]
    if args.max_tasks > 0:
        shard = shard[: args.max_tasks]
    return shard






def validate_official_sampling_args(args: argparse.Namespace) -> None:
    required = {
        "height": (args.height, 480),
        "width": (args.width, 832),
        "num_frames": (args.num_frames, 81),
        "steps": (args.steps, 50),
        "cfg_scale": (args.cfg_scale, 6.0),
        "flow_shift": (args.flow_shift, 8.0),
    }
    mismatches = [f"{key}={actual} (required {expected})" for key, (actual, expected) in required.items() if actual != expected]
    if mismatches:
        raise ValueError("ReHyAt-comparable sampling settings changed: " + ", ".join(mismatches))

