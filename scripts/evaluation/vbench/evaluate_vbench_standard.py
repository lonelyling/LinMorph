#!/usr/bin/env python3
"""Strict evaluator for the official 16-dimension VBench standard protocol."""

from __future__ import annotations
import sys
from pathlib import Path

# Support running this entry point directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


import argparse
import fcntl
import gc
import importlib
import json
import os
import re
import traceback
from datetime import datetime, timezone
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache/vbench"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "output/vbench_standard/evaluation"

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
QUALITY_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
    "dynamic_degree",
)
SEMANTIC_DIMENSIONS = (
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
DIMENSION_WEIGHTS = {dimension: 1.0 for dimension in ALL_DIMENSIONS}
DIMENSION_WEIGHTS["dynamic_degree"] = 0.5
NORMALIZATION = {
    "subject_consistency": (0.1462, 1.0),
    "background_consistency": (0.2615, 1.0),
    "temporal_flickering": (0.6293, 1.0),
    "motion_smoothness": (0.7060, 0.9975),
    "dynamic_degree": (0.0, 1.0),
    "aesthetic_quality": (0.0, 1.0),
    "imaging_quality": (0.0, 1.0),
    "object_class": (0.0, 1.0),
    "multiple_objects": (0.0, 1.0),
    "human_action": (0.0, 1.0),
    "color": (0.0, 1.0),
    "spatial_relationship": (0.0, 1.0),
    "scene": (0.0, 0.8222),
    "appearance_style": (0.0009, 0.2855),
    "temporal_style": (0.0, 0.3640),
    "overall_consistency": (0.0, 0.3640),
}
REHYAT_REFERENCE = {
    "model": "Wan2.1 1.3B* + ReHyAt (15x Tc=3, To=1)",
    "total": 83.79,
    "quality": 84.57,
    "semantic": 80.70,
}
MEDIA_SUFFIXES = (".mp4", ".gif")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Evaluate videos using VBench vbench_standard. Unlike custom_input, "
            "this command validates all official prompt filenames and five samples per prompt."
        ),
    )
    parser.add_argument("videos", type=Path, help="Flat directory containing official prompt-index filenames.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--name", default="vbench_standard")
    parser.add_argument("--reference", choices=("rehyat", "none"), default="rehyat",
                        help="Use none for models outside the Wan2.1 ReHyAt comparison.")
    parser.add_argument("--dimensions", nargs="+", choices=ALL_DIMENSIONS, default=list(ALL_DIMENSIONS))
    parser.add_argument(
        "--temporal-filtered-dir",
        type=Path,
        default=None,
        help="VBench static_filter output/filtered_videos; used for temporal-flickering prompts.",
    )
    parser.add_argument(
        "--assume-temporal-filtered",
        action="store_true",
        help="Assert that videos/ already contains the final selected samples for flickering prompts.",
    )
    parser.add_argument("--full-info", type=Path, default=None)
    parser.add_argument(
        "--subset-diagnostic",
        action="store_true",
        help=(
            "Evaluate a reduced metadata file made only from official VBench prompts. "
            "This permits fewer than 946 metadata rows and never reports an official score."
        ),
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=5,
        help="Expected samples per prompt; values other than five require --subset-diagnostic.",
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--imaging-quality-preprocessing-mode",
        choices=("shorter", "longer", "shorter_centercrop", "None"),
        default="longer",
    )
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Permit diagnostic subset evaluation; no official aggregate score will be reported.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def safe_name(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return value or "vbench_standard"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def load_metadata(full_info: Path, allow_subset: bool) -> list[dict[str, Any]]:
    with full_info.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a non-empty VBench metadata list: {full_info}")
    if not allow_subset and len(payload) != 946:
        raise ValueError(f"Expected official VBench metadata with 946 records: {full_info}")
    for index, row in enumerate(payload):
        if not isinstance(row, dict) or not isinstance(row.get("prompt_en"), str):
            raise ValueError(f"Malformed VBench metadata row {index}: {full_info}")
        dimensions = row.get("dimension")
        if not isinstance(dimensions, list) or not dimensions:
            raise ValueError(f"Missing dimensions in VBench metadata row {index}: {full_info}")
        unknown = set(dimensions) - set(ALL_DIMENSIONS)
        if unknown:
            raise ValueError(f"Unknown dimensions in metadata row {index}: {sorted(unknown)}")
    return payload


def prompts_by_dimension(metadata: list[dict[str, Any]]) -> dict[str, list[str]]:
    result = {dimension: [] for dimension in ALL_DIMENSIONS}
    seen = {dimension: set() for dimension in ALL_DIMENSIONS}
    for row in metadata:
        prompt = str(row["prompt_en"])
        for dimension in row["dimension"]:
            if dimension in result and prompt not in seen[dimension]:
                result[dimension].append(prompt)
                seen[dimension].add(prompt)
    return result


def find_sample(directory: Path, prompt: str, sample_index: int) -> Path | None:
    for suffix in MEDIA_SUFFIXES:
        candidate = directory / f"{prompt}-{sample_index}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    return None


def stage_official_videos(
    videos: Path,
    filtered_temporal: Path | None,
    staging: Path,
    dimensions: list[str],
    prompt_map: dict[str, list[str]],
    assume_temporal_filtered: bool,
    allow_incomplete: bool,
    samples_per_prompt: int,
    subset_diagnostic: bool,
) -> dict[str, Any]:
    selected_prompts: list[str] = []
    prompt_dimensions: dict[str, set[str]] = {}
    for dimension in dimensions:
        for prompt in prompt_map[dimension]:
            if prompt not in prompt_dimensions:
                selected_prompts.append(prompt)
                prompt_dimensions[prompt] = set()
            prompt_dimensions[prompt].add(dimension)

    uses_flicker = "temporal_flickering" in dimensions
    if (
        uses_flicker
        and filtered_temporal is None
        and not assume_temporal_filtered
        and not subset_diagnostic
    ):
        raise ValueError(
            "Official temporal_flickering evaluation requires VBench static filtering. "
            "Pass --temporal-filtered-dir <filter_results/filtered_videos>, or explicitly "
            "use --assume-temporal-filtered for an already filtered official set."
        )

    required: list[tuple[str, int, Path]] = []
    missing: list[dict[str, Any]] = []
    # The official static filter may return fewer than five videos for a
    # temporal-flickering prompt when fewer than five generated candidates are
    # classified as static. Those absent filtered samples are expected filter
    # output, not missing generation inputs. Evaluating a moving fallback video
    # would invalidate the temporal-flickering metric.
    filtered_temporal_shortfall: list[dict[str, Any]] = []
    per_dimension_missing = {dimension: 0 for dimension in dimensions}
    selected_by_dimension = {dimension: 0 for dimension in dimensions}
    for prompt in selected_prompts:
        uses_filtered_source = (
            "temporal_flickering" in prompt_dimensions[prompt]
            and filtered_temporal is not None
        )
        source_dir = filtered_temporal if uses_filtered_source else videos
        for sample_index in range(samples_per_prompt):
            source = find_sample(source_dir, prompt, sample_index)
            if source is None:
                row = {
                    "prompt": prompt,
                    "sample_index": sample_index,
                    "searched_directory": str(source_dir),
                }
                if uses_filtered_source:
                    filtered_temporal_shortfall.append(row)
                else:
                    missing.append(row)
                    for dimension in prompt_dimensions[prompt]:
                        per_dimension_missing[dimension] += 1
            else:
                required.append((prompt, sample_index, source))
                for dimension in prompt_dimensions[prompt]:
                    selected_by_dimension[dimension] += 1

    if missing and not allow_incomplete:
        examples = "\n".join(
            f"  - {row['prompt']}-{row['sample_index']} in {row['searched_directory']}"
            for row in missing[:10]
        )
        raise FileNotFoundError(
            f"Official VBench input is incomplete: {len(missing)} required videos are missing.\n{examples}\n"
            "Use --allow-incomplete only for a non-official diagnostic run."
        )

    staging.mkdir(parents=True, exist_ok=True)
    expected_names = {f"{prompt}-{index}{source.suffix.lower()}" for prompt, index, source in required}
    for existing in staging.iterdir():
        if existing.name not in expected_names and existing.is_symlink():
            existing.unlink()
    for prompt, sample_index, source in required:
        destination = staging / f"{prompt}-{sample_index}{source.suffix.lower()}"
        if destination.is_symlink():
            if destination.resolve() == source:
                continue
            destination.unlink()
        elif destination.exists():
            raise FileExistsError(f"Refusing to replace non-symlink staging file: {destination}")
        destination.symlink_to(source)

    expected_by_dimension = {
        dimension: (
            selected_by_dimension[dimension]
            if dimension == "temporal_flickering" and filtered_temporal is not None
            else len(prompt_map[dimension]) * samples_per_prompt
        )
        for dimension in dimensions
    }
    complete_by_dimension = {
        dimension: per_dimension_missing[dimension] == 0 for dimension in dimensions
    }
    return {
        "selected_unique_prompts": len(selected_prompts),
        "staged_video_count": len(required),
        "missing_video_count": len(missing),
        "missing": missing,
        "filtered_temporal_shortfall_count": len(filtered_temporal_shortfall),
        "filtered_temporal_shortfall": filtered_temporal_shortfall,
        "expected_by_dimension": expected_by_dimension,
        "missing_by_dimension": per_dimension_missing,
        "complete_by_dimension": complete_by_dimension,
        "input_complete": not missing,
        "samples_per_prompt": samples_per_prompt,
        "temporal_filtering": {
            "required": uses_flicker,
            "filtered_directory": str(filtered_temporal) if filtered_temporal else None,
            "selected_video_count": selected_by_dimension.get("temporal_flickering", 0),
            "requested_video_count": (
                len(prompt_map["temporal_flickering"]) * samples_per_prompt
                if "temporal_flickering" in dimensions
                else 0
            ),
            "assumed_already_filtered": assume_temporal_filtered,
            "omitted_for_subset_diagnostic": (
                uses_flicker
                and filtered_temporal is None
                and not assume_temporal_filtered
                and subset_diagnostic
            ),
        },
    }


def prepare_torch_hub_cache(cache_dir: Path) -> None:
    torch_home = Path(os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache/torch"))).resolve()
    source = cache_dir / "dino_model/facebookresearch_dino_main"
    checkpoint = cache_dir / "dino_model/dino_vitbase16_pretrain.pth"
    target = torch_home / "hub/checkpoints" / checkpoint.name
    try:
        target_exists = target.exists()
    except OSError as exc:
        if getattr(exc, "errno", None) != 116 or not target.is_symlink():
            raise
        target.unlink()
        target_exists = False
    if checkpoint.is_file() and not target_exists:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        target.symlink_to(checkpoint.resolve())
    if not source.is_dir():
        print(f"[WARN] Local DINO repository is missing: {source}")


def choose_device(torch: Any, requested: str) -> Any:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    if requested == "cpu":
        print("[WARN] CPU VBench evaluation is extremely slow")
    # VBench's official launcher passes torch.device("cuda"). In particular,
    # the GRiT-backed semantic dimensions access ``device.type`` directly.
    return torch.device(requested)


def raw_result_path(raw_dir: Path, run_name: str, dimension: str) -> Path:
    return raw_dir / f"{run_name}_{dimension}_eval_results.json"


def read_dimension_result(path: Path, dimension: str) -> tuple[float, list[Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    value = payload.get(dimension)
    if not isinstance(value, list) or len(value) != 2 or not isinstance(value[1], list):
        raise ValueError(f"Malformed result for {dimension}: {path}")
    return float(value[0]), value[1]


def validate_result_row_count(
    dimension: str,
    actual_count: int,
    expected_count: int,
    result_path: Path,
    source: str,
) -> None:
    if actual_count == expected_count:
        return
    # VBench's official color implementation omits a video result when GRiT
    # cannot detect the requested object in any sampled frame. The aggregate
    # is intentionally computed over the detected videos, so a shorter result
    # list is valid even though every input video is present.
    if dimension == "color" and 0 < actual_count < expected_count:
        print(
            f"[WARN] {source} color result has {actual_count}/{expected_count} rows; "
            "VBench skipped videos where the target object was not detected: "
            f"{result_path}"
        )
        return
    raise ValueError(
        f"{source} {dimension} result has {actual_count} rows, expected "
        f"{expected_count}: {result_path}"
    )


def run_dimensions(
    staging: Path,
    raw_dir: Path,
    full_info: Path,
    run_name: str,
    dimensions: list[str],
    expected_by_dimension: dict[str, int],
    complete_by_dimension: dict[str, bool],
    device_name: str,
    imaging_mode: str,
    resume: bool,
    overwrite: bool,
    fail_fast: bool,
) -> dict[str, str]:
    try:
        import torch
        local_gpu_index = os.environ.get("VBENCH_LOCAL_GPU_INDEX")
        if local_gpu_index is not None:
            gpu = int(local_gpu_index)
            torch.cuda.set_device(gpu)
            print(f"[GPU] evaluation selected cuda:{gpu}", flush=True)
        import vbench
        from vbench import VBench
    except Exception as exc:
        raise RuntimeError(
            "VBench runtime is unavailable. See docs/evaluation.md for VBench installation. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    device = choose_device(torch, device_name)
    print(f"[INFO] python={sys.executable}")
    print(f"[INFO] vbench={Path(vbench.__file__).resolve()}")
    print(f"[INFO] device={device}")
    import_errors = {}
    for dimension in dimensions:
        try:
            importlib.import_module(f"vbench.{dimension}")
        except Exception as exc:
            import_errors[dimension] = f"{type(exc).__name__}: {exc}"
    if import_errors:
        details = "\n".join(f"  - {dimension}: {error}" for dimension, error in import_errors.items())
        raise RuntimeError(
            "VBench dimension dependencies are incomplete; refusing to begin a long evaluation.\n"
            f"{details}"
        )
    errors: dict[str, str] = {}
    raw_dir.mkdir(parents=True, exist_ok=True)

    for index, dimension in enumerate(dimensions, start=1):
        result_path = raw_result_path(raw_dir, run_name, dimension)
        expected_count = expected_by_dimension[dimension]
        if result_path.is_file() and resume and not overwrite:
            _, rows = read_dimension_result(result_path, dimension)
            if complete_by_dimension[dimension]:
                validate_result_row_count(
                    dimension, len(rows), expected_count, result_path, "Existing"
                )
            print(f"[RESUME {index}/{len(dimensions)}] {dimension}")
            continue
        if result_path.exists() and not overwrite:
            raise FileExistsError(f"Result exists; use --resume or --overwrite: {result_path}")
        if result_path.exists():
            result_path.unlink()

        print(f"[RUN {index}/{len(dimensions)}] {dimension}")
        # AMT checks the literal string cuda; GRiT expects torch.device.type.
        dimension_device = str(device) if dimension == "motion_smoothness" else device
        evaluator = VBench(dimension_device, str(full_info), str(raw_dir))
        try:
            evaluator.evaluate(
                videos_path=str(staging),
                name=f"{run_name}_{dimension}",
                prompt_list=[],
                dimension_list=[dimension],
                local=True,
                read_frame=False,
                mode="vbench_standard",
                imaging_quality_preprocessing_mode=imaging_mode,
            )
            _, rows = read_dimension_result(result_path, dimension)
            if complete_by_dimension[dimension]:
                validate_result_row_count(
                    dimension, len(rows), expected_count, result_path, "VBench"
                )
            print(f"[OK {index}/{len(dimensions)}] {dimension}: {result_path}")
        except Exception as exc:
            errors[dimension] = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] {dimension}: {errors[dimension]}", file=sys.stderr)
            traceback.print_exc()
            if fail_fast:
                raise
        finally:
            del evaluator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return errors


def calculate_official_aggregate(raw_scores: dict[str, float], reference: dict[str, Any] | None = REHYAT_REFERENCE) -> dict[str, Any] | None:
    if not all(dimension in raw_scores for dimension in ALL_DIMENSIONS):
        return None
    normalized = {}
    weighted = {}
    for dimension in ALL_DIMENSIONS:
        minimum, maximum = NORMALIZATION[dimension]
        normalized[dimension] = (raw_scores[dimension] - minimum) / (maximum - minimum)
        weighted[dimension] = normalized[dimension] * DIMENSION_WEIGHTS[dimension]
    quality = sum(weighted[d] for d in QUALITY_DIMENSIONS) / sum(
        DIMENSION_WEIGHTS[d] for d in QUALITY_DIMENSIONS
    )
    semantic = sum(weighted[d] for d in SEMANTIC_DIMENSIONS) / sum(
        DIMENSION_WEIGHTS[d] for d in SEMANTIC_DIMENSIONS
    )
    total = (4.0 * quality + semantic) / 5.0
    percentages = {
        "total": total * 100.0,
        "quality": quality * 100.0,
        "semantic": semantic * 100.0,
    }
    result = {**percentages, "normalized_dimensions": normalized}
    if reference is not None:
        result["delta_vs_rehyat_percentage_points"] = {
            key: percentages[key] - float(reference[key])
            for key in ("total", "quality", "semantic")
        }
    return result


def collect_scores(raw_dir: Path, run_name: str, dimensions: list[str]) -> tuple[dict[str, float], dict[str, str]]:
    scores: dict[str, float] = {}
    errors: dict[str, str] = {}
    for dimension in dimensions:
        path = raw_result_path(raw_dir, run_name, dimension)
        if not path.is_file():
            continue
        try:
            score, _ = read_dimension_result(path, dimension)
            scores[dimension] = score
        except Exception as exc:
            errors[dimension] = f"{type(exc).__name__}: {exc}"
    return scores, errors


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    protocol = summary["protocol"]
    title = (
        "# VBench official-prompt subset diagnostic"
        if protocol["subset_diagnostic"]
        else "# VBench standard evaluation"
    )
    lines = [title, ""]
    lines.extend(
        [
            f"- Subset diagnostic: `{protocol['subset_diagnostic']}`",
            f"- Input complete for this plan: `{protocol['input_complete']}`",
            f"- Official input complete: `{protocol['official_complete']}`",
            f"- Metadata records: `{protocol['metadata_record_count']}`",
            f"- Samples per prompt: `{protocol['samples_per_prompt']}`",
            f"- Staged videos: `{protocol['staged_video_count']}`",
            f"- Missing videos: `{protocol['missing_video_count']}`",
            "",
            "## Raw dimensions",
            "",
            "| Dimension | Raw score |",
            "|---|---:|",
        ]
    )
    for dimension in ALL_DIMENSIONS:
        score = summary["raw_scores"].get(dimension)
        lines.append(f"| {dimension} | {'-' if score is None else f'{score:.6f}'} |")
    aggregate = summary.get("official_aggregate")
    diagnostic_aggregate = summary.get("diagnostic_subset_aggregate")
    lines.extend(["", "## Aggregate", ""])
    reference = summary.get("rehyat_reference")
    if aggregate is not None and reference is None:
        lines.extend(["| Score | Current |", "|---|---:|"])
        for key in ("total", "quality", "semantic"):
            lines.append(f"| {key.title()} | {aggregate[key]:.2f} |")
    elif aggregate is not None:
        lines.extend(
            [
                "| Score | Current | ReHyAt | Delta (pp) |",
                "|---|---:|---:|---:|",
            ]
        )
        for key in ("total", "quality", "semantic"):
            lines.append(
                f"| {key.title()} | {aggregate[key]:.2f} | {reference[key]:.2f} | "
                f"{aggregate['delta_vs_rehyat_percentage_points'][key]:+.2f} |"
            )
    elif diagnostic_aggregate is not None:
        lines.extend(
            [
                "This is a high-variance subset proxy, not an official VBench score and not "
                "directly comparable with ReHyAt's full-suite score.",
                "",
                "| Proxy | Score |",
                "|---|---:|",
            ]
        )
        for key in ("total", "quality", "semantic"):
            lines.append(f"| {key.title()} | {diagnostic_aggregate[key]:.2f} |")
    else:
        lines.append("No aggregate was produced because the planned suite or dimensions are incomplete.")
    if summary["errors"]:
        lines.extend(["", "## Errors", ""])
        for dimension, error in summary["errors"].items():
            lines.append(f"- `{dimension}`: {error}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    reference = REHYAT_REFERENCE if args.reference == "rehyat" else None
    if args.samples_per_prompt < 1:
        raise ValueError("--samples-per-prompt must be at least 1")
    if not args.subset_diagnostic and args.samples_per_prompt != 5:
        raise ValueError("Official VBench standard requires --samples-per-prompt 5")
    videos = resolve(args.videos)
    output = resolve(args.output)
    # Multiple resume workers may converge on one dimension. Serialize staging
    # and result publication for this output; the next worker validates/resumes.
    output.mkdir(parents=True, exist_ok=True)
    output_lock = (output / ".evaluation.lock").open("a")
    print(f"[LOCK] Waiting for evaluation output: {output}", flush=True)
    fcntl.flock(output_lock.fileno(), fcntl.LOCK_EX)
    print(f"[LOCK] Acquired evaluation output: {output}", flush=True)
    cache_dir = resolve(args.cache_dir)
    filtered = resolve(args.temporal_filtered_dir) if args.temporal_filtered_dir else None
    if not videos.is_dir():
        raise FileNotFoundError(f"Video directory not found: {videos}")
    if filtered is not None and not filtered.is_dir():
        raise FileNotFoundError(f"Filtered temporal video directory not found: {filtered}")

    os.environ["VBENCH_CACHE_DIR"] = str(cache_dir)
    prepare_torch_hub_cache(cache_dir)
    if args.full_info is None:
        try:
            import vbench
        except Exception as exc:
            raise RuntimeError(
                "Cannot locate VBench metadata. See docs/evaluation.md for VBench installation. "
                f"Original error: {exc}"
            ) from exc
        full_info = Path(vbench.__file__).resolve().with_name("VBench_full_info.json")
    else:
        full_info = resolve(args.full_info)

    metadata = load_metadata(full_info, allow_subset=args.subset_diagnostic)
    metadata_is_full = len(metadata) == 946
    if args.subset_diagnostic and metadata_is_full:
        raise ValueError(
            "--subset-diagnostic requires a reduced official metadata file, not all 946 records"
        )
    prompt_map = prompts_by_dimension(metadata)
    absent_dimensions = [
        dimension for dimension in args.dimensions if not prompt_map[dimension]
    ]
    if absent_dimensions:
        raise ValueError(
            f"Selected metadata does not cover requested dimensions: {absent_dimensions}"
        )
    run_name = safe_name(args.name)
    staging = output / "staged_videos"
    raw_dir = output / "raw_results"
    protocol = stage_official_videos(
        videos=videos,
        filtered_temporal=filtered,
        staging=staging,
        dimensions=args.dimensions,
        prompt_map=prompt_map,
        assume_temporal_filtered=args.assume_temporal_filtered,
        allow_incomplete=args.allow_incomplete,
        samples_per_prompt=args.samples_per_prompt,
        subset_diagnostic=args.subset_diagnostic,
    )
    protocol["official_complete"] = (
        protocol["input_complete"]
        and metadata_is_full
        and args.samples_per_prompt == 5
        and set(args.dimensions) == set(ALL_DIMENSIONS)
        and not args.subset_diagnostic
    )
    protocol.update(
        {
            "mode": "vbench_standard",
            "full_info": str(full_info),
            "videos": str(videos),
            "dimensions": args.dimensions,
            "metadata_record_count": len(metadata),
            "metadata_is_full_official_suite": metadata_is_full,
            "subset_diagnostic": args.subset_diagnostic,
            "prompt_count_by_dimension": {
                dimension: len(prompt_map[dimension]) for dimension in args.dimensions
            },
            "prompts_by_dimension": (
                {dimension: prompt_map[dimension] for dimension in args.dimensions}
                if args.subset_diagnostic
                else None
            ),
        }
    )
    write_json(output / "protocol_manifest.json", protocol)
    print(
        f"[PREFLIGHT] prompts={protocol['selected_unique_prompts']} "
        f"staged={protocol['staged_video_count']} missing={protocol['missing_video_count']} "
        f"official_complete={protocol['official_complete']}"
    )
    if args.list_only:
        print(f"[DONE] protocol manifest: {output / 'protocol_manifest.json'}")
        return

    run_errors = run_dimensions(
        staging=staging,
        raw_dir=raw_dir,
        full_info=full_info,
        run_name=run_name,
        dimensions=args.dimensions,
        expected_by_dimension=protocol["expected_by_dimension"],
        complete_by_dimension=protocol["complete_by_dimension"],
        device_name=args.device,
        imaging_mode=args.imaging_quality_preprocessing_mode,
        resume=args.resume,
        overwrite=args.overwrite,
        fail_fast=args.fail_fast,
    )
    raw_scores, parse_errors = collect_scores(raw_dir, run_name, args.dimensions)
    errors = {**run_errors, **parse_errors}
    full_official_run = (
        protocol["official_complete"] and not errors
    )
    aggregate = calculate_official_aggregate(raw_scores, reference=reference) if full_official_run else None
    diagnostic_aggregate = None
    if (
        args.subset_diagnostic
        and protocol["input_complete"]
        and set(args.dimensions) == set(ALL_DIMENSIONS)
        and not errors
    ):
        diagnostic_aggregate = calculate_official_aggregate(raw_scores, reference=reference)
        if diagnostic_aggregate is not None:
            diagnostic_aggregate.pop("delta_vs_rehyat_percentage_points", None)
            diagnostic_aggregate["warning"] = (
                "High-variance official-prompt subset proxy; not an official VBench score "
                "and not directly comparable with ReHyAt's full-suite result."
            )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "raw_scores": raw_scores,
        "official_aggregate": aggregate,
        "diagnostic_subset_aggregate": diagnostic_aggregate,
        "rehyat_reference": reference,
        "errors": errors,
    }
    write_json(output / "summary.json", summary)
    write_markdown(output / "summary.md", summary)
    print(f"[DONE] summary: {output / 'summary.md'}")
    if aggregate is not None:
        print(
            f"[SCORE] Total={aggregate['total']:.2f} Quality={aggregate['quality']:.2f} "
            f"Semantic={aggregate['semantic']:.2f}; "
            + (f"delta_vs_ReHyAt={aggregate['delta_vs_rehyat_percentage_points']['total']:+.2f} pp"
               if reference is not None else "")
        )
    elif diagnostic_aggregate is not None:
        print(
            f"[DIAGNOSTIC PROXY] Total={diagnostic_aggregate['total']:.2f} "
            f"Quality={diagnostic_aggregate['quality']:.2f} "
            f"Semantic={diagnostic_aggregate['semantic']:.2f}; "
            "not directly comparable with ReHyAt"
        )
    else:
        print("[INFO] No official aggregate: complete all 16 dimensions with a complete official input set.")


if __name__ == "__main__":
    main()
