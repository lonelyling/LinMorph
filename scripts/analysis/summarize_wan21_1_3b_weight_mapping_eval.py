#!/usr/bin/env python3
"""Merge independently evaluated Wan2.1 mapping candidates into one report."""

from __future__ import annotations
import sys
from pathlib import Path

# Support running this entry point directly from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import argparse
import csv
import json
import os


METRICS = (
    "attention_mse",
    "attention_nmse",
    "attention_cosine",
    "block_mse",
    "block_nmse",
    "block_cosine",
)
PROTOCOL_KEYS = (
    "evaluated_indices",
    "timesteps",
    "target_blocks",
    "eval_seed",
    "height",
    "width",
    "num_frames",
)


def _atomic_write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def _candidate_metadata(label: str):
    if label == "manual_copy":
        return {
            "training_profile": "baseline",
            "family": "manual_copy",
            "stage": "baseline",
        }
    parts = label.split("_", 2)
    if len(parts) != 3:
        return {
            "training_profile": "unknown",
            "family": "unknown",
            "stage": "unknown",
        }
    training_profile, family, stage = parts
    return {
        "training_profile": training_profile,
        "family": family,
        "stage": stage,
    }


def _improvements(item, baseline):
    values = {}
    for prefix in ("attention", "block"):
        for metric in ("mse", "nmse"):
            key = f"{prefix}_{metric}_mean"
            base_value = float(baseline[key])
            values[f"{prefix}_{metric}_improvement_pct"] = (
                100.0 * (base_value - float(item[key])) / base_value
                if base_value
                else 0.0
            )
        cosine_key = f"{prefix}_cosine_mean"
        values[f"{prefix}_cosine_gain"] = float(item[cosine_key]) - float(
            baseline[cosine_key]
        )
    return values


def _load_summaries(result_root: Path):
    summaries = {}
    for path in sorted(result_root.glob("*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        label = summary["candidate"]
        if label in summaries:
            raise ValueError(f"Duplicate candidate label {label!r}: {path}")
        summaries[label] = {"path": str(path), "value": summary}
    if not summaries:
        raise FileNotFoundError(f"No */summary.json files found under {result_root}")
    if "manual_copy" not in summaries:
        raise FileNotFoundError(
            f"manual_copy/summary.json is required under {result_root}"
        )
    return summaries


def _validate_protocol(summaries):
    baseline = summaries["manual_copy"]["value"]["config"]
    expected = {key: baseline[key] for key in PROTOCOL_KEYS}
    mismatches = []
    for label, entry in summaries.items():
        actual = entry["value"]["config"]
        for key, expected_value in expected.items():
            if actual.get(key) != expected_value:
                mismatches.append(
                    {
                        "candidate": label,
                        "key": key,
                        "expected": expected_value,
                        "actual": actual.get(key),
                    }
                )
    if mismatches:
        raise ValueError(
            "Candidate protocols are not paired/identical:\n"
            + json.dumps(mismatches, ensure_ascii=False, indent=2)
        )
    return expected


def _build_comparison(summaries, protocol):
    manual = summaries["manual_copy"]["value"]
    overall = {}
    for label, entry in summaries.items():
        summary = entry["value"]
        overall[label] = {
            "candidate": label,
            **_candidate_metadata(label),
            "checkpoint_path": summary["checkpoint_path"],
            **summary["overall"],
            **_improvements(summary["overall"], manual["overall"]),
            "runtime_seconds": summary["runtime_seconds"],
            "peak_gpu_memory_gib": summary["peak_gpu_memory_gib"],
            "source_summary": entry["path"],
        }

    candidate_labels = [label for label in overall if label != "manual_copy"]
    ranking = sorted(
        candidate_labels,
        key=lambda label: overall[label]["block_nmse_mean"],
    )
    block_winners = {}
    for block_index in protocol["target_blocks"]:
        key = str(block_index)
        baseline = manual["by_block"][key]
        ranked = sorted(
            candidate_labels,
            key=lambda label: summaries[label]["value"]["by_block"][key][
                "block_nmse_mean"
            ],
        )
        winner = ranked[0]
        item = summaries[winner]["value"]["by_block"][key]
        block_winners[key] = {
            "candidate": winner,
            **item,
            **_improvements(item, baseline),
        }

    return {
        "protocol": protocol,
        "candidate_count": len(overall),
        "overall": overall,
        "ranking_by_block_nmse": ranking,
        "best_candidate": ranking[0] if ranking else None,
        "block_winners": block_winners,
    }


def _write_csv(path: Path, comparison) -> None:
    fieldnames = [
        "rank",
        "candidate",
        "training_profile",
        "family",
        "stage",
        "attention_nmse_mean",
        "attention_nmse_improvement_pct",
        "attention_cosine_mean",
        "attention_cosine_gain",
        "block_nmse_mean",
        "block_nmse_improvement_pct",
        "block_cosine_mean",
        "block_cosine_gain",
        "count",
        "runtime_seconds",
        "peak_gpu_memory_gib",
        "checkpoint_path",
    ]
    ranking = comparison["ranking_by_block_nmse"]
    order = ["manual_copy", *ranking]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for label in order:
            item = comparison["overall"][label]
            row = {name: item.get(name) for name in fieldnames}
            row["rank"] = "baseline" if label == "manual_copy" else ranking.index(label) + 1
            writer.writerow(row)
    os.replace(temporary, path)


def _write_report(path: Path, comparison) -> None:
    protocol = comparison["protocol"]
    overall = comparison["overall"]
    ranking = comparison["ranking_by_block_nmse"]
    manual = overall["manual_copy"]
    lines = [
        "# Wan2.1-1.3B exact-large-GDN mapping comparison",
        "",
        (
            f"Held-out shape: `{protocol['num_frames']}x{protocol['height']}x"
            f"{protocol['width']}`; dataset indices: `{protocol['evaluated_indices']}`; "
            f"timesteps: `{protocol['timesteps']}`."
        ),
        "",
        "Lower NMSE and higher cosine are better. Improvement is relative to the "
        "deterministic Manual Copy mapping evaluated on exactly the same cases.",
        "",
        "| rank | candidate | block NMSE | vs Manual Copy | block cosine | attention NMSE | vs Manual Copy |",
        "|---:|---|---:|---:|---:|---:|---:|",
        (
            f"| baseline | manual_copy | {manual['block_nmse_mean']:.8f} | "
            f"+0.00% | {manual['block_cosine_mean']:.8f} | "
            f"{manual['attention_nmse_mean']:.8f} | +0.00% |"
        ),
    ]
    for rank, label in enumerate(ranking, start=1):
        item = overall[label]
        lines.append(
            f"| {rank} | {label} | {item['block_nmse_mean']:.8f} | "
            f"{item['block_nmse_improvement_pct']:+.2f}% | "
            f"{item['block_cosine_mean']:.8f} | "
            f"{item['attention_nmse_mean']:.8f} | "
            f"{item['attention_nmse_improvement_pct']:+.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Best candidate per block",
            "",
            "| block | candidate | block NMSE | vs Manual Copy |",
            "|---:|---|---:|---:|",
        ]
    )
    for block_index, item in comparison["block_winners"].items():
        lines.append(
            f"| {block_index} | {item['candidate']} | "
            f"{item['block_nmse_mean']:.8f} | "
            f"{item['block_nmse_improvement_pct']:+.2f}% |"
        )

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    result_root = args.result_root.expanduser().resolve()
    summaries = _load_summaries(result_root)
    protocol = _validate_protocol(summaries)
    comparison = _build_comparison(summaries, protocol)
    _atomic_write_json(result_root / "comparison_summary.json", comparison)
    _write_csv(result_root / "comparison_summary.csv", comparison)
    _write_report(result_root / "comparison_report.md", comparison)
    print(
        f"Comparison complete: best={comparison['best_candidate']} "
        f"report={result_root / 'comparison_report.md'}"
    )


if __name__ == "__main__":
    main()
