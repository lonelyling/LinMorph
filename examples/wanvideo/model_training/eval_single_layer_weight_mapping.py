import csv
import json
import math
import os
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import accelerate
import torch
import torch.distributed as distributed
import torch.nn.functional as F
from accelerate import InitProcessGroupKwargs
from accelerate.utils import set_seed
from safetensors import safe_open

from diffsynth.core import load_state_dict
from diffsynth.core.data.uni_dataset import UnifiedDataset

from train_single_layer_weight_mapping import (
    WanSingleLayerWeightMappingModule,
    _env_bool,
    _manual_copy_matrix,
    _parse_bool,
    weight_mapping_parser,
)


METRIC_NAMES = (
    "attention_mse",
    "attention_nmse",
    "attention_cosine",
    "block_mse",
    "block_nmse",
    "block_cosine",
)


@dataclass
class Candidate:
    label: str
    kind: str
    group: str
    checkpoint_path: str | None
    checkpoint_step: int | None
    source_block_index: int | None
    mixing: torch.Tensor | None
    delta_norm: float | None
    mapping_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    materialized_parameters: dict[str, torch.Tensor] = field(default_factory=dict)
    mapped_parameters: dict[str, torch.Tensor] = field(default_factory=dict)
    source_block_indices: list[int] = field(default_factory=list)

    def metadata(self):
        return {
            "label": self.label,
            "kind": self.kind,
            "group": self.group,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_step": self.checkpoint_step,
            "source_block_index": self.source_block_index,
            "source_block_indices": self.source_block_indices,
            "delta_norm": self.delta_norm,
            "mixing": self.mixing.float().tolist() if self.mixing is not None else None,
            "mapping_tensor_names": sorted(self.mapping_tensors),
            "materialized_parameter_names": sorted(self.materialized_parameters),
        }


def _checkpoint_sort_key(path):
    match = re.fullmatch(r"model-step-(\d+)\.safetensors", path.name)
    if match:
        return (0, int(match.group(1)))
    if path.name == "model-final.safetensors":
        return (1, math.inf)
    return (2, path.name)


def _parse_checkpoint_group(spec):
    if "=" not in spec:
        raise ValueError(
            f"Invalid checkpoint group {spec!r}; expected LABEL=/absolute/or/relative/path"
        )
    label, directory = spec.split("=", 1)
    label = label.strip()
    directory = directory.strip()
    if not label or not directory:
        raise ValueError(f"Invalid checkpoint group {spec!r}")
    return label, Path(directory).expanduser().resolve()


def _parse_checkpoint_spec(spec):
    if "=" not in spec:
        raise ValueError(
            f"Invalid checkpoint {spec!r}; expected LABEL=/absolute/or/relative/file"
        )
    label, path = spec.split("=", 1)
    label = label.strip()
    path = Path(path.strip()).expanduser().resolve()
    if not label or not str(path):
        raise ValueError(f"Invalid checkpoint {spec!r}")
    return label, path


def _normalize_checkpoint_state(state_dict):
    prefixes = ("mapped_attention.", "module.mapped_attention.")
    normalized = {}
    for name, value in state_dict.items():
        normalized_name = name
        for prefix in prefixes:
            if normalized_name.startswith(prefix):
                normalized_name = normalized_name[len(prefix) :]
                break
        normalized[normalized_name] = value
    return normalized


def _load_candidate_state(path, target_block_index):
    """Read only one block from a large student checkpoint when possible."""
    student_prefixes = (
        f"blocks.{target_block_index}.self_attn.attn.",
        f"pipe.dit.blocks.{target_block_index}.self_attn.attn.",
        f"student_dit.blocks.{target_block_index}.self_attn.attn.",
    )
    try:
        with safe_open(str(path), framework="pt", device="cpu") as file:
            keys = list(file.keys())
            normalized_keys = {
                next(
                    (
                        name[len(prefix) :]
                        for prefix in ("mapped_attention.", "module.mapped_attention.")
                        if name.startswith(prefix)
                    ),
                    name,
                )
                for name in keys
            }
            if "mixing" not in normalized_keys:
                selected = [
                    name
                    for name in keys
                    if any(name.startswith(prefix) for prefix in student_prefixes)
                ]
                if selected:
                    return {name: file.get_tensor(name) for name in selected}
    except Exception as error:
        print(f"Selective safetensors read unavailable for {path}: {error}")
    return _normalize_checkpoint_state(
        load_state_dict(str(path), torch_dtype=None, device="cpu")
    )


def _checkpoint_candidate(
    label,
    group,
    path,
    target_block_index,
    manual,
    checkpoint_source_block_index=None,
):
    if not path.is_file():
        raise FileNotFoundError(f"Mapping checkpoint does not exist: {path}")
    state = _load_candidate_state(path, target_block_index)
    student_prefixes = (
        f"blocks.{target_block_index}.self_attn.attn.",
        f"pipe.dit.blocks.{target_block_index}.self_attn.attn.",
        f"student_dit.blocks.{target_block_index}.self_attn.attn.",
    )
    student_materialized_parameters = {}
    for name, value in state.items():
        for prefix in student_prefixes:
            if name.startswith(prefix):
                student_materialized_parameters[name[len(prefix) :]] = (
                    value.detach().cpu()
                )
                break
    if "mixing" not in state and student_materialized_parameters:
        match = re.fullmatch(r"model-step-(\d+)", path.stem)
        step = int(match.group(1)) if match else None
        return Candidate(
            label=label,
            kind="finetuned_gdn",
            group=group,
            checkpoint_path=str(path),
            checkpoint_step=step,
            source_block_index=target_block_index,
            mixing=None,
            delta_norm=None,
            materialized_parameters=student_materialized_parameters,
            source_block_indices=[target_block_index],
        )
    if "mixing" not in state:
        raise KeyError(
            f"Checkpoint has neither compact mapping tensors nor block "
            f"{target_block_index} GDN parameters: {path}"
        )
    checkpoint_version = state.get("mapping_format_version")
    if checkpoint_version is not None and int(checkpoint_version.item()) > 5:
        raise ValueError(
            f"Unsupported mapping checkpoint format "
            f"{int(checkpoint_version.item())}: {path}"
        )
    checkpoint_block = state.get("selected_block_index")
    checkpoint_blocks = state.get("selected_block_indices")
    checkpoint_block_indices = (
        [int(value) for value in checkpoint_blocks.reshape(-1).tolist()]
        if checkpoint_blocks is not None
        else (
            [int(checkpoint_block.item())]
            if checkpoint_block is not None
            else []
        )
    )
    checkpoint_block_index = (
        checkpoint_block_indices[0] if len(checkpoint_block_indices) == 1 else None
    )
    is_shared_mapping = len(checkpoint_block_indices) > 1
    expected_block_index = (
        target_block_index
        if checkpoint_source_block_index is None
        else checkpoint_source_block_index
    )
    if not checkpoint_block_indices and checkpoint_source_block_index is not None:
        raise ValueError(
            f"Transfer checkpoint has no selected block metadata: {path}"
        )
    if (
        not is_shared_mapping
        and checkpoint_block_index is not None
        and checkpoint_block_index != expected_block_index
    ):
        raise ValueError(
            f"Checkpoint {path} is for block {checkpoint_block_index}, "
            f"not expected source block {expected_block_index}"
        )
    mixing = state["mixing"].detach().float().cpu()
    if tuple(mixing.shape) != tuple(manual.shape):
        raise ValueError(f"Unexpected mixing shape in {path}: {tuple(mixing.shape)}")
    match = re.fullmatch(r"model-step-(\d+)", path.stem)
    step = int(match.group(1)) if match else None
    mapping_tensors = {
        name: value.detach().float().cpu()
        for name, value in state.items()
        if name == "mixing"
        or name.startswith("auxiliary_mixing.")
        or name.startswith("low_rank_u.")
        or name.startswith("low_rank_v.")
        or name.startswith("full_rank_residual.")
    }
    materialized_parameters = {
        name.removeprefix("gdn."): value.detach().cpu()
        for name, value in state.items()
        if name.startswith("gdn.")
    }
    if (
        materialized_parameters
        and (
            is_shared_mapping
            or (
                checkpoint_source_block_index is not None
                and checkpoint_source_block_index != target_block_index
            )
        )
    ):
        print(
            f"Ignoring {len(materialized_parameters)} layer-specific materialized "
            f"GDN tensors from {path}; evaluating its compact mapping coefficients "
            f"on target block {target_block_index}."
        )
        materialized_parameters = {}
    return Candidate(
        label=label,
        kind="mapped_gdn",
        group=group,
        checkpoint_path=str(path),
        checkpoint_step=step,
        source_block_index=checkpoint_block_index,
        mixing=mixing,
        delta_norm=float((mixing - manual).norm().item()),
        mapping_tensors=mapping_tensors,
        materialized_parameters=materialized_parameters,
        source_block_indices=checkpoint_block_indices,
    )


def _discover_candidates(
    group_specs,
    checkpoint_specs,
    block_index,
    include_softmax=False,
    include_random_gdn=False,
    checkpoint_source_block_index=None,
):
    manual = _manual_copy_matrix("manual_copy")
    candidates = []
    if include_softmax:
        candidates.append(
            Candidate(
                label="softmax_teacher",
                kind="softmax_reference",
                group="reference",
                checkpoint_path=None,
                checkpoint_step=None,
                source_block_index=block_index,
                mixing=None,
                delta_norm=None,
            )
        )
    if include_random_gdn:
        candidates.append(
            Candidate(
                label="random_gdn",
                kind="random_gdn",
                group="baseline",
                checkpoint_path=None,
                checkpoint_step=0,
                source_block_index=block_index,
                mixing=None,
                delta_norm=None,
            )
        )
    candidates.append(
        Candidate(
            label="manual_copy",
            kind="manual_copy_gdn",
            group="baseline",
            checkpoint_path=None,
            checkpoint_step=0,
            source_block_index=block_index,
            mixing=manual,
            delta_norm=0.0,
            mapping_tensors={"mixing": manual},
        )
    )
    seen_labels = {candidate.label for candidate in candidates}

    for spec in group_specs:
        group, directory = _parse_checkpoint_group(spec)
        if not directory.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {directory}")
        paths = sorted(
            directory.glob("model-step-*.safetensors"), key=_checkpoint_sort_key
        )
        final_path = directory / "model-final.safetensors"
        if final_path.is_file():
            paths.append(final_path)
        if not paths:
            raise FileNotFoundError(f"No mapping checkpoints found in {directory}")

        for path in paths:
            match = re.fullmatch(r"model-step-(\d+)", path.stem)
            step = int(match.group(1)) if match else None
            suffix = f"step-{step}" if step is not None else "final"
            label = f"{group}/{suffix}"
            if label in seen_labels:
                raise ValueError(f"Duplicate candidate label: {label}")
            seen_labels.add(label)
            candidates.append(
                _checkpoint_candidate(
                    label,
                    group,
                    path,
                    block_index,
                    manual,
                    checkpoint_source_block_index=checkpoint_source_block_index,
                )
            )

    for spec in checkpoint_specs:
        label, path = _parse_checkpoint_spec(spec)
        if label in seen_labels:
            raise ValueError(f"Duplicate candidate label: {label}")
        seen_labels.add(label)
        candidates.append(
            _checkpoint_candidate(
                label,
                "explicit",
                path,
                block_index,
                manual,
                checkpoint_source_block_index=checkpoint_source_block_index,
            )
        )
    return candidates


def _cache_materialized_weights(model, candidates):
    mapper = model.mapped_attention
    mapping_parameters = {
        name: parameter
        for name, parameter in mapper.named_parameters()
        if name == "mixing"
        or name.startswith("auxiliary_mixing.")
        or name.startswith("low_rank_u.")
        or name.startswith("low_rank_v.")
        or name.startswith("full_rank_residual.")
    }
    original_mapping = {
        name: parameter.detach().clone()
        for name, parameter in mapping_parameters.items()
    }
    with torch.no_grad():
        for candidate in candidates:
            if candidate.kind == "softmax_reference":
                continue
            if candidate.kind == "random_gdn":
                if not mapper.random_initial_parameters:
                    raise RuntimeError(
                        "Random GDN candidate requested without capturing native initialization"
                    )
                gdn_parameters = dict(mapper.gdn.named_parameters())
                candidate.mapped_parameters = {
                    name: value.to(
                        device=gdn_parameters[name].device,
                        dtype=gdn_parameters[name].dtype,
                    )
                    for name, value in mapper.random_initial_parameters.items()
                }
                continue
            if candidate.materialized_parameters:
                gdn_parameters = dict(mapper.gdn.named_parameters())
                expected_names = set(gdn_parameters)
                provided_names = set(candidate.materialized_parameters)
                if provided_names != expected_names:
                    missing = sorted(expected_names - provided_names)
                    unexpected = sorted(provided_names - expected_names)
                    raise ValueError(
                        f"Candidate {candidate.label} has incomplete materialized GDN weights; "
                        f"missing={missing}, unexpected={unexpected}"
                    )
                candidate.mapped_parameters = {}
                for name, parameter in gdn_parameters.items():
                    value = candidate.materialized_parameters[name]
                    if tuple(value.shape) != tuple(parameter.shape):
                        raise ValueError(
                            f"Candidate {candidate.label} parameter {name} has shape "
                            f"{tuple(value.shape)}, expected {tuple(parameter.shape)}"
                        )
                    candidate.mapped_parameters[name] = value.to(
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                continue
            if (
                not mapper.map_auxiliary_parameters
                and any(
                    name.startswith("auxiliary_mixing.")
                    for name in candidate.mapping_tensors
                )
            ):
                raise ValueError(
                    f"Candidate {candidate.label} contains full-parameter mixing; "
                    "set --map_auxiliary_parameters true to evaluate it"
                )
            for name, parameter in mapping_parameters.items():
                parameter.copy_(original_mapping[name])
            for name, value in candidate.mapping_tensors.items():
                if name not in mapping_parameters:
                    raise ValueError(
                        f"Candidate {candidate.label} contains unsupported mapping tensor {name}"
                    )
                mapping_parameters[name].copy_(
                    value.to(
                        device=mapping_parameters[name].device,
                        dtype=mapping_parameters[name].dtype,
                    )
                )
            candidate.mapped_parameters = {
                name: value.detach()
                for name, value in mapper.materialized_parameters().items()
            }
        for name, parameter in mapping_parameters.items():
            parameter.copy_(original_mapping[name])


def _feature_metrics(prediction, target):
    prediction = prediction.detach().float().reshape(prediction.shape[0], -1)
    target = target.detach().float().reshape(target.shape[0], -1)
    error = prediction - target
    mse = error.square().mean(dim=1)
    nmse = error.square().sum(dim=1) / target.square().sum(dim=1).clamp_min(1e-12)
    cosine = F.cosine_similarity(prediction, target, dim=1, eps=1e-12)
    return {
        "mse": float(mse.mean().item()),
        "nmse": float(nmse.mean().item()),
        "cosine": float(cosine.mean().item()),
    }


def _case_seed(eval_seed, dataset_index, timestep):
    modulus = (1 << 63) - 1
    return int((eval_seed + dataset_index * 1_000_003 + int(timestep) * 9_176) % modulus)


def _load_exact_dataset_item(dataset, dataset_index, retries):
    if getattr(dataset, "is_webdata", False) or getattr(dataset, "load_from_cache", False):
        raise ValueError("V1 exact-index evaluation requires a regular indexed dataset")
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            if hasattr(dataset, "_getitem_normal"):
                return dataset._getitem_normal(dataset_index)
            return dataset[dataset_index]
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(5.0, float(attempt)))
    raise RuntimeError(
        f"Exact dataset index {dataset_index} failed after {retries} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    )


def _prepare_video_inputs(model, data, seed):
    set_seed(seed, device_specific=False)
    inputs = model.get_pipeline_inputs(data)
    inputs = model.transfer_data_to_device(
        inputs,
        model.pipe.device,
        model.pipe.torch_dtype,
    )
    with torch.no_grad():
        for unit in model.pipe.units:
            inputs = model.pipe.unit_runner(unit, model.pipe, *inputs)
    inputs_shared, inputs_posi, _ = inputs
    inputs_shared["memory_tokens"] = None
    return inputs_shared, inputs_posi


def _build_noisy_model_kwargs(model, inputs_shared, inputs_posi, timestep, seed):
    clean_latents = inputs_shared.get("input_latents", inputs_shared["latents"])
    noise = model.pipe.generate_noise(
        clean_latents.shape,
        seed=seed,
        rand_device="cpu",
        device=clean_latents.device,
        torch_dtype=clean_latents.dtype,
    )
    timesteps = torch.full(
        (clean_latents.shape[0],),
        float(timestep),
        device=clean_latents.device,
        dtype=torch.float32,
    )
    noisy_latents = model.pipe.scheduler.add_noise(
        clean_latents,
        noise,
        timestep=timesteps,
    ).to(clean_latents.dtype)

    if (
        inputs_shared.get("fuse_vae_embedding_in_latents", False)
        and inputs_shared.get("first_frame_latents") is not None
    ):
        noisy_latents = noisy_latents.clone()
        noisy_latents[:, :, 0:1] = inputs_shared["first_frame_latents"]

    return {
        "latents": noisy_latents,
        "timestep": timesteps,
        "context": inputs_posi.get("context"),
        "y": inputs_shared.get("y"),
        "clip_feature": inputs_shared.get("clip_feature"),
        "fuse_vae_embedding_in_latents": inputs_shared.get(
            "fuse_vae_embedding_in_latents", False
        ),
    }


def _evaluate_case(model, candidates, model_kwargs, dataset_index, timestep, seed):
    teacher_features = model._capture_teacher_block_features(model_kwargs)
    # Multilayer mapping returns captures keyed by block index. Keep accepting
    # the original single-layer feature dictionary for older module versions.
    teacher = teacher_features.get(model.block_index, teacher_features)
    records = []
    with torch.no_grad():
        for candidate in candidates:
            if candidate.kind == "softmax_reference":
                mapped_attention = teacher["attention_output"]
                mapped_block = teacher["block_output"]
            else:
                mapped_attention, mapped_block = model._forward_mapped_block(
                    teacher["block_inputs"],
                    mapped_parameters=candidate.mapped_parameters,
                )
            attention = _feature_metrics(
                mapped_attention,
                teacher["attention_output"],
            )
            block = _feature_metrics(mapped_block, teacher["block_output"])
            records.append(
                {
                    "candidate": candidate.label,
                    "kind": candidate.kind,
                    "group": candidate.group,
                    "checkpoint_path": candidate.checkpoint_path,
                    "checkpoint_step": candidate.checkpoint_step,
                    "source_block_index": candidate.source_block_index,
                    "delta_norm": candidate.delta_norm,
                    "dataset_index": int(dataset_index),
                    "timestep": int(timestep),
                    "noise_seed": int(seed),
                    "attention_mse": attention["mse"],
                    "attention_nmse": attention["nmse"],
                    "attention_cosine": attention["cosine"],
                    "block_mse": block["mse"],
                    "block_nmse": block["nmse"],
                    "block_cosine": block["cosine"],
                }
            )
            model.mapped_attention.last_output = None
            del mapped_attention, mapped_block
    return records


def _atomic_write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def _atomic_write_jsonl(path, records):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _metric_summary(records):
    summary = {"count": len(records)}
    for metric in METRIC_NAMES:
        values = [float(record[metric]) for record in records]
        summary[f"{metric}_mean"] = statistics.fmean(values)
        summary[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return summary


def _group_by(records, key):
    grouped = {}
    for record in records:
        grouped.setdefault(record[key], []).append(record)
    return grouped


def _validate_record_coverage(records, candidates, config):
    expected_labels = {candidate.label for candidate in candidates}
    expected_timesteps = set(config["timesteps"])
    expected_case_count = config["eval_num_videos"] * len(expected_timesteps)
    expected_record_count = expected_case_count * len(expected_labels)

    if config["evaluated_video_count"] != config["eval_num_videos"]:
        raise RuntimeError(
            f"Expected {config['eval_num_videos']} evaluated videos, got "
            f"{config['evaluated_video_count']}"
        )
    if config["evaluated_case_count"] != expected_case_count:
        raise RuntimeError(
            f"Expected {expected_case_count} validation cases, got "
            f"{config['evaluated_case_count']}"
        )
    if len(records) != expected_record_count:
        raise RuntimeError(
            f"Expected {expected_record_count} candidate records, got {len(records)}"
        )

    by_case = {}
    candidate_counts = {label: 0 for label in expected_labels}
    for record in records:
        label = record["candidate"]
        if label not in expected_labels:
            raise RuntimeError(f"Unexpected candidate in validation records: {label}")
        if record["timestep"] not in expected_timesteps:
            raise RuntimeError(
                f"Unexpected timestep in validation records: {record['timestep']}"
            )
        for metric in METRIC_NAMES:
            if not math.isfinite(float(record[metric])):
                raise RuntimeError(
                    f"Non-finite {metric} for candidate={label}, "
                    f"index={record['dataset_index']}, timestep={record['timestep']}"
                )
        candidate_counts[label] += 1
        case_key = (record["dataset_index"], record["timestep"])
        by_case.setdefault(case_key, []).append(record)

    for label, count in candidate_counts.items():
        if count != expected_case_count:
            raise RuntimeError(
                f"Candidate {label} has {count} records, expected {expected_case_count}"
            )
    for case_key, case_records in by_case.items():
        case_labels = [record["candidate"] for record in case_records]
        case_seeds = {record["noise_seed"] for record in case_records}
        if len(case_labels) != len(expected_labels) or set(case_labels) != expected_labels:
            raise RuntimeError(f"Incomplete or duplicate candidate coverage for case {case_key}")
        if len(case_seeds) != 1:
            raise RuntimeError(f"Candidates use different noise seeds for case {case_key}")


def _summarize(records, candidates, config, failures):
    _validate_record_coverage(records, candidates, config)
    by_candidate = _group_by(records, "candidate")
    expected_labels = [candidate.label for candidate in candidates]
    missing = [label for label in expected_labels if label not in by_candidate]
    if missing:
        raise RuntimeError(f"No validation records for candidates: {missing}")

    overall = {}
    by_timestep = {}
    for candidate in candidates:
        label = candidate.label
        overall[label] = {
            **candidate.metadata(),
            **_metric_summary(by_candidate[label]),
        }
        timestep_groups = _group_by(by_candidate[label], "timestep")
        by_timestep[label] = {
            str(timestep): _metric_summary(timestep_records)
            for timestep, timestep_records in sorted(timestep_groups.items())
        }

    manual = overall["manual_copy"]
    for label, item in overall.items():
        for prefix in ("attention", "block"):
            for metric in ("mse", "nmse"):
                key = f"{prefix}_{metric}_mean"
                baseline = manual[key]
                item[f"{prefix}_{metric}_improvement_pct"] = (
                    100.0 * (baseline - item[key]) / baseline if baseline else 0.0
                )
            cosine_key = f"{prefix}_cosine_mean"
            item[f"{prefix}_cosine_gain"] = item[cosine_key] - manual[cosine_key]

    kinds = {candidate.label: candidate.kind for candidate in candidates}
    gdn_ranking = sorted(
        (label for label in overall if kinds[label] != "softmax_reference"),
        key=lambda label: overall[label]["block_nmse_mean"],
    )
    softmax_label = next(
        (label for label, kind in kinds.items() if kind == "softmax_reference"),
        None,
    )
    if softmax_label is not None:
        softmax = overall[softmax_label]
        for label, item in overall.items():
            item["attention_nmse_gap_to_softmax"] = (
                item["attention_nmse_mean"] - softmax["attention_nmse_mean"]
            )
            item["block_nmse_gap_to_softmax"] = (
                item["block_nmse_mean"] - softmax["block_nmse_mean"]
            )
            item["beats_softmax_on_teacher_distance"] = bool(
                item["attention_nmse_mean"]
                < softmax["attention_nmse_mean"] - 1e-12
                or item["block_nmse_mean"] < softmax["block_nmse_mean"] - 1e-12
            )
    best_gdn = gdn_ranking[0] if gdn_ranking else None
    return {
        "config": config,
        "candidates": [candidate.metadata() for candidate in candidates],
        "overall": overall,
        "by_timestep": by_timestep,
        "ranking_by_block_nmse": gdn_ranking,
        "gdn_ranking_by_block_nmse": gdn_ranking,
        "best_checkpoint": best_gdn,
        "best_gdn_candidate": best_gdn,
        "softmax_reference": softmax_label,
        "teacher_distance_note": (
            "Softmax is compared with its own captured features, so its MSE/NMSE is "
            "zero by definition. GDN candidates can only match, not beat, this reference "
            "on teacher-distance metrics."
        ),
        "failures": failures,
    }


def _write_summary_csv(path, summary):
    fieldnames = [
        "scope",
        "timestep",
        "candidate",
        "kind",
        "group",
        "checkpoint_step",
        "source_block_index",
        "delta_norm",
        "count",
        *[f"{metric}_{suffix}" for metric in METRIC_NAMES for suffix in ("mean", "std")],
        "attention_mse_improvement_pct",
        "attention_nmse_improvement_pct",
        "attention_cosine_gain",
        "block_mse_improvement_pct",
        "block_nmse_improvement_pct",
        "block_cosine_gain",
    ]
    rows = []
    for label, item in summary["overall"].items():
        rows.append(
            {
                "scope": "overall",
                "timestep": "all",
                "candidate": label,
                **{name: item.get(name) for name in fieldnames if name in item},
            }
        )
        manual_by_timestep = summary["by_timestep"]["manual_copy"]
        for timestep, timestep_item in summary["by_timestep"][label].items():
            row = {
                "scope": "timestep",
                "timestep": timestep,
                "candidate": label,
                "kind": item["kind"],
                "group": item["group"],
                "checkpoint_step": item["checkpoint_step"],
                "source_block_index": item["source_block_index"],
                "delta_norm": item["delta_norm"],
                **timestep_item,
            }
            manual = manual_by_timestep[timestep]
            for prefix in ("attention", "block"):
                for metric in ("mse", "nmse"):
                    key = f"{prefix}_{metric}_mean"
                    baseline = manual[key]
                    row[f"{prefix}_{metric}_improvement_pct"] = (
                        100.0 * (baseline - row[key]) / baseline if baseline else 0.0
                    )
                cosine_key = f"{prefix}_cosine_mean"
                row[f"{prefix}_cosine_gain"] = row[cosine_key] - manual[cosine_key]
            rows.append(row)

    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_markdown_report(path, summary):
    config = summary["config"]
    lines = [
        "# GDN Initialization Comparison",
        "",
        f"- Target block: `{config['block_index']}`",
        f"- Mapping checkpoint source block: `{config['mapping_source_block_index']}`",
        f"- Cross-layer transfer: `{config['is_cross_layer_transfer']}`",
        f"- Requested videos: `{config['eval_num_videos']}`",
        f"- Evaluated videos: `{config['evaluated_video_count']}`",
        f"- Timesteps: `{config['timesteps']}`",
        f"- Cases per candidate: `{config['evaluated_case_count']}`",
        f"- Best GDN candidate by block NMSE: `{summary['best_gdn_candidate']}`",
        f"- Softmax reference: `{summary['softmax_reference']}`",
        "- Metric meaning: smaller MSE/NMSE and larger cosine are closer to Softmax.",
        "- Important: Softmax is its own target, so its MSE/NMSE is 0 by definition; "
        "this evaluation ranks GDN initializations by closeness, not downstream quality.",
        "- Random GDN uses the seeded native FLA initialization for all 13 parameters; "
        "Manual Copy uses the stable handcrafted bases, while full-mix checkpoints may "
        "also override the auxiliary GDN parameters.",
        "",
        "## Overall",
        "",
        "| Rank | Candidate | Count | Block NMSE | vs Manual | Block Cosine | Attention NMSE | vs Manual | Delta norm |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ranking = []
    if summary["softmax_reference"] is not None:
        ranking.append(summary["softmax_reference"])
    ranking.extend(summary["gdn_ranking_by_block_nmse"])
    for rank, label in enumerate(ranking):
        item = summary["overall"][label]
        display_rank = "ref" if label == summary["softmax_reference"] else str(
            summary["gdn_ranking_by_block_nmse"].index(label) + 1
        )
        delta_norm = (
            f"{item['delta_norm']:.4f}" if item["delta_norm"] is not None else "n/a"
        )
        lines.append(
            f"| {display_rank} | `{label}` | {item['count']} | "
            f"{item['block_nmse_mean']:.6f} | {item['block_nmse_improvement_pct']:+.2f}% | "
            f"{item['block_cosine_mean']:.6f} | {item['attention_nmse_mean']:.6f} | "
            f"{item['attention_nmse_improvement_pct']:+.2f}% | {delta_norm} |"
        )

    best = summary["best_checkpoint"]
    if best is not None:
        lines.extend(
            [
                "",
                "## Best GDN Candidate by Timestep",
                "",
                "| Timestep | Manual block NMSE | Best block NMSE | Improvement | Manual attention NMSE | Best attention NMSE | Improvement |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for timestep, best_item in summary["by_timestep"][best].items():
            manual_item = summary["by_timestep"]["manual_copy"][timestep]
            block_gain = 100.0 * (
                manual_item["block_nmse_mean"] - best_item["block_nmse_mean"]
            ) / manual_item["block_nmse_mean"]
            attention_gain = 100.0 * (
                manual_item["attention_nmse_mean"] - best_item["attention_nmse_mean"]
            ) / manual_item["attention_nmse_mean"]
            lines.append(
                f"| {timestep} | {manual_item['block_nmse_mean']:.6f} | "
                f"{best_item['block_nmse_mean']:.6f} | {block_gain:+.2f}% | "
                f"{manual_item['attention_nmse_mean']:.6f} | "
                f"{best_item['attention_nmse_mean']:.6f} | {attention_gain:+.2f}% |"
            )

    if summary["failures"]:
        lines.extend(
            [
                "",
                "## Dataset Load Failures",
                "",
                f"`{len(summary['failures'])}` exact indices failed and were replaced by later deterministic stride indices.",
            ]
        )

    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validation_parser():
    parser = weight_mapping_parser()
    parser.description = "Fixed held-out validation for single-layer weight mapping"
    parser.add_argument(
        "--eval_start_index",
        type=int,
        default=int(os.environ.get("MAPPING_EVAL_START_INDEX", 8192)),
    )
    parser.add_argument(
        "--eval_num_videos",
        type=int,
        default=int(os.environ.get("MAPPING_EVAL_NUM_VIDEOS", 64)),
    )
    parser.add_argument(
        "--eval_timesteps",
        type=str,
        default=os.environ.get("MAPPING_EVAL_TIMESTEPS", "100,300,500,700,900"),
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=int(os.environ.get("MAPPING_EVAL_SEED", 20260715)),
    )
    parser.add_argument(
        "--eval_exact_retries",
        type=int,
        default=int(os.environ.get("MAPPING_EVAL_EXACT_RETRIES", 3)),
    )
    parser.add_argument(
        "--eval_scan_multiplier",
        type=int,
        default=int(os.environ.get("MAPPING_EVAL_SCAN_MULTIPLIER", 20)),
    )
    parser.add_argument(
        "--eval_log_interval",
        type=int,
        default=int(os.environ.get("MAPPING_EVAL_LOG_INTERVAL", 1)),
    )
    parser.add_argument(
        "--eval_mapping_source_block_index",
        type=int,
        default=int(os.environ.get("MAPPING_EVAL_SOURCE_BLOCK_INDEX", -1)),
        help=(
            "Layer that produced the mapping checkpoint. Set this to a different "
            "block than --block_index for zero-shot cross-layer transfer."
        ),
    )
    parser.add_argument(
        "--checkpoint_group",
        action="append",
        default=None,
        help="Repeat LABEL=DIR to evaluate every model-step/model-final checkpoint in DIR.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Repeat LABEL=FILE to evaluate only explicitly selected checkpoints.",
    )
    parser.add_argument(
        "--eval_include_softmax",
        type=_parse_bool,
        default=_env_bool("MAPPING_EVAL_INCLUDE_SOFTMAX", False),
    )
    parser.add_argument(
        "--eval_include_random_gdn",
        type=_parse_bool,
        default=_env_bool("MAPPING_EVAL_INCLUDE_RANDOM_GDN", False),
    )
    parser.add_argument(
        "--eval_destroy_process_group",
        type=_parse_bool,
        default=_env_bool("MAPPING_EVAL_DESTROY_PROCESS_GROUP", True),
    )
    return parser


def _parse_timesteps(value):
    timesteps = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not timesteps:
        raise ValueError("At least one evaluation timestep is required")
    if any(timestep < 0 or timestep > 1000 for timestep in timesteps):
        raise ValueError(f"Evaluation timesteps must be in [0, 1000], got {timesteps}")
    if len(set(timesteps)) != len(timesteps):
        raise ValueError(f"Duplicate evaluation timesteps: {timesteps}")
    return timesteps


def _checkpoint_groups_from_args(args):
    if args.checkpoint_group:
        return args.checkpoint_group
    env_value = os.environ.get("MAPPING_EVAL_CHECKPOINT_GROUPS", "")
    groups = [item.strip() for item in env_value.split(";") if item.strip()]
    return groups


def _checkpoint_specs_from_args(args):
    if args.checkpoint:
        return args.checkpoint
    env_value = os.environ.get("MAPPING_EVAL_CHECKPOINTS", "")
    return [item.strip() for item in env_value.split(";") if item.strip()]


def _patch_distributed_timeout(timeout):
    original_init_process_group = distributed.init_process_group
    original_new_group = distributed.new_group

    def init_process_group(*args, **kwargs):
        kwargs.setdefault("timeout", timeout)
        print(f"[NCCL patch] init_process_group timeout={kwargs['timeout']}", flush=True)
        return original_init_process_group(*args, **kwargs)

    def new_group(*args, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return original_new_group(*args, **kwargs)

    distributed.init_process_group = init_process_group
    distributed.new_group = new_group


def main():
    os.environ.setdefault("DIFFSYNTH_T2V_TRAINING", "1")
    timeout = timedelta(
        seconds=int(os.environ.get("DIFFSYNTH_NCCL_TIMEOUT_SEC", "7200"))
    )
    _patch_distributed_timeout(timeout)
    args = validation_parser().parse_args()
    if args.eval_start_index < 0:
        raise ValueError("--eval_start_index must be non-negative")
    if args.eval_num_videos <= 0:
        raise ValueError("--eval_num_videos must be positive")
    if args.eval_exact_retries <= 0 or args.eval_scan_multiplier <= 0:
        raise ValueError("Evaluation retry/scan settings must be positive")
    if args.eval_mapping_source_block_index < -1:
        raise ValueError("--eval_mapping_source_block_index must be -1 or non-negative")

    timesteps = _parse_timesteps(args.eval_timesteps)
    group_specs = _checkpoint_groups_from_args(args)
    checkpoint_specs = _checkpoint_specs_from_args(args)
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=1,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timeout)],
    )
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    set_seed(args.eval_seed, device_specific=False)

    qworld_config = os.environ.get("DIFFSYNTH_QWORLD_COMMON_CONFIG", "").strip()
    if qworld_config:
        from diffsynth.core.data.qworld_common_dataset import (
            build_qworld_common_dataset_from_env,
        )

        dataset = build_qworld_common_dataset_from_env(
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            repeat=1,
            seed=args.eval_seed,
        )
    else:
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=args.dataset_repeat,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.AOSS_video_operator(
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=4,
                time_division_remainder=1,
            ),
        )
    if args.eval_start_index >= len(dataset):
        raise ValueError(
            f"eval_start_index={args.eval_start_index} is outside dataset length {len(dataset)}"
        )

    model = WanSingleLayerWeightMappingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        device=accelerator.device,
        task=args.task,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        student_init_ckpt=None,
        block_index=args.block_index,
        gdn_block_indices=args.gdn_block_indices,
        middle_layer_start=args.middle_layer_start,
        middle_layer_end=args.middle_layer_end,
        mapping_seed=args.mapping_seed,
        mixing_init="manual_copy",
        low_rank=args.low_rank,
        full_rank_residual=args.full_rank_residual,
        low_rank_init_std=args.low_rank_init_std,
        gdn_expand_v=args.gdn_expand_v,
        gdn_head_dim=args.gdn_head_dim,
        shape_adapter=args.mapping_shape_adapter,
        gdn_use_short_conv=args.gdn_use_short_conv,
        identity_short_conv=args.identity_short_conv,
        map_auxiliary_parameters=args.map_auxiliary_parameters,
        train_base_mapping=True,
        capture_random_initialization=args.eval_include_random_gdn,
        attention_loss_weight=1.0,
        block_loss_weight=1.0,
        mapping_regularization=0.0,
        normalized_mse=False,
        mapping_log_interval=args.mapping_log_interval,
    )
    model.eval()
    model.requires_grad_(False)
    checkpoint_source_block_index = (
        None
        if args.eval_mapping_source_block_index < 0
        else args.eval_mapping_source_block_index
    )
    if (
        checkpoint_source_block_index is not None
        and checkpoint_source_block_index >= len(model.teacher_dit.blocks)
    ):
        raise ValueError(
            f"eval_mapping_source_block_index={checkpoint_source_block_index} is outside "
            f"[0, {len(model.teacher_dit.blocks) - 1}]"
        )
    candidates = _discover_candidates(
        group_specs,
        checkpoint_specs,
        model.block_index,
        include_softmax=args.eval_include_softmax,
        include_random_gdn=args.eval_include_random_gdn,
        checkpoint_source_block_index=checkpoint_source_block_index,
    )
    _cache_materialized_weights(model, candidates)
    accelerator.print(
        f"Validation candidates ({len(candidates)}): "
        f"{[candidate.label for candidate in candidates]}"
    )
    accelerator.print(
        f"V1 fixed validation: start={args.eval_start_index} videos={args.eval_num_videos} "
        f"timesteps={timesteps} seed={args.eval_seed} world_size={world_size} "
        f"mapping_source_block={checkpoint_source_block_index if checkpoint_source_block_index is not None else model.block_index} "
        f"target_block={model.block_index}"
    )

    target_count = len(range(rank, args.eval_num_videos, world_size))
    scan_limit = target_count * args.eval_scan_multiplier
    dataset_index = args.eval_start_index + rank
    successful_videos = 0
    scanned = 0
    local_records = []
    local_failures = []

    while successful_videos < target_count and scanned < scan_limit:
        if dataset_index >= len(dataset):
            break
        scanned += 1
        try:
            data = _load_exact_dataset_item(
                dataset,
                dataset_index,
                args.eval_exact_retries,
            )
        except Exception as error:
            local_failures.append(
                {
                    "rank": rank,
                    "dataset_index": dataset_index,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            print(
                f"[rank {rank}] skipping exact index {dataset_index}: {error}",
                flush=True,
            )
            dataset_index += world_size
            continue

        inputs_shared, inputs_posi = _prepare_video_inputs(
            model,
            data,
            args.eval_seed + dataset_index,
        )
        del data
        for timestep in timesteps:
            seed = _case_seed(args.eval_seed, dataset_index, timestep)
            model_kwargs = _build_noisy_model_kwargs(
                model,
                inputs_shared,
                inputs_posi,
                timestep,
                seed,
            )
            local_records.extend(
                _evaluate_case(
                    model,
                    candidates,
                    model_kwargs,
                    dataset_index,
                    timestep,
                    seed,
                )
            )
            del model_kwargs

        successful_videos += 1
        if (
            successful_videos % max(1, args.eval_log_interval) == 0
            or successful_videos == target_count
        ):
            print(
                f"[rank {rank}] evaluated videos {successful_videos}/{target_count} "
                f"last_index={dataset_index} records={len(local_records)}",
                flush=True,
            )
        dataset_index += world_size

    if successful_videos != target_count:
        raise RuntimeError(
            f"Rank {rank} evaluated only {successful_videos}/{target_count} videos after "
            f"scanning {scanned} exact indices"
        )

    _atomic_write_jsonl(output_path / f"v1_records.rank{rank}.jsonl", local_records)
    _atomic_write_json(output_path / f"v1_failures.rank{rank}.json", local_failures)
    if world_size > 1:
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        all_records = []
        all_failures = []
        for process_rank in range(world_size):
            record_path = output_path / f"v1_records.rank{process_rank}.jsonl"
            with record_path.open("r", encoding="utf-8") as file:
                all_records.extend(json.loads(line) for line in file if line.strip())
            failure_path = output_path / f"v1_failures.rank{process_rank}.json"
            all_failures.extend(json.loads(failure_path.read_text(encoding="utf-8")))

        evaluated_indices = sorted({record["dataset_index"] for record in all_records})
        evaluated_cases = {
            (record["dataset_index"], record["timestep"]) for record in all_records
        }
        config = {
            "block_index": model.block_index,
            "mapping_source_block_index": (
                model.block_index
                if checkpoint_source_block_index is None
                else checkpoint_source_block_index
            ),
            "is_cross_layer_transfer": bool(
                checkpoint_source_block_index is not None
                and checkpoint_source_block_index != model.block_index
            ),
            "eval_start_index": args.eval_start_index,
            "eval_num_videos": args.eval_num_videos,
            "evaluated_video_count": len(evaluated_indices),
            "evaluated_indices": evaluated_indices,
            "timesteps": timesteps,
            "evaluated_case_count": len(evaluated_cases),
            "eval_seed": args.eval_seed,
            "world_size": world_size,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "checkpoint_groups": group_specs,
            "checkpoints": checkpoint_specs,
            "include_softmax": args.eval_include_softmax,
            "include_random_gdn": args.eval_include_random_gdn,
        }
        summary = _summarize(all_records, candidates, config, all_failures)
        _atomic_write_json(output_path / "v1_summary.json", summary)
        _write_summary_csv(output_path / "v1_summary.csv", summary)
        _write_markdown_report(output_path / "v1_report.md", summary)
        _atomic_write_jsonl(output_path / "v1_records.jsonl", all_records)
        print(
            f"V1 complete: best={summary['best_checkpoint']} "
            f"report={output_path / 'v1_report.md'}",
            flush=True,
        )

    if world_size > 1:
        accelerator.wait_for_everyone()
    if args.eval_destroy_process_group and distributed.is_initialized():
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
