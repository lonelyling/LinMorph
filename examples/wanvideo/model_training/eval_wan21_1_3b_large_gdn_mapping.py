"""Held-out feature evaluation for the Wan2.1-1.3B -> exact-large-GDN mapper.

This evaluator is intentionally separate from eval_single_layer_weight_mapping.py.
The latter targets the older format-v5 square/repeat mapper, while the checkpoints
under output/wan21_1_3b_weight_mapping* use the format-v9 SubheadMix mapper from
``from_congyue/train_single_layer_linear_adapter.py``.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import time
from datetime import timedelta
from pathlib import Path

import accelerate
import torch
import torch.distributed as distributed
import torch.nn.functional as F
from accelerate import InitProcessGroupKwargs
from accelerate.utils import set_seed

from diffsynth.core.data.uni_dataset import UnifiedDataset
from train_single_layer_linear_adapter import (
    WanSingleLayerWeightMappingModule,
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

INTERVENTION_NAMES = (
    "none",
    "diagonal_only",
    "permute_unmatched_sources",
    "no_unmatched",
    "no_post_residual",
)


def _tensor_l1(parameters) -> float:
    return float(
        sum(parameter.detach().float().abs().sum().item() for parameter in parameters)
    )


def _apply_mapping_intervention(model, intervention: str):
    """Apply a controlled, inference-only ablation to the shared mapper.

    The format-v9 N30 checkpoint shares one mapper across all selected blocks,
    so changing mapped_attention also changes every block-specific view.
    diagonal_only removes cross-projection coefficients for the matched
    q/k/v/o targets while retaining the learned mappings for unmatched GDN
    targets. no_unmatched removes all learned routes into GDN-only gate,
    update, and recurrent-state parameters. no_post_residual isolates the
    contribution of the dense post-expansion residual from the compact map.
    """
    if intervention not in INTERVENTION_NAMES:
        raise ValueError(
            f"Unknown mapping intervention {intervention!r}; "
            f"expected one of {INTERVENTION_NAMES}"
        )

    mapper = model.mapped_attention
    groups = {
        "mixing": [mapper.mixing],
        "auxiliary": [
            parameter
            for container in (
                mapper.auxiliary_mixing,
                mapper.auxiliary_headwise_mixing,
                mapper.auxiliary_full_mixing,
                mapper.auxiliary_channel_mixing,
            )
            for parameter in container.values()
        ],
        "post_full_residual": list(mapper.post_full_residual.values()),
    }
    before = {name: _tensor_l1(parameters) for name, parameters in groups.items()}

    with torch.no_grad():
        if intervention == "diagonal_only":
            for target_index, target_name in enumerate(("q", "k", "v", "o")):
                source_mask = torch.zeros_like(mapper.mixing[target_index])
                source_mask[target_index] = 1
                mapper.mixing[target_index].mul_(source_mask)
                if target_name in mapper.subhead_mixing:
                    mask = torch.zeros_like(mapper.subhead_mixing[target_name])
                    mask[..., target_index] = 1
                    mapper.subhead_mixing[target_name].mul_(mask)
                if target_name in mapper.channel_mixing:
                    mask = torch.zeros_like(mapper.channel_mixing[target_name])
                    mask[..., target_index] = 1
                    mapper.channel_mixing[target_name].mul_(mask)
        elif intervention == "permute_unmatched_sources":
            mapper.mixing[4:].copy_(torch.roll(mapper.mixing[4:], shifts=1, dims=-1))
            mapper.headwise_mixing.copy_(
                torch.roll(mapper.headwise_mixing, shifts=1, dims=-1)
            )
            if "g" in mapper.subhead_mixing:
                mapper.subhead_mixing["g"].copy_(
                    torch.roll(mapper.subhead_mixing["g"], shifts=1, dims=-1)
                )
            if "g" in mapper.channel_mixing:
                mapper.channel_mixing["g"].copy_(
                    torch.roll(mapper.channel_mixing["g"], shifts=1, dims=-1)
                )
            for parameter in groups["auxiliary"]:
                parameter.copy_(torch.roll(parameter, shifts=1, dims=-1))
        elif intervention == "no_unmatched":
            # g and beta are the two main GDN-only target projections.
            mapper.mixing[4:].zero_()
            mapper.headwise_mixing.zero_()
            if "g" in mapper.subhead_mixing:
                mapper.subhead_mixing["g"].zero_()
            if "g" in mapper.channel_mixing:
                mapper.channel_mixing["g"].zero_()
            for parameter in groups["auxiliary"]:
                parameter.zero_()
            for target_name in ("g", "beta"):
                if target_name in mapper.post_full_residual:
                    mapper.post_full_residual[target_name].zero_()
        elif intervention == "no_post_residual":
            for parameter in groups["post_full_residual"]:
                parameter.zero_()

    after = {name: _tensor_l1(parameters) for name, parameters in groups.items()}
    return {
        "name": intervention,
        "l1_before": before,
        "l1_after": after,
    }


def _atomic_write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, records) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _parse_csv_ints(value: str, *, name: str, minimum: int, maximum: int):
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} contains duplicates: {values}")
    if any(item < minimum or item > maximum for item in values):
        raise ValueError(
            f"{name} values must be in [{minimum}, {maximum}], got {values}"
        )
    return values


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


def _metric_summary(records):
    summary = {"count": len(records)}
    for metric in METRIC_NAMES:
        values = [float(record[metric]) for record in records]
        summary[f"{metric}_mean"] = statistics.fmean(values)
        summary[f"{metric}_std"] = (
            statistics.pstdev(values) if len(values) > 1 else 0.0
        )
    return summary


def _case_seed(eval_seed: int, dataset_index: int, timestep: int) -> int:
    modulus = (1 << 63) - 1
    return int(
        (eval_seed + dataset_index * 1_000_003 + int(timestep) * 9_176) % modulus
    )


def _load_exact_dataset_item(dataset, dataset_index: int, retries: int):
    if dataset.is_webdata or dataset.load_from_cache:
        raise ValueError("Exact-index evaluation requires a regular indexed dataset")
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return dataset._getitem_normal(dataset_index)
        except Exception as error:  # Dataset/network failures are recorded below.
            last_error = error
            if attempt < retries:
                time.sleep(min(5.0, float(attempt)))
    raise RuntimeError(
        f"Exact dataset index {dataset_index} failed after {retries} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    )


def _prepare_video_inputs(model, data, seed: int):
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
        # The standard Wan2.1 teacher keeps its time MLP in the pipeline dtype.
        # Using FP32 here (as the older square-mapper evaluator does) fails at
        # the first BF16 Linear layer before any mapping is evaluated.
        dtype=model.pipe.torch_dtype,
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


def _evaluate_case(
    model,
    model_kwargs,
    target_blocks,
    label,
    checkpoint_path,
    dataset_index,
    timestep,
    seed,
):
    teacher_by_block = model._capture_teacher_block_features(
        model_kwargs,
        block_indices=target_blocks,
    )
    records = []
    with torch.no_grad():
        for block_index in target_blocks:
            teacher = teacher_by_block[block_index]
            mapper = model._mapper_for_block(block_index)
            mapped_parameters = mapper.materialized_parameters()
            mapped_attention, mapped_block = model._forward_mapped_block(
                teacher["block_inputs"],
                mapped_parameters=mapped_parameters,
                block_index=block_index,
            )
            attention = _feature_metrics(
                mapped_attention,
                teacher["attention_output"],
            )
            block = _feature_metrics(mapped_block, teacher["block_output"])
            records.append(
                {
                    "candidate": label,
                    "checkpoint_path": checkpoint_path,
                    "block_index": int(block_index),
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
            mapper.last_output = None
            del mapped_parameters, mapped_attention, mapped_block
    del teacher_by_block
    return records


def _group_summary(records, key):
    grouped = {}
    for record in records:
        grouped.setdefault(str(record[key]), []).append(record)
    return {
        name: _metric_summary(items)
        for name, items in sorted(grouped.items(), key=lambda item: int(item[0]))
    }


def _write_summary_csv(path: Path, summary) -> None:
    fieldnames = ["scope", "value", "count"] + [
        f"{metric}_{suffix}"
        for metric in METRIC_NAMES
        for suffix in ("mean", "std")
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for scope, value, item in [
            ("overall", "all", summary["overall"]),
            *(
                ("block", block_index, block_summary)
                for block_index, block_summary in summary["by_block"].items()
            ),
            *(
                ("timestep", timestep, timestep_summary)
                for timestep, timestep_summary in summary["by_timestep"].items()
            ),
        ]:
            row = {"scope": scope, "value": value}
            row.update({name: item[name] for name in fieldnames if name in item})
            writer.writerow(row)
    os.replace(temporary, path)


def _write_report(path: Path, summary) -> None:
    config = summary["config"]
    overall = summary["overall"]
    lines = [
        f"# {summary['candidate']} held-out mapping evaluation",
        "",
        f"- Checkpoint: `{summary['checkpoint_path'] or 'manual-copy baseline'}`",
        (
            f"- Shape: `{config['num_frames']}x{config['height']}x"
            f"{config['width']}`"
        ),
        f"- Dataset indices: `{config['evaluated_indices']}`",
        f"- Timesteps: `{config['timesteps']}`",
        f"- Target blocks: `{config['target_blocks']}`",
        f"- Total wall time: `{summary['runtime_seconds']:.1f}s`",
        f"- Peak allocated GPU memory: `{summary['peak_gpu_memory_gib']:.2f} GiB`",
        "",
        "## Overall",
        "",
        "| attention NMSE | attention cosine | block NMSE | block cosine | records |",
        "|---:|---:|---:|---:|---:|",
        (
            f"| {overall['attention_nmse_mean']:.8f} | "
            f"{overall['attention_cosine_mean']:.8f} | "
            f"{overall['block_nmse_mean']:.8f} | "
            f"{overall['block_cosine_mean']:.8f} | {overall['count']} |"
        ),
        "",
        "## Per block",
        "",
        "| block | attention NMSE | attention cosine | block NMSE | block cosine |",
        "|---:|---:|---:|---:|---:|",
    ]
    for block_index, item in summary["by_block"].items():
        lines.append(
            f"| {block_index} | {item['attention_nmse_mean']:.8f} | "
            f"{item['attention_cosine_mean']:.8f} | "
            f"{item['block_nmse_mean']:.8f} | "
            f"{item['block_cosine_mean']:.8f} |"
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def evaluation_parser():
    parser = weight_mapping_parser()
    parser.description = (
        "Held-out feature evaluation for Wan2.1-1.3B exact-large-GDN mappings"
    )
    parser.add_argument(
        "--eval_checkpoint",
        default=os.environ.get("MAPPING_EVAL_CHECKPOINT", ""),
        help="Mapping checkpoint; omit for the deterministic Manual Copy baseline.",
    )
    parser.add_argument(
        "--eval_label",
        default=os.environ.get("MAPPING_EVAL_LABEL", "candidate"),
    )
    parser.add_argument(
        "--eval_intervention",
        choices=INTERVENTION_NAMES,
        default=os.environ.get("MAPPING_EVAL_INTERVENTION", "none"),
        help=(
            "Inference-only Section 4.1 intervention applied after loading the "
            "mapping checkpoint."
        ),
    )
    parser.add_argument(
        "--eval_start_index",
        type=int,
        default=int(os.environ.get("MAPPING_EVAL_START_INDEX", 8192)),
    )
    parser.add_argument(
        "--eval_num_videos",
        type=int,
        default=int(os.environ.get("MAPPING_EVAL_NUM_VIDEOS", 8)),
    )
    parser.add_argument(
        "--eval_timesteps",
        default=os.environ.get("MAPPING_EVAL_TIMESTEPS", "100,300,500,700,900"),
    )
    parser.add_argument(
        "--eval_block_indices",
        default=os.environ.get(
            "MAPPING_EVAL_BLOCK_INDICES", ",".join(str(index) for index in range(30))
        ),
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
    return parser


def main():
    started_at = time.monotonic()
    os.environ.setdefault("DIFFSYNTH_T2V_TRAINING", "1")
    timeout = timedelta(
        seconds=int(os.environ.get("DIFFSYNTH_NCCL_TIMEOUT_SEC", "7200"))
    )
    args = evaluation_parser().parse_args()
    if args.eval_start_index < 0:
        raise ValueError("--eval_start_index must be non-negative")
    if args.eval_num_videos <= 0:
        raise ValueError("--eval_num_videos must be positive")
    if args.eval_exact_retries <= 0 or args.eval_scan_multiplier <= 0:
        raise ValueError("Evaluation retry/scan settings must be positive")

    timesteps = _parse_csv_ints(
        args.eval_timesteps,
        name="eval_timesteps",
        minimum=0,
        maximum=1000,
    )
    target_blocks = _parse_csv_ints(
        args.eval_block_indices,
        name="eval_block_indices",
        minimum=0,
        maximum=29,
    )
    checkpoint_path = (
        str(Path(args.eval_checkpoint).expanduser().resolve())
        if args.eval_checkpoint
        else None
    )
    if checkpoint_path and not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Mapping checkpoint does not exist: {checkpoint_path}")

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=1,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timeout)],
    )
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    set_seed(args.eval_seed, device_specific=False)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(accelerator.device)

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
            f"eval_start_index={args.eval_start_index} is outside dataset length "
            f"{len(dataset)}"
        )

    # Instantiate only blocks that are actually evaluated. The format-v9
    # checkpoint is shared across all 30 layers, so the training module allows
    # loading it into any multi-block subset. This makes smoke/diagnostic runs
    # much cheaper while the full profile still instantiates all 30 blocks.
    model_blocks = list(target_blocks)
    if checkpoint_path and len(model_blocks) == 1:
        model_blocks.append(0 if model_blocks[0] != 0 else 1)
    model_block_spec = ",".join(str(index) for index in model_blocks)
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
        student_init_ckpt=checkpoint_path,
        block_index=model_blocks[0],
        gdn_block_indices=model_block_spec,
        mapping_num_layers=len(model_blocks),
        middle_layer_start=args.middle_layer_start,
        middle_layer_end=args.middle_layer_end,
        mapping_seed=args.mapping_seed,
        mixing_init="manual_copy",
        low_rank=args.low_rank,
        full_residual=args.full_residual,
        post_full_residual=args.post_full_residual,
        postmix=args.postmix,
        mix_granularity=args.mix_granularity,
        mix_order=args.mix_order,
        source_specific_expansion=args.source_specific_expansion,
        low_rank_init_std=args.low_rank_init_std,
        gdn_expand_v=args.gdn_expand_v,
        gdn_head_dim=args.gdn_head_dim,
        gdn_hidden_size=args.gdn_hidden_size,
        gdn_num_heads=args.gdn_num_heads,
        shape_adapter=args.mapping_shape_adapter,
        gdn_use_short_conv=args.gdn_use_short_conv,
        identity_short_conv=args.identity_short_conv,
        map_auxiliary_parameters=args.map_auxiliary_parameters,
        train_base_mapping=args.train_base_mapping,
        capture_random_initialization=False,
        attention_loss_weight=1.0,
        block_loss_weight=1.0,
        mapping_regularization=0.0,
        normalized_mse=False,
        mapping_log_interval=args.mapping_log_interval,
        mapping_blocks_per_step=0,
        checkpoint_mapped_block=False,
    )
    model.eval()
    model.requires_grad_(False)
    intervention_stats = _apply_mapping_intervention(model, args.eval_intervention)
    accelerator.print(
        f"Candidate={args.eval_label} checkpoint={checkpoint_path or 'manual_copy'} "
        f"intervention={args.eval_intervention} "
        f"shape={args.num_frames}x{args.height}x{args.width} "
        f"blocks={target_blocks} videos={args.eval_num_videos} "
        f"timesteps={timesteps} world_size={world_size}"
    )

    target_count = len(range(rank, args.eval_num_videos, world_size))
    scan_limit = max(1, target_count) * args.eval_scan_multiplier
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
                    model_kwargs,
                    target_blocks,
                    args.eval_label,
                    checkpoint_path,
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
            f"Rank {rank} evaluated only {successful_videos}/{target_count} videos "
            f"after scanning {scanned} exact indices"
        )

    _atomic_write_jsonl(
        output_path / f"records.rank{rank}.jsonl", local_records
    )
    _atomic_write_json(
        output_path / f"failures.rank{rank}.json", local_failures
    )
    if world_size > 1:
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        records = []
        failures = []
        for process_rank in range(world_size):
            record_path = output_path / f"records.rank{process_rank}.jsonl"
            with record_path.open("r", encoding="utf-8") as file:
                records.extend(json.loads(line) for line in file if line.strip())
            failures.extend(
                json.loads(
                    (output_path / f"failures.rank{process_rank}.json").read_text(
                        encoding="utf-8"
                    )
                )
            )

        evaluated_indices = sorted({record["dataset_index"] for record in records})
        expected_record_count = (
            args.eval_num_videos * len(timesteps) * len(target_blocks)
        )
        if len(records) != expected_record_count:
            raise RuntimeError(
                f"Expected {expected_record_count} records, got {len(records)}"
            )
        for record in records:
            for metric in METRIC_NAMES:
                if not math.isfinite(float(record[metric])):
                    raise RuntimeError(
                        f"Non-finite {metric}: block={record['block_index']} "
                        f"dataset_index={record['dataset_index']} "
                        f"timestep={record['timestep']}"
                    )

        peak_gpu_memory_gib = (
            torch.cuda.max_memory_allocated(accelerator.device) / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        )
        summary = {
            "candidate": args.eval_label,
            "checkpoint_path": checkpoint_path,
            "config": {
                "eval_start_index": args.eval_start_index,
                "eval_num_videos": args.eval_num_videos,
                "evaluated_indices": evaluated_indices,
                "timesteps": timesteps,
                "target_blocks": target_blocks,
                "eval_seed": args.eval_seed,
                "world_size": world_size,
                "height": args.height,
                "width": args.width,
                "num_frames": args.num_frames,
                "mix_granularity": args.mix_granularity,
                "mix_order": args.mix_order,
                "post_full_residual": bool(args.post_full_residual),
                "intervention": args.eval_intervention,
            },
            "intervention_stats": intervention_stats,
            "overall": _metric_summary(records),
            "by_block": _group_summary(records, "block_index"),
            "by_timestep": _group_summary(records, "timestep"),
            "failures": failures,
            "runtime_seconds": time.monotonic() - started_at,
            "peak_gpu_memory_gib": peak_gpu_memory_gib,
        }
        _atomic_write_json(output_path / "summary.json", summary)
        _write_summary_csv(output_path / "summary.csv", summary)
        _write_report(output_path / "report.md", summary)
        _atomic_write_jsonl(output_path / "records.jsonl", records)
        print(
            f"Evaluation complete: candidate={args.eval_label} "
            f"block_nmse={summary['overall']['block_nmse_mean']:.8f} "
            f"report={output_path / 'report.md'}",
            flush=True,
        )

    if world_size > 1:
        accelerator.wait_for_everyone()
    if distributed.is_initialized():
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
