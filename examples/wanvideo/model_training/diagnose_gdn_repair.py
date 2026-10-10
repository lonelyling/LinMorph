"""Profile Wan softmax -> hybrid GDN errors at matched hidden states.

This is a read-only diagnostic.  For every video/timestep pair it runs the
original Wan teacher once, evaluates each selected GDN attention module on the
exact hidden state seen by the corresponding teacher attention, and then runs
the complete hybrid student.  The report recommends the highest-error blocks
for the short repair LoRA launcher.
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
from accelerate import InitProcessGroupKwargs
from accelerate.utils import set_seed

from diffsynth.core.data.uni_dataset import UnifiedDataset
from train_6gdn import WanTrainingModule6GDN, wan_gdn_parser


DEFAULT_GDN_BLOCKS = (
    "0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26"
)


def _parse_csv_ints(value, *, name, minimum, maximum):
    values = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique integer values")
    if any(value < minimum or value > maximum for value in values):
        raise ValueError(f"{name} values must be in [{minimum}, {maximum}]")
    return values


def _atomic_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, records):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _energy_metrics(prediction, target):
    prediction = prediction.detach().float()
    target = target.detach().float()
    delta = prediction - target
    error_energy = delta.square().sum()
    target_energy = target.square().sum().clamp_min(1e-12)
    target_norm = target.norm().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        prediction.reshape(1, -1), target.reshape(1, -1), dim=1, eps=1e-12
    )
    return {
        "mse": float(delta.square().mean().item()),
        "nmse": float((error_energy / target_energy).item()),
        "cosine": float(cosine.item()),
        "norm_ratio": float((prediction.norm() / target_norm).item()),
    }


def _gradient_nmse(prediction, target, *, token_grid=None):
    prediction = prediction.detach().float()
    target = target.detach().float()
    if token_grid is not None:
        frames, height, width = token_grid
        expected_tokens = frames * height * width
        if prediction.ndim != 3 or prediction.shape[1] != expected_tokens:
            raise ValueError(
                f"Cannot reshape attention output {tuple(prediction.shape)} "
                f"to token grid {token_grid}"
            )
        prediction = prediction.reshape(
            prediction.shape[0], frames, height, width, prediction.shape[-1]
        ).permute(0, 4, 1, 2, 3)
        target = target.reshape(
            target.shape[0], frames, height, width, target.shape[-1]
        ).permute(0, 4, 1, 2, 3)

    spatial_error = prediction.new_zeros(())
    spatial_target = prediction.new_zeros(())
    for dimension in (-1, -2):
        pred_gradient = torch.diff(prediction, dim=dimension)
        target_gradient = torch.diff(target, dim=dimension)
        spatial_error = spatial_error + (pred_gradient - target_gradient).square().sum()
        spatial_target = spatial_target + target_gradient.square().sum()

    temporal_error = prediction.new_zeros(())
    temporal_target = prediction.new_zeros(())
    if prediction.shape[-3] > 1:
        pred_gradient = torch.diff(prediction, dim=-3)
        target_gradient = torch.diff(target, dim=-3)
        temporal_error = (pred_gradient - target_gradient).square().sum()
        temporal_target = target_gradient.square().sum()
    return {
        "spatial_grad_nmse": float(
            (spatial_error / spatial_target.clamp_min(1e-12)).item()
        ),
        "temporal_grad_nmse": float(
            (temporal_error / temporal_target.clamp_min(1e-12)).item()
        ),
    }


def _prepare_inputs(model, data):
    inputs = model.get_pipeline_inputs(data)
    inputs = model.transfer_data_to_device(
        inputs, model.pipe.device, model.pipe.torch_dtype
    )
    with torch.no_grad():
        for unit in model.pipe.units:
            inputs = model.pipe.unit_runner(unit, model.pipe, *inputs)
    inputs_shared, inputs_posi, inputs_nega = inputs
    inputs_shared.pop("memory_tokens", None)
    inputs_shared["cfg_scale"] = 1.0
    return inputs_shared, inputs_posi, inputs_nega


def _noisy_latents(model, inputs_shared, timestep, seed):
    clean = inputs_shared.get("input_latents", inputs_shared["latents"])
    noise = model.pipe.generate_noise(
        clean.shape,
        seed=seed,
        rand_device="cpu",
        device=clean.device,
        torch_dtype=clean.dtype,
    )
    timesteps = torch.full(
        (clean.shape[0],),
        float(timestep),
        device=clean.device,
        dtype=model.pipe.torch_dtype,
    )
    noisy = model.pipe.scheduler.add_noise(clean, noise, timestep=timesteps).to(
        clean.dtype
    )
    if (
        inputs_shared.get("fuse_vae_embedding_in_latents", False)
        and inputs_shared.get("first_frame_latents") is not None
    ):
        noisy = noisy.clone()
        noisy[:, :, 0:1] = inputs_shared["first_frame_latents"]
    return noisy, timesteps


def _case_seed(base_seed, dataset_index, timestep):
    return int(
        (base_seed + dataset_index * 1_000_003 + timestep * 9_176)
        % ((1 << 63) - 1)
    )


def _evaluate_case(
    model,
    inputs_shared,
    inputs_posi,
    inputs_nega,
    noisy,
    timesteps,
    block_indices,
    dataset_index,
    timestep,
):
    fuse = inputs_shared.get("fuse_vae_embedding_in_latents", False)
    teacher_blocks = {}

    def make_output_hook(destination, block_index):
        def hook(_module, _args, output):
            if isinstance(output, tuple):
                output = output[0]
            destination[block_index] = output.detach()

        return hook

    teacher_handles = [
        model.teacher_dit.blocks[block_index].register_forward_hook(
            make_output_hook(teacher_blocks, block_index)
        )
        for block_index in block_indices
    ]
    try:
        with torch.no_grad():
            teacher_pred, captures = model._teacher_prediction_with_repair_features(
                model.pipe,
                noisy,
                timesteps,
                inputs_shared,
                inputs_posi,
                inputs_nega,
                1.0,
                fuse,
            )
    finally:
        for handle in teacher_handles:
            handle.remove()

    patch = tuple(int(value) for value in model.pipe.dit.patch_size)
    token_grid = tuple(
        int(noisy.shape[dimension + 2] // patch[dimension])
        for dimension in range(3)
    )
    records = []
    with torch.no_grad():
        for block_index in block_indices:
            hidden, freqs, teacher_attention = captures[block_index]
            student_attention = model.pipe.dit.blocks[block_index].self_attn(
                hidden, freqs
            )
            if isinstance(student_attention, tuple):
                student_attention = student_attention[0]
            metrics = _energy_metrics(student_attention, teacher_attention)
            metrics.update(
                _gradient_nmse(
                    student_attention, teacher_attention, token_grid=token_grid
                )
            )
            metrics.update(
                {
                    "scope": "attention",
                    "block_index": int(block_index),
                    "dataset_index": int(dataset_index),
                    "timestep": int(timestep),
                }
            )
            records.append(metrics)
            del hidden, freqs, teacher_attention, student_attention

        student_blocks = {}
        student_handles = [
            model.pipe.dit.blocks[block_index].register_forward_hook(
                make_output_hook(student_blocks, block_index)
            )
            for block_index in block_indices
        ]
        try:
            student_pred = model._model_prediction_with_cfg(
                model.pipe,
                model.pipe.dit,
                noisy,
                timesteps,
                inputs_shared,
                inputs_posi,
                inputs_nega,
                1.0,
                fuse,
                use_checkpointing=False,
            )
        finally:
            for handle in student_handles:
                handle.remove()

        for block_index in block_indices:
            metrics = _energy_metrics(
                student_blocks[block_index], teacher_blocks[block_index]
            )
            metrics.update(
                _gradient_nmse(
                    student_blocks[block_index],
                    teacher_blocks[block_index],
                    token_grid=token_grid,
                )
            )
            metrics.update(
                {
                    "scope": "rollout_block",
                    "block_index": int(block_index),
                    "dataset_index": int(dataset_index),
                    "timestep": int(timestep),
                }
            )
            records.append(metrics)
        output_metrics = _energy_metrics(student_pred, teacher_pred)
        output_metrics.update(_gradient_nmse(student_pred, teacher_pred))
        output_metrics.update(
            {
                "scope": "output",
                "block_index": -1,
                "dataset_index": int(dataset_index),
                "timestep": int(timestep),
            }
        )
        records.append(output_metrics)
    return records


def _mean_metrics(records):
    result = {"count": len(records)}
    for name in (
        "mse",
        "nmse",
        "cosine",
        "norm_ratio",
        "spatial_grad_nmse",
        "temporal_grad_nmse",
    ):
        values = [float(record[name]) for record in records]
        result[name] = statistics.fmean(values)
        result[f"{name}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
    return result


def _summarize(records, block_indices, low_noise_max, recommend_count):
    attention = [record for record in records if record["scope"] == "attention"]
    rollout = [record for record in records if record["scope"] == "rollout_block"]
    output = [record for record in records if record["scope"] == "output"]
    by_block = {}
    ranking = []
    for block_index in block_indices:
        items = [r for r in attention if r["block_index"] == block_index]
        rollout_items = [r for r in rollout if r["block_index"] == block_index]
        low_noise = [r for r in items if r["timestep"] <= low_noise_max]
        if not low_noise:
            low_noise = items
        item = _mean_metrics(items)
        item["low_noise"] = _mean_metrics(low_noise)
        item["rollout"] = _mean_metrics(rollout_items)
        by_block[str(block_index)] = item
        score = (
            item["low_noise"]["nmse"]
            + 0.25 * item["low_noise"]["spatial_grad_nmse"]
            + 0.25 * item["low_noise"]["temporal_grad_nmse"]
        )
        ranking.append((score, block_index))
    ranking.sort(reverse=True)
    recommended = [index for _, index in ranking[:recommend_count]]
    return {
        "attention_overall": _mean_metrics(attention),
        "rollout_block_overall": _mean_metrics(rollout),
        "output_overall": _mean_metrics(output),
        "by_block": by_block,
        "ranking": [
            {"block_index": index, "repair_score": score}
            for score, index in ranking
        ],
        "recommended_blocks": recommended,
    }


def _write_csv(path: Path, summary):
    fields = [
        "block_index",
        "repair_score",
        "nmse",
        "cosine",
        "spatial_grad_nmse",
        "temporal_grad_nmse",
        "low_noise_nmse",
        "low_noise_spatial_grad_nmse",
        "low_noise_temporal_grad_nmse",
    ]
    scores = {
        item["block_index"]: item["repair_score"]
        for item in summary["ranking"]
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for key, item in summary["by_block"].items():
            low = item["low_noise"]
            writer.writerow(
                {
                    "block_index": int(key),
                    "repair_score": scores[int(key)],
                    "nmse": item["nmse"],
                    "cosine": item["cosine"],
                    "spatial_grad_nmse": item["spatial_grad_nmse"],
                    "temporal_grad_nmse": item["temporal_grad_nmse"],
                    "low_noise_nmse": low["nmse"],
                    "low_noise_spatial_grad_nmse": low["spatial_grad_nmse"],
                    "low_noise_temporal_grad_nmse": low["temporal_grad_nmse"],
                }
            )
    os.replace(temporary, path)


def _write_report(path: Path, summary):
    lines = [
        "# Wan2.1 28k GDN repair diagnostic",
        "",
        f"- Checkpoint: `{summary['config']['student_checkpoint']}`",
        f"- Videos/timesteps: `{summary['config']['num_videos']}` / `{summary['config']['timesteps']}`",
        f"- Low-noise cutoff: `t <= {summary['config']['low_noise_max']}`",
        f"- Recommended repair blocks: `{','.join(map(str, summary['recommended_blocks']))}`",
        "",
        "Use the recommendation with:",
        "",
        "```bash",
        "export REPAIR_GDN_BLOCKS=" + ",".join(map(str, summary["recommended_blocks"])),
        "bash scripts/launch_train_wan21_13b_step28000_gdn_repair_lora_a800.sh",
        "```",
        "",
        "| block | repair score | low-noise NMSE | spatial grad NMSE | temporal grad NMSE |",
        "|---:|---:|---:|---:|---:|",
    ]
    for item in summary["ranking"]:
        block = str(item["block_index"])
        low = summary["by_block"][block]["low_noise"]
        lines.append(
            f"| {block} | {item['repair_score']:.6f} | {low['nmse']:.6f} | "
            f"{low['spatial_grad_nmse']:.6f} | {low['temporal_grad_nmse']:.6f} |"
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def diagnostic_parser():
    parser = wan_gdn_parser()
    parser.description = "Matched-hidden-state GDN repair diagnostic"
    parser.add_argument(
        "--diag_num_videos",
        type=int,
        default=int(os.environ.get("DIFFSYNTH_DIAG_NUM_VIDEOS", "8")),
    )
    parser.add_argument(
        "--diag_start_index",
        type=int,
        default=int(os.environ.get("DIFFSYNTH_DIAG_START_INDEX", "12000")),
    )
    parser.add_argument(
        "--diag_timesteps",
        default=os.environ.get(
            "DIFFSYNTH_DIAG_TIMESTEPS", "50,100,200,350,500,700,850,950"
        ),
    )
    parser.add_argument(
        "--diag_block_indices",
        default=os.environ.get(
            "DIFFSYNTH_DIAG_BLOCK_INDICES", DEFAULT_GDN_BLOCKS
        ),
    )
    parser.add_argument(
        "--diag_low_noise_max",
        type=int,
        default=int(os.environ.get("DIFFSYNTH_DIAG_LOW_NOISE_MAX", "350")),
    )
    parser.add_argument(
        "--diag_recommend_count",
        type=int,
        default=int(os.environ.get("DIFFSYNTH_DIAG_RECOMMEND_COUNT", "6")),
    )
    parser.add_argument(
        "--diag_seed",
        type=int,
        default=int(os.environ.get("DIFFSYNTH_DIAG_SEED", "20260903")),
    )
    return parser


def main():
    started = time.monotonic()
    os.environ.setdefault("DIFFSYNTH_T2V_TRAINING", "1")
    os.environ.setdefault("DIFFSYNTH_REPAIR_FEATURE_WEIGHT", "0")
    args = diagnostic_parser().parse_args()
    if not args.student_init_ckpt or not Path(args.student_init_ckpt).is_file():
        raise FileNotFoundError(
            f"--student_init_ckpt must point to the 28k checkpoint: {args.student_init_ckpt}"
        )
    if args.diag_num_videos <= 0 or args.diag_recommend_count <= 0:
        raise ValueError("Diagnostic video/recommend counts must be positive")
    timesteps = _parse_csv_ints(
        args.diag_timesteps, name="diag_timesteps", minimum=1, maximum=999
    )
    blocks = _parse_csv_ints(
        args.diag_block_indices,
        name="diag_block_indices",
        minimum=0,
        maximum=29,
    )
    os.environ["DIFFSYNTH_REPAIR_FEATURE_BLOCKS"] = ",".join(map(str, blocks))

    timeout = timedelta(
        seconds=int(os.environ.get("DIFFSYNTH_NCCL_TIMEOUT_SEC", "7200"))
    )
    accelerator = accelerate.Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timeout)]
    )
    set_seed(args.diag_seed, device_specific=False)
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

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
            seed=args.diag_seed,
        )
    else:
        dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.dataset_metadata_path,
            repeat=1,
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

    model = WanTrainingModule6GDN(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models="",
        use_gradient_checkpointing=False,
        extra_inputs=None,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        device=accelerator.device,
        task="kd_sft",
        student_init_ckpt=args.student_init_ckpt,
        gdn_group_size=args.gdn_group_size,
        gdn_block_indices=args.gdn_block_indices,
        gdn_expand_v=args.gdn_expand_v,
        gdn_head_dim=args.gdn_head_dim,
        gdn_hidden_size=args.gdn_hidden_size,
        gdn_num_heads=args.gdn_num_heads,
        no_teacher=False,
        kd_loss_weight=0.0,
    )
    model.eval()
    model.requires_grad_(False)

    rank = accelerator.process_index
    world_size = accelerator.num_processes
    local_records = []
    assigned = list(range(rank, args.diag_num_videos, world_size))
    for ordinal in assigned:
        dataset_index = (args.diag_start_index + ordinal) % len(dataset)
        data = dataset[dataset_index]
        inputs_shared, inputs_posi, inputs_nega = _prepare_inputs(model, data)
        del data
        for timestep in timesteps:
            seed = _case_seed(args.diag_seed, dataset_index, timestep)
            noisy, timestep_tensor = _noisy_latents(
                model, inputs_shared, timestep, seed
            )
            local_records.extend(
                _evaluate_case(
                    model,
                    inputs_shared,
                    inputs_posi,
                    inputs_nega,
                    noisy,
                    timestep_tensor,
                    blocks,
                    dataset_index,
                    timestep,
                )
            )
            del noisy, timestep_tensor
        print(
            f"[rank {rank}] diagnostic video {ordinal + 1}/{args.diag_num_videos} "
            f"dataset_index={dataset_index}",
            flush=True,
        )

    _atomic_jsonl(output_path / f"records.rank{rank}.jsonl", local_records)
    if world_size > 1:
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        records = []
        for process_rank in range(world_size):
            with (output_path / f"records.rank{process_rank}.jsonl").open(
                encoding="utf-8"
            ) as file:
                records.extend(json.loads(line) for line in file if line.strip())
        expected = args.diag_num_videos * len(timesteps) * (2 * len(blocks) + 1)
        if len(records) != expected:
            raise RuntimeError(f"Expected {expected} metric rows, got {len(records)}")
        if any(
            not math.isfinite(float(record[name]))
            for record in records
            for name in (
                "mse",
                "nmse",
                "cosine",
                "norm_ratio",
                "spatial_grad_nmse",
                "temporal_grad_nmse",
            )
        ):
            raise RuntimeError("Diagnostic produced a non-finite metric")
        summary = _summarize(
            records,
            blocks,
            args.diag_low_noise_max,
            min(args.diag_recommend_count, len(blocks)),
        )
        summary["config"] = {
            "student_checkpoint": str(Path(args.student_init_ckpt).resolve()),
            "num_videos": args.diag_num_videos,
            "timesteps": timesteps,
            "blocks": blocks,
            "low_noise_max": args.diag_low_noise_max,
            "seed": args.diag_seed,
            "world_size": world_size,
            "shape": [args.num_frames, args.height, args.width],
        }
        summary["runtime_seconds"] = time.monotonic() - started
        _atomic_json(output_path / "summary.json", summary)
        _atomic_jsonl(output_path / "records.jsonl", records)
        _write_csv(output_path / "block_ranking.csv", summary)
        _write_report(output_path / "report.md", summary)
        print(
            "GDN diagnostic complete; recommended blocks="
            + ",".join(map(str, summary["recommended_blocks"]))
            + f" report={output_path / 'report.md'}",
            flush=True,
        )

    if world_size > 1:
        accelerator.wait_for_everyone()
    if distributed.is_initialized():
        distributed.destroy_process_group()


if __name__ == "__main__":
    main()
