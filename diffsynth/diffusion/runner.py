import os, sys, torch, threading, time, shutil, traceback
from tqdm import tqdm
from accelerate import Accelerator
from ..core import load_state_dict
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger
from datetime import datetime


def _identity_collate(batch):
    """batch_size=1 时直接解包，替换 lambda x: x[0] 以支持 num_workers > 0 序列化。"""
    return batch[0]


def collate_extract_before_multi_sample_collate_fn(data):
    assert len(data) == 1
    res = collate_multi_sample_collate_fn(data[0])
    return res


def collate_multi_sample_collate_fn(data):
    res = {}
    keys = list(data[0].keys())
    for key in keys:
        res[key] = []
    for item in data:
        for key in keys:
            res[key].append(item.get(key, None))
    return res


def multi_sample_collate_fn(data):
    res = {}
    keys = list(data[0].keys())
    for key in keys:
        res[key] = []
    for item in data:
        for key in keys:
            res[key].append(item.get(key, None))
    return res


def warmup_lr_fn(step, warmup_steps=1000, min_alpha=0.01):
    # in accelerator, `scheduler steps` equals to `number of samples`
    # 10000 means warmup for 10000 samples, not iterations.
    if step < warmup_steps:
        alpha = (1 - min_alpha) * step / warmup_steps + min_alpha
    else:
        alpha = 1
    return alpha


def _set_optimizer_foreach(optimizer, enabled):
    """Apply foreach mode through Accelerate/DeepSpeed optimizer wrappers."""
    pending = [optimizer]
    visited = set()
    changed_groups = 0
    while pending:
        current = pending.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))

        defaults = getattr(current, "defaults", None)
        if isinstance(defaults, dict) and "foreach" in defaults:
            defaults["foreach"] = enabled
        param_groups = getattr(current, "param_groups", None)
        if param_groups is not None:
            for group in param_groups:
                if "foreach" in group:
                    group["foreach"] = enabled
                    changed_groups += 1

        for attribute in ("optimizer", "base_optimizer"):
            nested = getattr(current, attribute, None)
            if nested is not None and nested is not current:
                pending.append(nested)
    return changed_groups


def _format_dtype(dtype):
    return str(dtype).removeprefix("torch.")


def _summarize_trainable_dtypes(model):
    summary = {}
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        bucket = summary.setdefault(
            parameter.dtype,
            {"tensors": 0, "elements": 0},
        )
        bucket["tensors"] += 1
        bucket["elements"] += parameter.numel()
    return ", ".join(
        f"{_format_dtype(dtype)}={values['tensors']} tensors/"
        f"{values['elements']:,} elements"
        for dtype, values in sorted(summary.items(), key=lambda item: str(item[0]))
    )


def _optimizer_group_max_elements():
    raw_value = os.environ.get("DIFFSYNTH_OPTIMIZER_GROUP_MAX_ELEMENTS", "0")
    try:
        max_elements = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "DIFFSYNTH_OPTIMIZER_GROUP_MAX_ELEMENTS must be a non-negative "
            f"integer, got {raw_value!r}"
        ) from exc
    if max_elements < 0:
        raise ValueError(
            "DIFFSYNTH_OPTIMIZER_GROUP_MAX_ELEMENTS must be non-negative, "
            f"got {max_elements}"
        )
    return max_elements


def _build_optimizer_param_groups(model, weight_decay, max_elements=0):
    """Keep ZeRO groups dtype-homogeneous and optionally cap flattened groups."""
    buckets = {}
    for parameter in model.trainable_modules():
        parameter_weight_decay = (
            0.0 if getattr(parameter, "_no_weight_decay", False) else weight_decay
        )
        key = (parameter.dtype, parameter_weight_decay)
        buckets.setdefault(key, []).append(parameter)

    if not buckets:
        raise ValueError("No trainable parameters were found.")

    dtype_priority = {
        torch.bfloat16: 0,
        torch.float16: 1,
        torch.float32: 2,
        torch.float64: 3,
    }
    ordered_buckets = sorted(
        buckets.items(),
        key=lambda item: (
            dtype_priority.get(item[0][0], 99),
            item[0][1],
            str(item[0][0]),
        ),
    )
    optimizer_groups = []
    for (_, parameter_weight_decay), parameters in ordered_buckets:
        parameter_chunks = [parameters]
        if max_elements > 0:
            parameter_chunks = []
            current_chunk = []
            current_elements = 0
            for parameter in parameters:
                parameter_elements = parameter.numel()
                if (
                    current_chunk
                    and current_elements + parameter_elements > max_elements
                ):
                    parameter_chunks.append(current_chunk)
                    current_chunk = []
                    current_elements = 0
                current_chunk.append(parameter)
                current_elements += parameter_elements
            if current_chunk:
                parameter_chunks.append(current_chunk)

        optimizer_groups.extend(
            {
                "params": parameter_chunk,
                "weight_decay": parameter_weight_decay,
            }
            for parameter_chunk in parameter_chunks
        )
    return optimizer_groups


def _capture_trainable_dtypes(model):
    return {
        name: parameter.dtype
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _validate_trainable_dtypes(model, expected_dtypes):
    actual_dtypes = _capture_trainable_dtypes(model)
    mismatches = []
    for name, expected_dtype in expected_dtypes.items():
        actual_dtype = actual_dtypes.get(name)
        if actual_dtype != expected_dtype:
            mismatches.append((name, expected_dtype, actual_dtype))

    if mismatches:
        sample = ", ".join(
            f"{name}: {_format_dtype(expected)} -> "
            f"{_format_dtype(actual) if actual is not None else 'missing'}"
            for name, expected, actual in mismatches[:8]
        )
        raise RuntimeError(
            "Trainable parameter dtypes changed during accelerator.prepare; "
            f"{len(mismatches)} mismatches. Sample: {sample}"
        )


def _make_training_dataloader(dataset, *, num_workers, batch_size, drop_last=True):
    """Build a dataloader with reusable worker/prefetch knobs.

    The defaults stay conservative, but 16-GPU runs often benefit from keeping workers
    alive and prefetching a small number of batches per worker.
    """
    loader_kwargs = dict(
        shuffle=False,
        collate_fn=_identity_collate,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=drop_last,
    )
    if num_workers and num_workers > 0:
        loader_kwargs["persistent_workers"] = os.environ.get("DIFFSYNTH_DATALOADER_PERSISTENT_WORKERS", "1") == "1"
        loader_kwargs["prefetch_factor"] = max(
            1, int(os.environ.get("DIFFSYNTH_DATALOADER_PREFETCH_FACTOR", "2"))
        )
    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


# ── 训练进度元数据文件：用于跨卡数正确续跑 ────────────────────────────────────
# 主单位 = 「per-rank step」（跟历史 checkpoint 命名一致）：
#   global_step = 当前 rank 已完成的 batch 数；DDP 下所有 rank 同步推进。
#   checkpoint-{global_step} ↔ 进度条 {global_step}/{steps_per_epoch}
# 跨卡数续跑通过 state JSON 里的「global_samples」字段做换算：
#   global_samples = global_step × world_size × batch_size  （在保存时计算）
#   新卡数下 new_global_step = global_samples / (new_world_size × batch_size)
# 旧 checkpoint（无 state 文件，目录名就是 per-rank step）：
#   - 不指定 DIFFSYNTH_RESUME_OLD_WORLD_SIZE → 假设卡数没变，直接当 per-rank step 用
#   - 指定 DIFFSYNTH_RESUME_OLD_WORLD_SIZE=<原卡数> → 按 global_samples 换算
_CKPT_STATE_FILENAME = "diffsynth_training_state.json"
_CKPT_STATE_SCHEMA_VERSION = 3  # v3 = per-rank step + global_samples; v2/v1 = 历史格式
_CKPT_COMPLETE_FILENAME = "diffsynth_checkpoint_complete.json"


def _save_training_state(ckpt_dir, *, global_step, epoch_id, batch_idx,
                         world_size, batch_size, dataset_size):
    """将训练进度元数据写入 checkpoint 目录下的 JSON 文件（仅主进程调用）。

    global_step = per-rank step（跟 checkpoint 名一致）。
    global_samples 同时存一份，用于跨卡数恢复时正确换算。
    """
    import json
    state = {
        "schema_version": _CKPT_STATE_SCHEMA_VERSION,
        "global_step": int(global_step),
        "global_samples": int(global_step) * int(world_size) * int(batch_size),
        "epoch_id": int(epoch_id),
        "batch_idx_in_epoch": int(batch_idx),
        "world_size": int(world_size),
        "batch_size": int(batch_size),
        "dataset_size": int(dataset_size),
    }
    path = os.path.join(ckpt_dir, _CKPT_STATE_FILENAME)
    try:
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"⚠ 写入 {_CKPT_STATE_FILENAME} 失败: {e}", flush=True)


def _load_training_state(resume_dir):
    """从 checkpoint 目录读取训练进度元数据；不存在则返回 None（兼容旧 checkpoint）。"""
    import json
    path = os.path.join(resume_dir, _CKPT_STATE_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠ 读取 {_CKPT_STATE_FILENAME} 失败: {e}", flush=True)
        return None


def _checkpoint_is_complete(ckpt_dir):
    """判断 checkpoint 目录是否可用于恢复。

    新 checkpoint 以完成标记为准；旧 checkpoint 兼容依赖 DeepSpeed 核心文件存在。
    """
    if os.path.isfile(os.path.join(ckpt_dir, _CKPT_COMPLETE_FILENAME)):
        return True

    legacy_core_files = [
        os.path.join(ckpt_dir, "latest"),
        os.path.join(ckpt_dir, "scheduler.bin"),
        os.path.join(ckpt_dir, "random_states_0.pkl"),
        os.path.join(ckpt_dir, "pytorch_model", "mp_rank_00_model_states.pt"),
    ]
    if all(os.path.exists(path) for path in legacy_core_files):
        return True

    return False


def _prune_old_step_checkpoint_dirs(output_path, keep_limit):
    """Keep only the newest complete step-state directories.

    Model/lora safetensors are intentionally retained for evaluation. A state
    directory becomes eligible for pruning only when its matching model-step
    weight exists, so cleanup never runs halfway through a new save.
    """
    import re

    if keep_limit <= 0 or not os.path.isdir(output_path):
        return []

    complete_steps = []
    for name in os.listdir(output_path):
        match = re.fullmatch(r"checkpoint-([0-9]+)", name)
        if match is None:
            continue
        step = int(match.group(1))
        ckpt_dir = os.path.join(output_path, name)
        weight_path = os.path.join(output_path, f"model-step-{step}.safetensors")
        if _checkpoint_is_complete(ckpt_dir) and os.path.isfile(weight_path):
            complete_steps.append(step)

    complete_steps.sort(reverse=True)
    removed_steps = []
    for step in complete_steps[keep_limit:]:
        shutil.rmtree(os.path.join(output_path, f"checkpoint-{step}"))
        removed_steps.append(step)
    return removed_steps


def _mark_checkpoint_complete(ckpt_dir, *, global_step, epoch_id, batch_idx, world_size, batch_size):
    """写入成功标记，作为原子切换完成后的完成信号。"""
    import json
    path = os.path.join(ckpt_dir, _CKPT_COMPLETE_FILENAME)
    payload = {
        "schema_version": 1,
        "global_step": int(global_step),
        "epoch_id": int(epoch_id),
        "batch_idx_in_epoch": int(batch_idx),
        "world_size": int(world_size),
        "batch_size": int(batch_size),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _derive_resume_position(resume, ckpt_state, *, batch_size, new_world_size, dataset_size):
    """根据 checkpoint 元数据推导当前训练应从哪里继续。

    返回 (global_step, start_epoch, skip_batches, source_desc)。
    """
    samples_per_step = new_world_size * batch_size
    steps_per_epoch_now = dataset_size // samples_per_step if samples_per_step > 0 else 0

    if ckpt_state is not None and ckpt_state.get("schema_version", 1) >= 3 and "global_samples" in ckpt_state:
        global_samples = int(ckpt_state["global_samples"])
        global_step = global_samples // samples_per_step
        start_epoch = global_samples // dataset_size
        samples_in_cur = global_samples - start_epoch * dataset_size
        skip_batches = samples_in_cur // samples_per_step
        return global_step, start_epoch, skip_batches, "v3 state (global_samples)"

    if ckpt_state is not None and ckpt_state.get("schema_version", 1) == 2 and "global_samples" in ckpt_state:
        global_samples = int(ckpt_state["global_samples"])
        global_step = global_samples // samples_per_step
        start_epoch = global_samples // dataset_size
        samples_in_cur = global_samples - start_epoch * dataset_size
        skip_batches = samples_in_cur // samples_per_step
        return global_step, start_epoch, skip_batches, "v2 state (global_samples)"

    ckpt_name = os.path.basename(os.path.normpath(resume))
    raw_step = 0
    for sep in ['-', '_']:
        if sep in ckpt_name:
            for part in reversed(ckpt_name.split(sep)):
                if part.isdigit():
                    raw_step = int(part)
                    break
            if raw_step > 0:
                break
    if raw_step == 0:
        raise ValueError(f"无法从目录名 '{ckpt_name}' 解析 per-rank step 数")

    _ws_env = os.environ.get("DIFFSYNTH_RESUME_OLD_WORLD_SIZE", "").strip()
    if _ws_env.isdigit() and int(_ws_env) > 0:
        old_world_size = int(_ws_env)
        global_samples = raw_step * old_world_size * batch_size
        global_step = global_samples // samples_per_step
        start_epoch = global_samples // dataset_size
        samples_in_cur = global_samples - start_epoch * dataset_size
        skip_batches = samples_in_cur // samples_per_step
        return global_step, start_epoch, skip_batches, "legacy ckpt (env)"

    global_step = raw_step
    start_epoch = (global_step * samples_per_step) // dataset_size
    samples_in_cur = global_step * samples_per_step - start_epoch * dataset_size
    skip_batches = samples_in_cur // samples_per_step
    return global_step, start_epoch, skip_batches, "legacy ckpt"


def _find_model_step_checkpoint(resume, output_path, global_step):
    """查找与 resume 对应的纯模型权重 checkpoint。"""
    candidates = []
    if output_path:
        candidates.append(os.path.join(output_path, f"model-step-{global_step}.safetensors"))
    if resume:
        parent_dir = os.path.dirname(os.path.normpath(resume))
        candidates.append(os.path.join(parent_dir, f"model-step-{global_step}.safetensors"))

    seen = set()
    for path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path
    return None


def _load_weights_only_checkpoint(accelerator, model, weight_path):
    """仅把模型权重载入当前模型，跳过 optimizer/scheduler/RNG 状态。"""
    state_dict = load_state_dict(weight_path, torch_dtype=None, device="cpu")
    unwrap_model = accelerator.unwrap_model(model)
    load_res = unwrap_model.load_state_dict(state_dict, strict=False)
    missing_keys = len(getattr(load_res, "missing_keys", []))
    unexpected_keys = len(getattr(load_res, "unexpected_keys", []))
    accelerator.print(
        f"✓ 已仅加载模型权重: {weight_path} | "
        f"missing_keys={missing_keys} unexpected_keys={unexpected_keys}"
    )
    return unwrap_model


def _save_checkpoint_state_safely(
    accelerator,
    ckpt_dir,
    *,
    global_step,
    epoch_id,
    batch_idx,
    world_size,
    batch_size,
    dataset_size,
    strict_save_errors=False,
    save_state_enabled=True,
):
    """Save one distributed Accelerate checkpoint through a shared temp directory.

    Every rank must pass the same directory to ``accelerator.save_state`` so
    DeepSpeed can place all optimizer/RNG shards in one checkpoint.  The main
    rank publishes that directory only after every rank reports success.
    """
    if not save_state_enabled:
        accelerator.print("⚠ DIFFSYNTH_SAVE_ACCEL_STATE=0，跳过完整 Accelerate checkpoint。")
        return False

    # This path is intentionally identical on every rank.  A PID suffix would
    # split ZeRO optimizer shards across unrelated directories.
    tmp_ckpt_dir = f"{ckpt_dir}.tmp"

    def _finish_failure(stage, local_error):
        if local_error is not None:
            rank = getattr(accelerator, "process_index", 0)
            print(
                f"[rank{rank}] checkpoint {stage} failed: "
                f"{type(local_error).__name__}: {local_error}",
                flush=True,
            )
        if accelerator.is_main_process:
            shutil.rmtree(tmp_ckpt_dir, ignore_errors=True)
        accelerator.wait_for_everyone()
        accelerator.print(f"⚠ 保存完整 checkpoint 失败（{stage}）: {ckpt_dir}")
        if strict_save_errors:
            if local_error is not None:
                raise local_error
            raise RuntimeError(f"checkpoint {stage} failed on another rank: {ckpt_dir}")
        return False

    prepare_error = None
    if accelerator.is_main_process:
        try:
            shutil.rmtree(tmp_ckpt_dir, ignore_errors=True)
            os.makedirs(tmp_ckpt_dir, exist_ok=True)
        except Exception as e:
            prepare_error = e
    if _distributed_any_rank_failed(prepare_error is not None, accelerator.device):
        return _finish_failure("prepare", prepare_error)

    save_error = None
    try:
        accelerator.save_state(tmp_ckpt_dir)
    except Exception as e:
        save_error = e
    if _distributed_any_rank_failed(save_error is not None, accelerator.device):
        return _finish_failure("write", save_error)

    publish_error = None
    if accelerator.is_main_process:
        try:
            _save_training_state(
                tmp_ckpt_dir,
                global_step=global_step,
                epoch_id=epoch_id,
                batch_idx=batch_idx,
                world_size=world_size,
                batch_size=batch_size,
                dataset_size=dataset_size,
            )
            state_path = os.path.join(tmp_ckpt_dir, _CKPT_STATE_FILENAME)
            if not os.path.isfile(state_path):
                raise RuntimeError(f"training state metadata was not written: {state_path}")
            _mark_checkpoint_complete(
                tmp_ckpt_dir,
                global_step=global_step,
                epoch_id=epoch_id,
                batch_idx=batch_idx,
                world_size=world_size,
                batch_size=batch_size,
            )
            if os.path.isdir(ckpt_dir):
                shutil.rmtree(ckpt_dir)
            os.rename(tmp_ckpt_dir, ckpt_dir)
        except Exception as e:
            publish_error = e
    if _distributed_any_rank_failed(publish_error is not None, accelerator.device):
        return _finish_failure("publish", publish_error)

    # Do not let non-main ranks start the next training step until the final
    # directory rename is visible on the shared filesystem.
    accelerator.wait_for_everyone()
    return True


def _save_single_rank_metadata(ckpt_dir, *, global_step, epoch_id, batch_idx, world_size, batch_size, dataset_size):
    _save_training_state(
        ckpt_dir,
        global_step=global_step,
        epoch_id=epoch_id,
        batch_idx=batch_idx,
        world_size=world_size,
        batch_size=batch_size,
        dataset_size=dataset_size,
    )
    # This directory is metadata-only.  Do not write the complete marker:
    # accelerator.load_state() cannot restore optimizer/RNG state from it.


def _distributed_any_rank_failed(local_failed: bool, device):
    """Return True if any rank reports failure for the current step."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return local_failed

    flag = torch.tensor(1 if local_failed else 0, device=device, dtype=torch.int32)
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
    return bool(flag.item())


def _try_fast_resume_skip(accelerator, dataset, dataloader, *, skip_batches,
                          num_workers, batch_size):
    """用 Subset 裁掉前 N 个样本重建 dataloader，避免主循环里 next() 空转拉 I/O。

    返回 (new_dataloader, new_skip_batches)。
    若不满足快速续跑条件（环境变量关闭、dataset 不支持 __len__/__getitem__、
    或裁切后样本数不足），返回原 dataloader 和原 skip_batches，由调用方继续走 continue 跳过。

    注意：仅适用于 accelerate 默认分片策略（stride shard，shuffle=False 最常见）。
    若启用 split_batches/dispatch_batches 等特殊配置，请设
    DIFFSYNTH_FAST_RESUME_SKIP=0 回退到主循环 continue 跳过。
    """
    fast_resume_skip = os.environ.get("DIFFSYNTH_FAST_RESUME_SKIP", "1") == "1"
    if not (fast_resume_skip and skip_batches > 0
            and hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")):
        return dataloader, skip_batches

    from torch.utils.data import Subset
    num_processes = max(1, int(getattr(accelerator, "num_processes", 1) or 1))
    # global_skip = per-rank batches × world_size × batch_size = 全局已消费样本数
    global_skip = skip_batches * num_processes * batch_size
    if global_skip >= len(dataset):
        accelerator.print(
            f"⚠ 快速续跑条件不满足：global_skip={global_skip} >= len(dataset)={len(dataset)}，"
            f"回退到主循环 continue 跳过。"
        )
        return dataloader, skip_batches

    sub_indices = list(range(global_skip, len(dataset)))
    subset = Subset(dataset, sub_indices)
    rebuilt_loader = _make_training_dataloader(
        subset,
        num_workers=num_workers,
        batch_size=batch_size,
        drop_last=True,
    )
    new_dataloader = accelerator.prepare_data_loader(rebuilt_loader)
    accelerator.print(
        f"⚡ 快速续跑：per-rank skip_batches={skip_batches} × world_size={num_processes} "
        f"× batch_size={batch_size} = global_skip={global_skip}；"
        f"已从全局 dataset（{len(dataset)}）裁到 {len(sub_indices)} 个样本，"
        f"per-rank 将走 ~{len(sub_indices)//(num_processes * batch_size)} 步。"
    )
    return new_dataloader, 0


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    batch_size: int = 1,
    resume: str = '',
    args=None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        batch_size = args.batch_size
        resume = getattr(args, 'resume', resume) or resume
        resume_step = getattr(args, 'resume_step', 0) or 0
    else:
        resume_step = 0
    is_lora_training = bool(args is not None and getattr(args, "lora_base_model", None))

    # === 优化器与调度器 ===
    adamw_foreach_env = os.environ.get("DIFFSYNTH_ADAMW_FOREACH")
    optimizer_kwargs = {}
    if adamw_foreach_env is not None:
        adamw_foreach_value = adamw_foreach_env.strip().lower()
        if adamw_foreach_value not in {"0", "1", "false", "true"}:
            raise ValueError(
                "DIFFSYNTH_ADAMW_FOREACH must be one of "
                f"0, 1, false, true; got {adamw_foreach_env!r}"
            )
        optimizer_kwargs["foreach"] = adamw_foreach_value in {"1", "true"}
        accelerator.print(
            f"AdamW foreach: {optimizer_kwargs['foreach']} "
            "(configured by DIFFSYNTH_ADAMW_FOREACH)"
        )
    empty_cache_optimizer_interval_raw = os.environ.get(
        "DIFFSYNTH_EMPTY_CACHE_OPTIMIZER_INTERVAL", "0"
    )
    try:
        empty_cache_optimizer_interval = int(
            empty_cache_optimizer_interval_raw
        )
    except ValueError as exc:
        raise ValueError(
            "DIFFSYNTH_EMPTY_CACHE_OPTIMIZER_INTERVAL must be a "
            f"non-negative integer, got {empty_cache_optimizer_interval_raw!r}"
        ) from exc
    if empty_cache_optimizer_interval < 0:
        raise ValueError(
            "DIFFSYNTH_EMPTY_CACHE_OPTIMIZER_INTERVAL must be non-negative, "
            f"got {empty_cache_optimizer_interval}"
        )
    if empty_cache_optimizer_interval:
        accelerator.print(
            "CUDA allocator cache cleanup: every "
            f"{empty_cache_optimizer_interval} optimizer update(s)"
        )
    max_optimizer_steps_raw = os.environ.get(
        "DIFFSYNTH_MAX_OPTIMIZER_STEPS", "0"
    )
    try:
        max_optimizer_steps = int(max_optimizer_steps_raw)
    except ValueError as exc:
        raise ValueError(
            "DIFFSYNTH_MAX_OPTIMIZER_STEPS must be a non-negative integer, "
            f"got {max_optimizer_steps_raw!r}"
        ) from exc
    if max_optimizer_steps < 0:
        raise ValueError(
            "DIFFSYNTH_MAX_OPTIMIZER_STEPS must be non-negative, "
            f"got {max_optimizer_steps}"
        )
    if max_optimizer_steps:
        accelerator.print(
            f"Training limit: {max_optimizer_steps} optimizer updates"
        )
    optimizer_group_max_elements = _optimizer_group_max_elements()
    optimizer_param_groups = _build_optimizer_param_groups(
        model,
        weight_decay,
        max_elements=optimizer_group_max_elements,
    )
    optimizer_group_sizes = [
        sum(parameter.numel() for parameter in group["params"])
        for group in optimizer_param_groups
    ]
    expected_trainable_dtypes = _capture_trainable_dtypes(model)
    accelerator.print(
        "Trainable parameter dtypes before prepare: "
        f"{_summarize_trainable_dtypes(model)} | "
        f"optimizer groups={len(optimizer_param_groups)} "
        f"largest_group_elements={max(optimizer_group_sizes):,} "
        f"group_cap={optimizer_group_max_elements or 'disabled'}"
    )
    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
        lr=learning_rate,
        weight_decay=weight_decay,
        **optimizer_kwargs,
    )
    warmup_updates_env = os.environ.get("DIFFSYNTH_WARMUP_UPDATES")
    if warmup_updates_env is None:
        scheduler_warmup_steps = 1000
    else:
        warmup_updates = int(warmup_updates_env)
        if warmup_updates < 0:
            raise ValueError(
                "DIFFSYNTH_WARMUP_UPDATES must be non-negative, "
                f"got {warmup_updates}"
            )
        # Accelerate advances a prepared scheduler once per process for every
        # optimizer update when batches are sharded across processes.
        scheduler_warmup_steps = warmup_updates * accelerator.num_processes
        accelerator.print(
            f"LR warmup: {warmup_updates} optimizer updates "
            f"({scheduler_warmup_steps} scheduler steps across "
            f"{accelerator.num_processes} processes)"
        )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_lr_fn(
            step, warmup_steps=scheduler_warmup_steps
        )
    )

    # === 数据加载器 ===
    dataloader = _make_training_dataloader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        drop_last=True,
    )

    # === Accelerate 准备（必须在恢复状态前）===
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    base_model = accelerator.unwrap_model(model)
    accelerator.print(
        "Trainable parameter dtypes after prepare: "
        f"{_summarize_trainable_dtypes(base_model)}"
    )

    # === DeepSpeed FP32 修复（保持原逻辑）===
    if os.environ.get("DIFFSYNTH_FORCE_FP32_TIME_MLP", "1") == "1":
        pipe = getattr(base_model, "pipe", None)
        if pipe is not None:
            for dit_attr in ("dit", "dit2"):
                dit_module = getattr(pipe, dit_attr, None)
                if dit_module is not None and hasattr(dit_module, "time_embedding") and hasattr(dit_module, "time_projection"):
                    dit_module.time_embedding = dit_module.time_embedding.float()
                    dit_module.time_projection = dit_module.time_projection.float()
                    accelerator.print(f"Re-cast {dit_attr} timestep MLP to fp32 after prepare.")

    # DeepSpeed mixed precision may cast the timestep MLP to bf16 during
    # prepare().  Validate only after the intentional fp32 restoration above;
    # validating earlier turns that expected, repaired cast into a false error.
    if os.environ.get("DIFFSYNTH_ENFORCE_TRAINABLE_DTYPES", "0") == "1":
        _validate_trainable_dtypes(base_model, expected_trainable_dtypes)
        accelerator.print(
            "Trainable parameter dtypes after post-prepare restoration: "
            f"{_summarize_trainable_dtypes(base_model)}"
        )

    # === 恢复训练状态 ===
    # global_step 语义 = per-rank step 数（跟 checkpoint 名一致）
    global_step = 0       # per-rank step
    start_epoch = 0
    skip_batches = 0      # per-rank batch 数（仅用于跳过 dataloader）
    train_error_count = 0
    new_world_size = max(1, int(getattr(accelerator, "num_processes", 1) or 1))
    _samples_per_step = new_world_size * batch_size
    _dataset_size = len(dataset) if hasattr(dataset, "__len__") else len(dataloader) * _samples_per_step
    _steps_per_epoch_now = _dataset_size // _samples_per_step  # 当前卡数下 per-rank steps/epoch
    strict_train_errors = os.environ.get("DIFFSYNTH_STRICT_TRAIN_ERRORS", "0") == "1"
    # 多卡 fail-fast：默认关闭。开启后任意 rank 出错都会立即终止训练。
    # 注意：raise 必须在 accelerator.accumulate 上下文外执行，否则 DeepSpeed __exit__
    # 会触发额外 collective op 造成 NCCL desync。代码已按此规则处理。
    fail_fast_distributed = os.environ.get("DIFFSYNTH_FAIL_FAST_DISTRIBUTED", "0") == "1"
    skip_accelerate_state_save = os.environ.get("DIFFSYNTH_SKIP_ACCELERATE_SAVE_STATE", "0") == "1"
    strict_save_errors = os.environ.get("DIFFSYNTH_STRICT_SAVE_ERRORS", "0") == "1"
    save_state_enabled = os.environ.get("DIFFSYNTH_SAVE_ACCEL_STATE", "1") == "1"
    weak_save_barrier = os.environ.get("DIFFSYNTH_WEAK_SAVE_BARRIER", "0") == "1"
    save_degraded_mode = False

    checkpoint_total_limit_raw = os.environ.get(
        "DIFFSYNTH_CHECKPOINT_TOTAL_LIMIT", "0"
    )
    try:
        checkpoint_total_limit = int(checkpoint_total_limit_raw)
    except ValueError as exc:
        raise ValueError(
            "DIFFSYNTH_CHECKPOINT_TOTAL_LIMIT must be a non-negative integer, "
            f"got {checkpoint_total_limit_raw!r}"
        ) from exc
    if checkpoint_total_limit < 0:
        raise ValueError(
            "DIFFSYNTH_CHECKPOINT_TOTAL_LIMIT must be non-negative, "
            f"got {checkpoint_total_limit}"
        )
    if checkpoint_total_limit:
        accelerator.print(
            "Checkpoint retention: keep the newest "
            f"{checkpoint_total_limit} complete state director"
            f"{'y' if checkpoint_total_limit == 1 else 'ies'}; "
            "model-step safetensors are retained"
        )

    def _resume_from_global_samples(global_samples, *, source_desc):
        """根据「全局已消费样本数」计算当前卡数下的 (global_step, start_epoch, skip_batches)。"""
        new_global_step = global_samples // _samples_per_step
        ep = global_samples // _dataset_size
        samples_in_cur = global_samples - ep * _dataset_size
        skip_b = samples_in_cur // _samples_per_step
        accelerator.print(
            f"  {source_desc}: global_samples={global_samples} → "
            f"per-rank global_step={new_global_step} (= checkpoint-{new_global_step} 命名) | "
            f"start_epoch={ep} | epoch 内 skip={skip_b}/{_steps_per_epoch_now}"
        )
        return new_global_step, ep, skip_b

    if resume and os.path.isdir(resume):
        accelerator.print(f"Resuming training from checkpoint: {resume}")
        try:
            if not _checkpoint_is_complete(resume):
                raise ValueError(
                    f"checkpoint 不完整，缺少 {_CKPT_COMPLETE_FILENAME}: {resume}"
                )
            ckpt_state = _load_training_state(resume)
            global_step, start_epoch, skip_batches, source_desc = _derive_resume_position(
                resume,
                ckpt_state,
                batch_size=batch_size,
                new_world_size=new_world_size,
                dataset_size=_dataset_size,
            )

            weights_only_mode = os.environ.get("DIFFSYNTH_RESUME_WEIGHTS_ONLY", "auto").strip().lower()
            force_weights_only = weights_only_mode in {"1", "true", "yes", "on"}
            auto_weights_only = weights_only_mode in {"auto", ""}

            try:
                # Accelerate 会恢复: 模型权重、优化器状态、调度器状态、RNG 状态
                accelerator.load_state(resume)
                resume_mode = "full-state"
                accelerator.print("✓ 已恢复完整训练状态（模型 + optimizer + scheduler + RNG）")
            except Exception as e:
                err_msg = str(e)
                is_zero_world_size_mismatch = "world size" in err_msg and "ZeRORuntimeException" in err_msg
                if not (force_weights_only or (auto_weights_only and is_zero_world_size_mismatch)):
                    raise

                checkpoint_weight_step = (
                    int(ckpt_state["global_step"])
                    if ckpt_state is not None and "global_step" in ckpt_state
                    else global_step
                )
                weight_path = _find_model_step_checkpoint(
                    resume,
                    model_logger.output_path,
                    checkpoint_weight_step,
                )
                if weight_path is None:
                    raise FileNotFoundError(
                        f"未找到与 checkpoint-{checkpoint_weight_step} 对应的模型权重文件"
                        f"（model-step-{checkpoint_weight_step}.safetensors）"
                    ) from e

                _load_weights_only_checkpoint(accelerator, model, weight_path)
                resume_mode = "weights-only"
                accelerator.print(
                    "⚠ 完整恢复失败，已改为只加载模型权重继续训练。"
                )
                accelerator.print(
                    f"  原因: {type(e).__name__}: {e}"
                )

            model_logger.num_steps = global_step
            accelerator.print(
                f"✓ 恢复位置:\n"
                f"  - mode: {resume_mode}\n"
                f"  - per-rank global_step: {global_step} (checkpoint-{global_step})\n"
                f"  - start_epoch: {start_epoch}\n"
                f"  - skip_batches in current epoch: {skip_batches}/{_steps_per_epoch_now}\n"
                f"  - world_size: {new_world_size} | batch_size: {batch_size} | "
                f"steps_per_epoch: {_steps_per_epoch_now}\n"
                f"  - 学习率: {scheduler.get_last_lr()[0]:.2e}"
            )

            if skip_batches > 0:
                accelerator.print(f"跳过当前 epoch 的前 {skip_batches} 个 per-rank batches...")
                dataloader, skip_batches = _try_fast_resume_skip(
                    accelerator, dataset, dataloader,
                    skip_batches=skip_batches,
                    num_workers=num_workers,
                    batch_size=batch_size,
                )

        except Exception as e:
            accelerator.print(f"✗ 恢复训练失败: {e}")
            if os.environ.get("DIFFSYNTH_ALLOW_RESUME_FALLBACK", "0") != "1":
                raise
            accelerator.print("DIFFSYNTH_ALLOW_RESUME_FALLBACK=1，允许恢复失败后从头开始训练")
            resume = ''
            global_step = 0
            start_epoch = 0
            skip_batches = 0

    # --resume_step: 仅有模型权重、无优化器状态时的轻量续跑。
    # 语义：resume_step = 「原卡数下的 per-rank step」。
    # 跨卡数续跑：export DIFFSYNTH_RESUME_OLD_WORLD_SIZE=<原卡数> 进行换算；
    # 不指定时默认假设卡数没变，直接用 resume_step 作为当前 per-rank step。
    if resume_step > 0 and global_step == 0:
        _ws_env = os.environ.get("DIFFSYNTH_RESUME_OLD_WORLD_SIZE", "").strip()
        if _ws_env.isdigit() and int(_ws_env) > 0:
            old_world_size = int(_ws_env)
            global_samples = resume_step * old_world_size * batch_size
            accelerator.print(
                f"✓ resume_step={resume_step} per-rank × OLD_WORLD_SIZE={old_world_size} × bs={batch_size} "
                f"= {global_samples} 全局样本"
            )
            global_step, start_epoch, skip_batches = _resume_from_global_samples(
                global_samples, source_desc="resume_step (env)"
            )
        else:
            accelerator.print(
                f"✓ resume_step={resume_step}（未指定 DIFFSYNTH_RESUME_OLD_WORLD_SIZE，"
                f"假设卡数未变，直接当当前 per-rank step 用）"
            )
            global_step = resume_step
            start_epoch = (global_step * _samples_per_step) // _dataset_size
            samples_in_cur = global_step * _samples_per_step - start_epoch * _dataset_size
            skip_batches = samples_in_cur // _samples_per_step
            accelerator.print(
                f"  → start_epoch={start_epoch}, skip={skip_batches}/{_steps_per_epoch_now}"
            )

        model_logger.num_steps = global_step

        dataloader, skip_batches = _try_fast_resume_skip(
            accelerator, dataset, dataloader,
            skip_batches=skip_batches,
            num_workers=num_workers,
            batch_size=batch_size,
        )

    if global_step == 0 and resume_step == 0:
        accelerator.print("✓ 从头开始训练")

    if "foreach" in optimizer_kwargs:
        changed_groups = _set_optimizer_foreach(
            optimizer, optimizer_kwargs["foreach"]
        )
        accelerator.print(
            f"AdamW foreach reapplied after checkpoint loading: "
            f"{optimizer_kwargs['foreach']} ({changed_groups} param groups)"
        )

    # 平台/TLS 日志常按「换行」采集，tqdm 同一行 \r 刷新时 postfix(loss/lr) 易被截断；
    # 用独立换行日志保证可检索（间隔可用 DIFFSYNTH_TRAIN_LOG_INTERVAL 调整，默认 10）。
    train_log_interval = max(1, int(os.environ.get("DIFFSYNTH_TRAIN_LOG_INTERVAL", "1")))
    accelerator.print(
        f"日志间隔: DIFFSYNTH_TRAIN_LOG_INTERVAL={train_log_interval} (每 {train_log_interval} 步打印一次)"
    )
    step_trace_enabled = os.environ.get("DIFFSYNTH_STEP_TRACE", "1") == "1"
    step_trace_max_steps = max(1, int(os.environ.get("DIFFSYNTH_STEP_TRACE_MAX_STEPS", "8")))

    # 平台日志采集/字体常缺 Unicode 块字符，tqdm 会显示成 ??；非 TTY 时默认用 ASCII（#）
    _ta = os.environ.get("DIFFSYNTH_TQDM_ASCII", "").strip().lower()
    if _ta == "1":
        tqdm_ascii = True
    elif _ta == "0":
        tqdm_ascii = False
    else:
        tqdm_ascii = not sys.stdout.isatty()

    # 后台心跳线程：每 60 秒打一行，告知进程仍在运行（COS 读大视频时主线程会长时间无输出）
    _heartbeat_interval = int(os.environ.get("DIFFSYNTH_HEARTBEAT_SEC", "60"))
    _heartbeat_stop = threading.Event()

    def _heartbeat_fn():
        while not _heartbeat_stop.wait(_heartbeat_interval):
            if accelerator.is_local_main_process:
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"[heartbeat] still alive — global_step={global_step}",
                    flush=True,
                )

    _heartbeat_thread = threading.Thread(target=_heartbeat_fn, daemon=True)
    _heartbeat_thread.start()

    def _trace_step(stage, step_id, t0, extra=""):
        if not step_trace_enabled or not accelerator.is_local_main_process:
            return
        if step_id > step_trace_max_steps:
            return
        elapsed = time.perf_counter() - t0
        suffix = f" | {extra}" if extra else ""
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f">>> step-trace step={step_id} stage={stage} elapsed={elapsed:.2f}s{suffix}",
            flush=True,
        )

    _epoch_total = max(num_epochs, 1)
    # 进度条单位 = per-rank step；分母 = steps_per_epoch（当前卡数下）。
    # 4 卡 968845/4 ≈ 242211 步/epoch；16 卡 ≈ 60552 步/epoch。
    _bar_total = _steps_per_epoch_now
    _last_epoch_id = start_epoch
    _last_batch_idx = skip_batches - 1  # 续跑时上次完成的 per-rank batch；首次跑为 -1
    gradient_accumulation_steps = max(
        1, int(getattr(accelerator, "gradient_accumulation_steps", 1) or 1)
    )
    # global_step is the persisted per-rank micro-step count.  The staged
    # launchers save only at accumulation boundaries, so this reconstructs the
    # absolute optimizer-update position on resume as well as on a fresh run.
    optimizer_update_count = global_step // gradient_accumulation_steps
    stop_training = bool(
        max_optimizer_steps
        and optimizer_update_count >= max_optimizer_steps
    )
    if stop_training:
        accelerator.print(
            "Optimizer-update limit was already reached by the resumed "
            f"checkpoint: completed={optimizer_update_count} "
            f"limit={max_optimizer_steps}"
        )
    model.train()
    for epoch_id in range(start_epoch, num_epochs):
        if stop_training:
            break
        # 进度条 initial = 当前 epoch 内已完成的 per-rank steps。
        # 直接由 global_step 反推，避免依赖 skip_batches（fast_resume_skip 触发后会清零）。
        if epoch_id == start_epoch:
            _steps_in_cur_epoch_init = global_step - start_epoch * _steps_per_epoch_now
        else:
            _steps_in_cur_epoch_init = 0
        progress_bar = tqdm(
            total=_bar_total,
            initial=_steps_in_cur_epoch_init,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch_id + 1}/{_epoch_total}",
            ascii=tqdm_ascii,
            unit="step",
        )

        if epoch_id == start_epoch:
            current_skip = skip_batches
            skip_batches = 0
        else:
            current_skip = 0

        for batch_idx, data in enumerate(dataloader):
            if current_skip > 0:
                current_skip -= 1
                continue

            step_id = global_step + 1
            step_t0 = time.perf_counter()

            _trace_step("accumulate_enter", step_id, step_t0, f"batch_idx={batch_idx}")

            _step_error = None  # 捕获本步异常，延迟到 accumulate 上下文外处理
            with accelerator.accumulate(model):
                try:
                    if dataset.load_from_cache:
                        loss = model({}, inputs=data)
                    else:
                        loss = model(data)
                    _trace_step("forward_done", step_id, step_t0, f"sync_gradients={accelerator.sync_gradients}")

                    accelerator.backward(loss)
                    _trace_step("backward_done", step_id, step_t0, f"sync_gradients={accelerator.sync_gradients}")

                except Exception as e:
                    train_error_count += 1
                    rank_id = int(getattr(accelerator, "process_index", 0) or 0)
                    err_msg = (
                        f"[Rank {rank_id} | Step {global_step}] "
                        f"训练错误#{train_error_count}: {type(e).__name__}: {e}"
                    )
                    print(err_msg, flush=True)
                    if train_error_count == 1:
                        traceback.print_exc()
                    if accelerator.sync_gradients:
                        optimizer.zero_grad()
                    # OOM 类错误时主动归还显存，避免重试时碎片越积越多。
                    if isinstance(e, (torch.cuda.OutOfMemoryError, RuntimeError)) and (
                        "out of memory" in str(e).lower() or "CUDA" in str(e)
                    ):
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        except Exception:
                            pass
                    _step_error = RuntimeError(err_msg)

            _trace_step("accumulate_exit", step_id, step_t0, f"sync_gradients={accelerator.sync_gradients}")

            # Once one rank has failed inside forward/backward, collective
            # sequences may already differ. In strict mode, exit that rank
            # immediately and let the elastic launcher terminate its peers.
            if _step_error is not None and (
                strict_train_errors or fail_fast_distributed
            ):
                try:
                    model.zero_grad(set_to_none=True)
                except TypeError:
                    model.zero_grad()
                optimizer.zero_grad()
                raise _step_error

            # ⚠ 必须在 accumulate 上下文外才能 raise：在上下文内 raise 会触发
            # DeepSpeed accumulate.__exit__ 的 collective 收尾逻辑，导致 NCCL desync。
            step_failed = _distributed_any_rank_failed(_step_error is not None, accelerator.device)
            _trace_step("failure_check_done", step_id, step_t0, f"step_failed={step_failed}")
            if step_failed:
                # DeepSpeed's optimizer wrapper implements zero_grad as a no-op,
                # so clear the engine/model gradients as well on every rank.
                try:
                    model.zero_grad(set_to_none=True)
                except TypeError:
                    model.zero_grad()
                optimizer.zero_grad()
                if strict_train_errors or fail_fast_distributed:
                    if _step_error is not None:
                        raise _step_error
                    raise RuntimeError(
                        f"[Rank {getattr(accelerator, 'process_index', 0)}] "
                        "Distributed step failure detected on another rank."
                    )
                continue

            if accelerator.sync_gradients:
                _trace_step("optimizer_step_start", step_id, step_t0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_update_count += 1
                if (
                    empty_cache_optimizer_interval
                    and optimizer_update_count
                    % empty_cache_optimizer_interval
                    == 0
                    and torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()
                _trace_step("optimizer_step_done", step_id, step_t0)

            global_step += 1  # per-rank step
            _trace_step("step_complete", step_id, step_t0, f"global_step={global_step}")
            model_logger.num_steps = global_step
            _last_epoch_id = epoch_id
            _last_batch_idx = batch_idx

            # === 每 save_steps 个 per-rank step 保存一次 ===
            if save_steps and global_step % save_steps == 0:
                accelerator.print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"开始保存 checkpoint（per-rank global_step={global_step}），全 rank 同步等待写盘，"
                    f"GPU 利用率会暂时下降属正常。"
                )
                ckpt_dir = os.path.join(model_logger.output_path, f"checkpoint-{global_step}")
                step_state_saved_ok = False
                if skip_accelerate_state_save:
                    accelerator.print(f"\n⚠ 跳过 accelerate.save_state（DIFFSYNTH_SKIP_ACCELERATE_SAVE_STATE=1）: {ckpt_dir}")
                else:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if not save_degraded_mode:
                        saved_ok = _save_checkpoint_state_safely(
                            accelerator,
                            ckpt_dir,
                            global_step=global_step,
                            epoch_id=epoch_id,
                            batch_idx=batch_idx,
                            world_size=new_world_size,
                            batch_size=batch_size,
                            dataset_size=_dataset_size,
                            strict_save_errors=strict_save_errors,
                            save_state_enabled=save_state_enabled,
                        )
                        if saved_ok:
                            step_state_saved_ok = True
                            accelerator.print(f"\n✓ 保存完整训练状态到: {ckpt_dir}")
                        else:
                            accelerator.print(
                                f"\n⚠ 完整训练状态保存失败，切换到降级保存模式: {ckpt_dir}"
                            )
                            save_degraded_mode = True

                    if save_degraded_mode:
                        if accelerator.is_main_process:
                            os.makedirs(ckpt_dir, exist_ok=True)
                            _save_single_rank_metadata(
                                ckpt_dir,
                                global_step=global_step,
                                epoch_id=epoch_id,
                                batch_idx=batch_idx,
                                world_size=new_world_size,
                                batch_size=batch_size,
                                dataset_size=_dataset_size,
                            )
                        accelerator.print(
                            f"⚠ 当前处于降级保存模式，仅主 rank 写元数据与权重: {ckpt_dir}"
                        )

                # 无论是否跳过 accelerate 状态，都写入 v3 state 元数据（供跨卡数续跑使用）
                if accelerator.is_main_process and skip_accelerate_state_save:
                    _save_training_state(
                        ckpt_dir,
                        global_step=global_step,
                        epoch_id=epoch_id,
                        batch_idx=batch_idx,
                        world_size=new_world_size,
                        batch_size=batch_size,
                        dataset_size=_dataset_size,
                    )

                step_model_saved_ok = False
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Step checkpoint uses trainable-only weights to reduce save-time peak memory.
                    model_logger.save_model(
                        accelerator,
                        model,
                        f"model-step-{global_step}.safetensors",
                        save_full_model=False,
                        synchronize=not (save_degraded_mode or weak_save_barrier),
                    )
                    accelerator.print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"model-step-{global_step}.safetensors save done."
                    )
                    step_model_saved_ok = True
                except Exception as e:
                    accelerator.print(
                        f"⚠ Step checkpoint save failed at step {global_step}: {type(e).__name__}: {e}"
                    )
                    if strict_save_errors:
                        raise
                if is_lora_training:
                    model_logger.save_model(
                        accelerator,
                        model,
                        f"lora-step-{global_step}.safetensors",
                        save_full_model=False,
                        synchronize=not (save_degraded_mode or weak_save_barrier),
                        trainable_filter="lora",
                    )

                if (
                    checkpoint_total_limit
                    and step_state_saved_ok
                    and step_model_saved_ok
                ):
                    # The new state+weight pair is complete. Delete only old
                    # checkpoint-N state directories; retain every safetensors
                    # file for evaluation and loss/quality curves.
                    accelerator.wait_for_everyone()
                    prune_error = None
                    removed_steps = []
                    if accelerator.is_main_process:
                        try:
                            removed_steps = _prune_old_step_checkpoint_dirs(
                                model_logger.output_path,
                                checkpoint_total_limit,
                            )
                        except Exception as e:
                            prune_error = e
                    if _distributed_any_rank_failed(
                        prune_error is not None, accelerator.device
                    ):
                        if prune_error is not None:
                            raise prune_error
                        raise RuntimeError(
                            "checkpoint state-directory pruning failed on "
                            "the main process"
                        )
                    if removed_steps:
                        accelerator.print(
                            "✓ Removed old checkpoint state director"
                            f"{'y' if len(removed_steps) == 1 else 'ies'}: "
                            + ", ".join(
                                f"checkpoint-{step}" for step in removed_steps
                            )
                            + "; safetensors retained"
                        )
                    accelerator.wait_for_everyone()

            # === 统一的日志输出策略 ===
            # 当前 epoch 内已完成的 per-rank steps（用于进度条显示和日志）
            steps_in_epoch = global_step - epoch_id * _steps_per_epoch_now
            if accelerator.is_local_main_process:
                current_loss = loss.item() if 'loss' in locals() else 0.0
                current_lr = scheduler.get_last_lr()[0]
                loss_metrics = getattr(accelerator.unwrap_model(model), "latest_loss_metrics", {})
                task_loss = loss_metrics.get("task_loss", None)
                kd_loss = loss_metrics.get("kd_loss", None)
                raw_mse = loss_metrics.get("raw_mse", None)
                training_weight = loss_metrics.get("training_weight", None)
                timestep = loss_metrics.get("timestep", None)
                trajectory_kd_loss = loss_metrics.get("trajectory_kd_loss", None)
                base_kd_loss = loss_metrics.get("base_kd_loss", None)
                detail_edge_loss = loss_metrics.get("detail_edge_loss", None)
                detail_sample = loss_metrics.get("detail_sample", None)
                cfg_scale = loss_metrics.get("cfg_scale", None)
                rollout_steps = loss_metrics.get("rollout_steps", None)
                rollout_stride = loss_metrics.get("rollout_stride", None)
                high_noise = loss_metrics.get("high_noise", None)
                task_loss_value = task_loss.item() if hasattr(task_loss, "item") else None
                kd_loss_value = kd_loss.item() if hasattr(kd_loss, "item") else None
                raw_mse_value = raw_mse.item() if hasattr(raw_mse, "item") else None
                training_weight_value = (
                    training_weight.item()
                    if hasattr(training_weight, "item")
                    else None
                )
                timestep_value = timestep.item() if hasattr(timestep, "item") else None
                trajectory_kd_loss_value = (
                    trajectory_kd_loss.item()
                    if hasattr(trajectory_kd_loss, "item")
                    else None
                )
                base_kd_loss_value = (
                    base_kd_loss.item() if hasattr(base_kd_loss, "item") else None
                )
                detail_edge_loss_value = (
                    detail_edge_loss.item()
                    if hasattr(detail_edge_loss, "item")
                    else None
                )
                detail_sample_value = (
                    detail_sample.item() if hasattr(detail_sample, "item") else None
                )
                cfg_scale_value = cfg_scale.item() if hasattr(cfg_scale, "item") else None
                rollout_steps_value = (
                    rollout_steps.item()
                    if hasattr(rollout_steps, "item")
                    else None
                )
                rollout_stride_value = (
                    rollout_stride.item()
                    if hasattr(rollout_stride, "item")
                    else None
                )
                high_noise_value = high_noise.item() if hasattr(high_noise, "item") else None

                progress_bar.update(1)
                postfix = {
                    "loss": f"{current_loss:.4f}",
                    "lr": f"{current_lr:.2e}"
                }
                if task_loss_value is not None:
                    postfix["task_loss"] = f"{task_loss_value:.4f}"
                if kd_loss_value is not None:
                    postfix["kd_loss"] = f"{kd_loss_value:.4f}"
                if raw_mse_value is not None:
                    postfix["raw_mse"] = f"{raw_mse_value:.4f}"
                if trajectory_kd_loss_value is not None:
                    postfix["traj_kd"] = f"{trajectory_kd_loss_value:.3e}"
                if base_kd_loss_value is not None:
                    postfix["base_kd"] = f"{base_kd_loss_value:.3e}"
                if detail_edge_loss_value is not None:
                    postfix["edge"] = f"{detail_edge_loss_value:.3e}"
                progress_bar.set_postfix(postfix)

                # 独立换行日志（平台 TLS 可检索；间隔见 DIFFSYNTH_TRAIN_LOG_INTERVAL）
                if global_step % train_log_interval == 0:
                    loss_parts = [f"Loss: {current_loss:.4f}"]
                    if task_loss_value is not None:
                        loss_parts.append(f"Task Loss: {task_loss_value:.4f}")
                    if kd_loss_value is not None:
                        loss_parts.append(f"KD Loss: {kd_loss_value:.4f}")
                    if raw_mse_value is not None:
                        loss_parts.append(f"Raw MSE: {raw_mse_value:.4f}")
                    if trajectory_kd_loss_value is not None:
                        loss_parts.append(f"Traj KD: {trajectory_kd_loss_value:.3e}")
                    if base_kd_loss_value is not None:
                        loss_parts.append(f"Base KD: {base_kd_loss_value:.3e}")
                    if detail_edge_loss_value is not None:
                        loss_parts.append(f"Edge: {detail_edge_loss_value:.3e}")
                    if detail_sample_value is not None:
                        loss_parts.append(f"DetailSample: {detail_sample_value:.0f}")
                    if training_weight_value is not None:
                        loss_parts.append(f"Weight: {training_weight_value:.4f}")
                    if timestep_value is not None:
                        loss_parts.append(f"Timestep: {timestep_value:.2f}")
                    if cfg_scale_value is not None:
                        loss_parts.append(f"CFG: {cfg_scale_value:.2f}")
                    if rollout_steps_value is not None:
                        loss_parts.append(f"Rollout: {rollout_steps_value:.0f}")
                    if rollout_stride_value is not None:
                        loss_parts.append(f"Stride: {rollout_stride_value:.0f}")
                    if high_noise_value is not None:
                        loss_parts.append(f"HighNoise: {high_noise_value:.0f}")
                    accelerator.print(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                        f"Epoch {epoch_id + 1}/{_epoch_total} | "
                        f"Step {steps_in_epoch}/{_steps_per_epoch_now} | "
                        f"Global Step {global_step} (= checkpoint-{global_step}) | "
                        f"{' | '.join(loss_parts)} | "
                        f"LR: {current_lr:.2e}"
                    )

            if (
                max_optimizer_steps
                and optimizer_update_count >= max_optimizer_steps
            ):
                stop_training = True
                accelerator.print(
                    "Reached optimizer-update limit: "
                    f"{optimizer_update_count}/{max_optimizer_steps} "
                    f"at per-rank global_step={global_step}"
                )
                break

        # Epoch 结束提示
        if accelerator.is_local_main_process:
            accelerator.print(f"\n{'='*60}")
            accelerator.print(f"✓ Epoch {epoch_id + 1} 完成 | global_step: {global_step}")
            accelerator.print(f"{'='*60}\n")

        if hasattr(progress_bar, 'close'):
            progress_bar.close()

        if stop_training:
            break

    _heartbeat_stop.set()

    # 训练结束总结
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        accelerator.print(f"\n{'='*60}")
        accelerator.print(f"训练完成！")
        accelerator.print(f"{'='*60}")
        accelerator.print(f"• 总 per-rank step 数 (global_step): {global_step}")
        accelerator.print(f"• 总 optimizer update 数: {optimizer_update_count}")
        accelerator.print(f"• 总训练 epoch: {num_epochs}")
        accelerator.print(f"• 最终学习率: {scheduler.get_last_lr()[0]:.2e}")
        accelerator.print(f"• 训练错误次数: {train_error_count}")
        accelerator.print(f"• 模型保存路径: {model_logger.output_path}")
        accelerator.print(f"{'='*60}\n")

    if os.environ.get("DIFFSYNTH_SKIP_FINAL_CHECKPOINT", "0") == "1":
        accelerator.print(
            "Final checkpoint save skipped by "
            "DIFFSYNTH_SKIP_FINAL_CHECKPOINT=1; the latest complete "
            "checkpoint-N state directory remains resumable and all "
            "model-step safetensors remain available."
        )
        return

    final_ckpt = os.path.join(model_logger.output_path, "checkpoint-final")
    if skip_accelerate_state_save:
        accelerator.print(f"⚠ 跳过最终 accelerate.save_state（DIFFSYNTH_SKIP_ACCELERATE_SAVE_STATE=1）: {final_ckpt}")
    else:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        saved_ok = _save_checkpoint_state_safely(
            accelerator,
            final_ckpt,
            global_step=global_step,
            epoch_id=_last_epoch_id,
            batch_idx=_last_batch_idx,
            world_size=new_world_size,
            batch_size=batch_size,
            dataset_size=_dataset_size,
            strict_save_errors=strict_save_errors,
            save_state_enabled=save_state_enabled,
        )
        if saved_ok:
            accelerator.print(f"✓ 保存最终训练状态到: {final_ckpt}")
        else:
            accelerator.print(f"⚠ 最终训练状态保存失败: {final_ckpt}")
    if accelerator.is_main_process and skip_accelerate_state_save:
        _save_training_state(
            final_ckpt,
            global_step=global_step,
            epoch_id=_last_epoch_id,
            batch_idx=_last_batch_idx,
            world_size=new_world_size,
            batch_size=batch_size,
            dataset_size=_dataset_size,
        )

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model_logger.save_model(
            accelerator,
            model,
            "model-final.safetensors",
            save_full_model=True,
            synchronize=not weak_save_barrier,
        )
    except Exception as e:
        accelerator.print(f"⚠ Final checkpoint save failed: {type(e).__name__}: {e}")
        if strict_save_errors:
            raise
    if is_lora_training:
        model_logger.save_model(
            accelerator,
            model,
            "lora-final.safetensors",
            save_full_model=False,
            synchronize=not weak_save_barrier,
            trainable_filter="lora",
        )


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args=None,
):
    if args is not None:
        num_workers = args.dataset_num_workers

    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model, dataloader = accelerator.prepare(model, dataloader)

    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
