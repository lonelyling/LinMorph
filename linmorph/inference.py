"""Shared inference implementation; command-line entry points live in scripts/."""

import glob
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


PROJECT_ROOT = SCRIPT_DIR.parent


DEFAULT_GDN_BLOCKS = "0,1,2,3,5,6,7,8,10,11,12,13,15,16,17,18,20,21,22,23,25,26,27,28"


def project_path(path):
    path = str(path)
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", path):
        return path
    if os.path.isabs(path):
        return path
    return str(PROJECT_ROOT / path)


def parse_block_indices(text, num_layers=None):
    indices = []
    seen = set()
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        idx = int(item)
        if idx in seen:
            continue
        seen.add(idx)
        indices.append(idx)
    if not indices:
        raise ValueError("gdn_block_indices is empty")
    if num_layers is not None:
        invalid = [idx for idx in indices if idx < 0 or idx >= num_layers]
        if invalid:
            raise ValueError(f"gdn_block_indices out of range for num_layers={num_layers}: {invalid}")
    return indices


def checkpoint_sort_key(path):
    match = re.fullmatch(r"model-step-(\d+)\.safetensors", path.name)
    if match:
        return (1, int(match.group(1)), path.stat().st_mtime)
    if path.name == "model-final.safetensors":
        return (2, 10**18, path.stat().st_mtime)
    return (0, -1, path.stat().st_mtime)


def resolve_checkpoint(path_or_dir, ckpt_step=None):
    path = Path(project_path(path_or_dir))
    if path.is_file():
        if ckpt_step is not None:
            raise ValueError(f"ckpt_step was provided but PATH is already a file: {path}")
        return str(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Run directory or checkpoint file not found: {path}")

    if ckpt_step is not None:
        checkpoint = path / f"model-step-{ckpt_step}.safetensors"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found for step {ckpt_step}: {checkpoint}")
        return str(checkpoint)

    candidates = list(path.glob("model-step-*.safetensors"))
    final_ckpt = path / "model-final.safetensors"
    if final_ckpt.exists():
        candidates.append(final_ckpt)
    if not candidates:
        raise FileNotFoundError(f"No model-step-*.safetensors or model-final.safetensors found in {path}")
    candidates.sort(key=checkpoint_sort_key)
    return str(candidates[-1])


def patch_torch_distributed_tensor_for_fla():
    try:
        import torch.distributed._tensor as old_dtensor
        import torch.distributed.tensor as dtensor
    except Exception:
        return

    for name in ("Replicate", "Shard", "Placement", "distribute_module", "DTensor", "DeviceMesh"):
        if not hasattr(dtensor, name) and hasattr(old_dtensor, name):
            setattr(dtensor, name, getattr(old_dtensor, name))


def patch_diffusion_package_for_inference():
    if "diffsynth.diffusion" in sys.modules:
        return

    import importlib.util
    import types

    diffusion_dir = PROJECT_ROOT / "diffsynth" / "diffusion"
    flow_match_path = diffusion_dir / "flow_match.py"
    spec = importlib.util.spec_from_file_location("diffsynth.diffusion.flow_match", flow_match_path)
    flow_match = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = flow_match
    spec.loader.exec_module(flow_match)

    diffusion_pkg = types.ModuleType("diffsynth.diffusion")
    diffusion_pkg.__file__ = str(diffusion_dir / "__init__.py")
    diffusion_pkg.__path__ = [str(diffusion_dir)]
    diffusion_pkg.FlowMatchScheduler = flow_match.FlowMatchScheduler
    sys.modules["diffsynth.diffusion"] = diffusion_pkg


def import_runtime_deps():
    global imageio, np, torch, Image
    global ModelConfig, load_state_dict, GDN_SelfAttention, WanVideoPipeline

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    import imageio
    import numpy as np
    import torch
    from PIL import Image

    patch_torch_distributed_tensor_for_fla()
    patch_diffusion_package_for_inference()

    from diffsynth.core import ModelConfig, load_state_dict
    from diffsynth.models.wan_video_dit_linear import SelfAttention as GDN_SelfAttention
    from diffsynth.pipelines.pipeline_cross import WanVideoPipeline


def torch_dtype_from_arg(dtype):
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def select_memory_mode(args):
    if not str(args.device).startswith("cuda"):
        if args.memory_mode == "cpu_offload":
            raise ValueError("--memory_mode cpu_offload requires a CUDA device")
        return "resident", None
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device requested ({args.device}), but torch.cuda.is_available() is False. "
            "Run with the same Python environment used for training."
        )

    free_bytes, total_bytes = torch.cuda.mem_get_info(args.device)
    gib = 1024 ** 3
    memory_info = {
        "device": str(args.device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "free_gb": round(free_bytes / gib, 2),
        "total_gb": round(total_bytes / gib, 2),
    }
    print(
        f"[VRAM] visible={memory_info['cuda_visible_devices'] or '(not set)'} "
        f"device={args.device} free={memory_info['free_gb']:.2f} GiB "
        f"total={memory_info['total_gb']:.2f} GiB"
    )

    if args.memory_mode == "auto":
        mode = "resident" if memory_info["free_gb"] >= args.resident_min_free_gb else "cpu_offload"
    else:
        mode = args.memory_mode
    print(
        f"[VRAM] memory_mode={mode}"
        + (
            f" (auto threshold={args.resident_min_free_gb:.2f} GiB)"
            if args.memory_mode == "auto"
            else ""
        )
    )
    if mode == "resident" and memory_info["free_gb"] < args.resident_min_free_gb:
        print(
            f"[WARN] resident mode has only {memory_info['free_gb']:.2f} GiB free; "
            "use --memory_mode cpu_offload or select a less occupied GPU if OOM occurs."
        )
    return mode, memory_info


def replace_selected_self_attn(
    dit,
    block_indices,
    device,
    torch_dtype,
    expand_v,
    head_dim,
    hidden_size,
    num_heads,
):
    for idx in block_indices:
        block = dit.blocks[idx]
        block.self_attn = GDN_SelfAttention(
            dim=block.dim,
            num_heads=block.num_heads,
            eps=block.norm1.eps,
            expand_v=expand_v,
            head_dim=head_dim,
            layer_idx=idx,
            gdn_hidden_size=hidden_size,
            gdn_num_heads=num_heads,
        ).to(device=device, dtype=torch_dtype)


def enable_sequential_cpu_offload(pipe, device):
    candidate_names = (
        "text_encoder",
        "image_encoder",
        "dit",
        "dit2",
        "vae",
        "motion_controller",
        "vace",
        "vace2",
        "vap",
        "animate_adapter",
        "audio_encoder",
    )
    managed_models = {
        name: getattr(pipe, name)
        for name in candidate_names
        if getattr(pipe, name, None) is not None
    }
    loaded_names = set()

    def load_models_to_device(self, model_names):
        requested = set(model_names)
        target_names = {name for name in requested if name in managed_models}
        if target_names == loaded_names:
            return

        for name in sorted(loaded_names - target_names):
            managed_models[name].to(device="cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for name in sorted(target_names - loaded_names):
            managed_models[name].to(device=device)

        loaded_names.clear()
        loaded_names.update(target_names)
        print(f"[VRAM] GPU-resident components: {sorted(loaded_names) or ['none']}")

    pipe.load_models_to_device = types.MethodType(load_models_to_device, pipe)
    pipe.sequential_cpu_offload_enabled = True
    print("[INFO] Sequential CPU offload enabled for pipeline components.")


def normalize_checkpoint_keys(state_dict):
    prefixes = ("pipe.dit.", "dit.")
    for prefix in prefixes:
        matched = [key for key in state_dict if key.startswith(prefix)]
        if matched and len(matched) == len(state_dict):
            return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def expected_gdn_self_attn_keys(dit, block_indices):
    expected = set()
    for idx in block_indices:
        prefix = f"blocks.{idx}.self_attn."
        expected.update(prefix + key for key in dit.blocks[idx].self_attn.state_dict().keys())
    return expected


def validate_24gdn_checkpoint_keys(state_dict, dit, block_indices):
    expected = expected_gdn_self_attn_keys(dit, block_indices)
    actual = set(state_dict)
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(
            "Checkpoint is missing GDN self_attn weights for the selected 24-GDN blocks. "
            f"first missing keys: {missing[:8]}"
        )

    selected_prefixes = tuple(f"blocks.{idx}.self_attn." for idx in block_indices)
    outside_selected_self_attn = [
        key for key in state_dict
        if key.startswith("blocks.") and not key.startswith(selected_prefixes)
    ]
    if outside_selected_self_attn:
        print(
            "[CKPT] checkpoint contains keys beyond selected 24-GDN self_attn; "
            "loading with strict=False. first keys:",
            outside_selected_self_attn[:8],
        )


LORA_KEY_PATTERN = re.compile(
    r"^(?P<module>.+)\.lora_(?P<side>[AB])(?:\.default)?\.weight$"
)


def split_mixed_lora_checkpoint(state_dict):
    base_state = {}
    lora_state = {}
    lora_modules = {}
    for key, value in state_dict.items():
        match = LORA_KEY_PATTERN.match(key)
        if match is None:
            base_state[key] = value
            continue
        lora_state[key] = value
        module_name = match.group("module")
        lora_modules.setdefault(module_name, {})[match.group("side")] = key

    if not lora_state:
        return base_state, lora_state, [], None

    incomplete = {
        name: sorted({"A", "B"} - set(sides))
        for name, sides in lora_modules.items()
        if set(sides) != {"A", "B"}
    }
    if incomplete:
        raise ValueError(
            "Checkpoint has incomplete LoRA A/B pairs: "
            f"{list(incomplete.items())[:8]}"
        )

    ranks = {
        int(lora_state[sides["A"]].shape[0])
        for sides in lora_modules.values()
    }
    if len(ranks) != 1:
        raise ValueError(f"Checkpoint uses multiple LoRA ranks: {sorted(ranks)}")
    lora_rank = ranks.pop()
    return base_state, lora_state, sorted(lora_modules), lora_rank


def inject_checkpoint_lora(dit, lora_state, target_modules, lora_rank):
    from peft import LoraConfig, inject_adapter_in_model

    available_modules = set(dict(dit.named_modules()))
    missing_targets = sorted(set(target_modules) - available_modules)
    if missing_targets:
        raise ValueError(
            "LoRA checkpoint targets modules absent from the configured model: "
            f"{missing_targets[:8]}"
        )

    # Training uses lora_alpha == lora_rank, so the effective PEFT scale is 1.
    config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=target_modules,
    )
    dit = inject_adapter_in_model(config, dit)
    model_lora_keys = {
        key for key in dit.state_dict() if LORA_KEY_PATTERN.match(key)
    }
    checkpoint_lora_keys = set(lora_state)
    if checkpoint_lora_keys != model_lora_keys:
        raise ValueError(
            "Injected LoRA layout does not match checkpoint: "
            f"missing={sorted(model_lora_keys - checkpoint_lora_keys)[:8]} "
            f"unexpected={sorted(checkpoint_lora_keys - model_lora_keys)[:8]}"
        )
    load_res = dit.load_state_dict(lora_state, strict=False)
    if load_res.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys while loading LoRA adapters: "
            f"{load_res.unexpected_keys[:8]}"
        )
    dit.eval()
    print(
        f"[CKPT] PEFT LoRA loaded: rank={lora_rank} "
        f"modules={len(target_modules)} tensors={len(lora_state)} scale=1"
    )
    return dit


def extract_video(output):
    if isinstance(output, tuple):
        return output[0]
    return output


def normalize_video_frames(video):
    if isinstance(video, torch.Tensor):
        video = video.detach().cpu().float().numpy()
    elif isinstance(video, Image.Image):
        video = [video]
    elif isinstance(video, list):
        if len(video) == 1 and isinstance(video[0], (list, tuple, torch.Tensor, np.ndarray)):
            video = video[0]
        if all(isinstance(frame, Image.Image) for frame in video):
            return [np.ascontiguousarray(np.array(frame.convert("RGB"))) for frame in video]
        video = np.array(video)
    else:
        video = np.array(video)

    while isinstance(video, np.ndarray) and video.ndim > 4 and video.shape[0] == 1:
        video = video[0]

    if not isinstance(video, np.ndarray):
        raise TypeError(f"Unsupported video type: {type(video)!r}")
    if video.ndim == 5:
        raise ValueError(f"Expected one video, got batched video array with shape={video.shape}")
    if video.ndim != 4:
        raise ValueError(f"Expected 4D video array, got shape={video.shape}")

    if video.shape[-1] in (1, 3, 4):
        frames = video
    elif video.shape[0] in (1, 3, 4):
        frames = np.transpose(video, (1, 2, 3, 0))
    elif video.shape[1] in (1, 3, 4):
        frames = np.transpose(video, (0, 2, 3, 1))
    else:
        raise ValueError(f"Cannot infer channel dimension from video shape={video.shape}")

    if frames.dtype != np.uint8:
        if frames.size and frames.max() <= 1.0:
            frames = frames * 255
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    out = []
    for frame in frames:
        if frame.ndim != 3 or frame.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Invalid frame shape for video encoding: {frame.shape}")
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        if frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        out.append(np.ascontiguousarray(frame))
    return out


def save_video_file(video, save_path, fps):
    frames = normalize_video_frames(video)
    if not frames:
        raise ValueError("No frames to save.")
    height, width = frames[0].shape[:2]
    tmp_path = save_path + ".tmp.mp4"
    print(f"[INFO] Saving {len(frames)} frames to {save_path}; frame={width}x{height}, fps={fps}")

    ffmpeg_bin = shutil.which("ffmpeg")
    ffmpeg_errors = []
    if ffmpeg_bin:
        encoder_attempts = [
            ("libx264", ["-vcodec", "libx264", "-preset", "ultrafast", "-crf", "18"]),
            ("libopenh264", ["-vcodec", "libopenh264", "-b:v", "8M"]),
            ("mpeg4", ["-vcodec", "mpeg4", "-q:v", "2"]),
        ]
        for encoder_name, encoder_args in encoder_attempts:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                *encoder_args,
                "-threads",
                "1",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                tmp_path,
            ]
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except OSError as exc:
                detail = f"{encoder_name}: failed to start ffmpeg: {exc}"
                ffmpeg_errors.append(detail)
                print(f"[WARN] {detail}")
                continue

            write_error = None
            try:
                for frame in frames:
                    proc.stdin.write(frame.tobytes())
            except BrokenPipeError as exc:
                write_error = exc
            finally:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass

            stderr = proc.stderr.read()
            returncode = proc.wait()
            if returncode == 0 and os.path.exists(tmp_path):
                os.replace(tmp_path, save_path)
                return

            stderr_text = stderr.decode("utf-8", errors="replace").strip()
            if not stderr_text:
                stderr_text = "no stderr; ffmpeg may have been killed by the job memory limit"
            detail = f"{encoder_name}: returncode={returncode}, stderr={stderr_text}"
            if write_error is not None:
                detail += f", pipe_error={write_error}"
            ffmpeg_errors.append(detail)
            print(f"[WARN] ffmpeg encode failed: {detail}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    try:
        print("[WARN] Direct ffmpeg encoders failed; trying imageio fallback.")
        with imageio.get_writer(tmp_path, format="FFMPEG", fps=fps, quality=6) as writer:
            for frame in frames:
                writer.append_data(frame)
        os.replace(tmp_path, save_path)
    except Exception as exc:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        details = " | ".join(ffmpeg_errors) if ffmpeg_errors else "ffmpeg executable not found"
        raise RuntimeError(
            f"Unable to save video to {save_path}. "
            f"Direct ffmpeg errors: {details}. imageio error: {exc}"
        ) from exc


def make_side_by_side(left_path, right_path, out_path):
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        print("[WARN] --make_side_by_side requested, but ffmpeg was not found.")
        return False
    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        left_path,
        "-i",
        right_path,
        "-filter_complex",
        "[0:v][1:v]hstack=inputs=2[v]",
        "-map",
        "[v]",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return True


def build_pipeline(args, block_indices, memory_mode, replace_gdn=True):
    model_dir = Path(project_path(args.model_dir))
    tokenizer_dir = model_dir / "google" / "umt5-xxl"
    t5_path = model_dir / "models_t5_umt5-xxl-enc-bf16.pth"
    vae_path = next(
        (
            path
            for path in (
                model_dir / "Wan2.2_VAE.pth",
                model_dir / "Wan2.1_VAE.pth",
            )
            if path.exists()
        ),
        model_dir / "Wan2.2_VAE.pth",
    )
    dit_shards = sorted(glob.glob(str(model_dir / "diffusion_pytorch_model-*.safetensors")))
    if not dit_shards:
        single_dit = model_dir / "diffusion_pytorch_model.safetensors"
        if single_dit.exists():
            dit_shards = [str(single_dit)]
    missing = [str(path) for path in (tokenizer_dir, t5_path, vae_path) if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing base model files: {missing}")
    if not dit_shards:
        raise FileNotFoundError(f"No base DiT shards found under {model_dir}")

    dtype = torch_dtype_from_arg(args.dtype)
    model_device = "cpu" if memory_mode == "cpu_offload" else args.device
    model_config_kwargs = (
        {"computation_device": "cpu", "computation_dtype": dtype}
        if memory_mode == "cpu_offload"
        else {}
    )
    print("[INFO] Loading base Wan pipeline ...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=args.device,
        model_configs=[
            ModelConfig(path=str(t5_path), **model_config_kwargs),
            ModelConfig(path=dit_shards, **model_config_kwargs),
            ModelConfig(path=str(vae_path), **model_config_kwargs),
        ],
        tokenizer_config=ModelConfig(path=str(tokenizer_dir)),
    )
    if pipe.dit2 is not None:
        raise RuntimeError(
            "pipe.dit2 is present, but this script patches only pipe.dit. "
            "Refusing to run to avoid mixed patched/unpatched inference."
        )

    if replace_gdn:
        parse_block_indices(args.gdn_block_indices, num_layers=len(pipe.dit.blocks))
        print(f"[INFO] Replacing self-attn with GDN in blocks: {block_indices}")
        replace_selected_self_attn(
            pipe.dit,
            block_indices,
            device=model_device,
            torch_dtype=dtype,
            expand_v=args.gdn_expand_v,
            head_dim=args.gdn_head_dim,
            hidden_size=args.gdn_hidden_size,
            num_heads=args.gdn_num_heads,
        )
    else:
        print("[INFO] Keeping all original Softmax self-attention layers.")
    if memory_mode == "cpu_offload":
        enable_sequential_cpu_offload(pipe, args.device)
    pipe.eval()
    return pipe


def load_checkpoint_into_pipe(pipe, ckpt_path, block_indices, dtype):
    print(f"[CKPT] Loading {ckpt_path}")
    # Preserve fp32 GDN recurrence parameters such as A_log and dt_bias.
    # Module.load_state_dict handles casting the remaining weights as needed.
    state_dict = load_state_dict(ckpt_path, torch_dtype=None, device="cpu")
    state_dict = normalize_checkpoint_keys(state_dict)
    base_state, lora_state, lora_targets, lora_rank = split_mixed_lora_checkpoint(
        state_dict
    )
    validate_24gdn_checkpoint_keys(base_state, pipe.dit, block_indices)
    load_res = pipe.dit.load_state_dict(base_state, strict=False)
    if load_res.unexpected_keys:
        raise RuntimeError(f"Unexpected keys while loading checkpoint: {load_res.unexpected_keys[:8]}")
    print(
        f"[CKPT] loaded_base_keys={len(base_state)} "
        f"missing_base_keys={len(load_res.missing_keys)} unexpected_keys={len(load_res.unexpected_keys)}"
    )
    if lora_state:
        pipe.dit = inject_checkpoint_lora(
            pipe.dit,
            lora_state=lora_state,
            target_modules=lora_targets,
            lora_rank=lora_rank,
        )
    del state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def maybe_force_fp32_time_mlp(pipe):
    if os.environ.get("DIFFSYNTH_FORCE_FP32_TIME_MLP", "1") != "1":
        return
    for dit_module in [pipe.dit, pipe.dit2]:
        if dit_module is None:
            continue
        if hasattr(dit_module, "time_embedding") and hasattr(dit_module, "time_projection"):
            dit_module.time_embedding = dit_module.time_embedding.float()
            dit_module.time_projection = dit_module.time_projection.float()
    print("[INFO] Enabled fp32 timestep MLP.")

