import torch, os, argparse, accelerate, warnings
from datetime import timedelta
from accelerate import InitProcessGroupKwargs
from diffsynth.core import load_state_dict
from diffsynth.core.data.uni_dataset import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.pipeline_cross import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
import random
import glob
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def env_float(name, default, min_value=None, max_value=None):
    value = float(os.environ.get(name, default))
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}, got {value}")
    return value


def env_int(name, default, min_value=None):
    value = int(os.environ.get(name, default))
    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}, got {value}")
    return value


def memory_token_dropout_from_env(default=0.0):
    value = float(os.environ.get("DIFFSYNTH_MEMORY_TOKEN_DROPOUT", default))
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            "DIFFSYNTH_MEMORY_TOKEN_DROPOUT must be in [0, 1], "
            f"got {value}"
        )
    return value


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        student_init_ckpt=None,
        kd_loss_weight=0.05,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models. Prefer user-provided paths; fall back to local TI2V-5B layout.
        if model_paths is not None or model_id_with_origin_paths is not None:
            model_configs = self.parse_model_configs(
                model_paths,
                model_id_with_origin_paths,
                fp8_models=fp8_models,
                offload_models=offload_models,
                device=device,
            )
        else:
            local_dir = os.environ.get(
                "WAN_LOCAL_MODEL_DIR",
                "model_weight/Wan2.2-TI2V-5B",
            )
            t5_path = os.path.join(local_dir, "models_t5_umt5-xxl-enc-bf16.pth")
            vae_path = os.path.join(local_dir, "Wan2.2_VAE.pth")
            dit_shards = sorted(
                glob.glob(os.path.join(local_dir, "diffusion_pytorch_model-*.safetensors"))
            )
            if len(dit_shards) == 0:
                raise FileNotFoundError(
                    f"No DiT shards found under {local_dir}. "
                    "Please pass --model_paths or set WAN_LOCAL_MODEL_DIR."
                )
            if tokenizer_path is None:
                tokenizer_path = os.path.join(local_dir, "google/umt5-xxl")
            model_configs = [
                ModelConfig(path=t5_path),
                ModelConfig(path=dit_shards),
                ModelConfig(path=vae_path),
            ]
            warnings.warn(
                "Using local fallback model paths. "
                "Pass --model_paths/--model_id_with_origin_paths for explicit control."
            )
        tokenizer_config = ModelConfig(path=tokenizer_path) if tokenizer_path is not None else None
        
        import math
        from diffsynth.models.wan_video_dit_mem_attn import WanModel as OriginalWanModel
        # 只加载一次 pipeline，作为 teacher（Softmax Attention）。
        # 历史代码这里调用了两次 from_pretrained，会把 5B 权重在 GPU 上分配两份并立即丢弃，
        # 留下严重的显存碎片，4 卡续跑时 Triton kernel 分配不到连续大块就 OOM。
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )

        self.teacher_dit = self.pipe.dit
        self.teacher_dit.eval()
        self.teacher_dit.requires_grad_(False)
        torch.cuda.empty_cache()
        from diffsynth.models.wan_video_dit_linear import WanModel as StudentWanModel
        import math
        
        self.student_dit = StudentWanModel(
            dim=self.teacher_dit.dim, in_dim=self.teacher_dit.in_dim, ffn_dim=self.teacher_dit.ffn_dim, 
            out_dim=self.teacher_dit.head.head.out_features // math.prod(self.teacher_dit.patch_size),
            text_dim=self.teacher_dit.text_embedding[0].in_features, freq_dim=self.teacher_dit.freq_dim,
            eps=self.teacher_dit.blocks[0].norm1.eps, patch_size=self.teacher_dit.patch_size, 
            num_heads=self.teacher_dit.blocks[0].num_heads, num_layers=len(self.teacher_dit.blocks),
            has_image_input=self.teacher_dit.has_image_input,
            seperated_timestep=self.teacher_dit.seperated_timestep,
            require_clip_embedding=self.teacher_dit.require_clip_embedding,
            require_vae_embedding=self.teacher_dit.require_vae_embedding,
            fuse_vae_embedding_in_latents=self.teacher_dit.fuse_vae_embedding_in_latents
        ).to(device=device, dtype=torch.bfloat16)
        
        self.student_dit.load_state_dict(self.teacher_dit.state_dict(), strict=False)
        if student_init_ckpt:
            student_state = load_state_dict(student_init_ckpt, torch_dtype=None, device="cpu")
            load_res = self.student_dit.load_state_dict(student_state, strict=False)
            print(
                f"Loaded student init ckpt: {student_init_ckpt} | "
                f"missing_keys={len(load_res.missing_keys)} unexpected_keys={len(load_res.unexpected_keys)}"
            )
        self.pipe.dit = self.student_dit
        torch.cuda.empty_cache()

        if os.environ.get("DIFFSYNTH_FORCE_FP32_TIME_MLP", "1") == "1":
            for dit_module in [self.pipe.dit, self.pipe.dit2, self.teacher_dit]:
                if dit_module is None:
                    continue
                if hasattr(dit_module, "time_embedding") and hasattr(dit_module, "time_projection"):
                    dit_module.time_embedding = dit_module.time_embedding.float()
                    dit_module.time_projection = dit_module.time_projection.float()
            print("Enabled fp32 timestep MLP for Student and Teacher.")

        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.memory_token_dropout = memory_token_dropout_from_env(default=0.0)
        self._init_distillation_objective(kd_loss_weight)
        self._init_training_loss_config(
            default_cfg_scale=1.0,
            default_trajectory_kd_steps=1,
        )
        self.task = task

        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": self.custom_distillation_loss,       
            "sft:train": self.custom_distillation_loss, 
            "kd_sft:data_process": lambda pipe, *args: args,
            "kd_sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: self.custom_distillation_loss(pipe, inputs_shared, inputs_posi, inputs_nega),
            "kd_sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: self.custom_distillation_loss(pipe, inputs_shared, inputs_posi, inputs_nega),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def _init_distillation_objective(self, kd_loss_weight):
        self.kd_loss_weight = float(kd_loss_weight)
        if self.kd_loss_weight < 0:
            raise ValueError(
                f"kd_loss_weight must be non-negative, got {self.kd_loss_weight}"
            )
        self.task_loss_weight = env_float(
            "DIFFSYNTH_TASK_LOSS_WEIGHT", 1.0, min_value=0.0
        )
        self.kd_loss_type = os.environ.get(
            "DIFFSYNTH_KD_LOSS_TYPE", "mse"
        ).strip().lower()
        if self.kd_loss_type not in {"mse", "l1"}:
            raise ValueError(
                "DIFFSYNTH_KD_LOSS_TYPE must be 'mse' or 'l1', "
                f"got {self.kd_loss_type!r}"
            )
        kd_weighting_raw = os.environ.get(
            "DIFFSYNTH_KD_USE_TRAINING_WEIGHT", "1"
        ).strip().lower()
        if kd_weighting_raw not in {"0", "1", "false", "true"}:
            raise ValueError(
                "DIFFSYNTH_KD_USE_TRAINING_WEIGHT must be "
                f"0/1/false/true, got {kd_weighting_raw!r}"
            )
        self.kd_use_training_weight = kd_weighting_raw in {"1", "true"}
        print(
            f"KD loss: type={self.kd_loss_type} "
            f"weight={self.kd_loss_weight:.6g}; "
            f"task loss weight={self.task_loss_weight:.6g}; "
            f"scheduler_weighted={int(self.kd_use_training_weight)}"
        )

    def _init_training_loss_config(
        self,
        default_cfg_scale=1.0,
        default_trajectory_kd_steps=1,
    ):
        self.timestep_high_noise_prob = env_float(
            "DIFFSYNTH_TIMESTEP_HIGH_NOISE_PROB", 0.0, 0.0, 1.0
        )
        self.timestep_high_noise_min = env_float(
            "DIFFSYNTH_TIMESTEP_HIGH_NOISE_MIN", 800.0, 0.0, 1000.0
        )
        self.timestep_low_noise_prob = env_float(
            "DIFFSYNTH_TIMESTEP_LOW_NOISE_PROB", 0.0, 0.0, 1.0
        )
        self.timestep_low_noise_max = env_float(
            "DIFFSYNTH_TIMESTEP_LOW_NOISE_MAX", 350.0, 0.0, 1000.0
        )
        if self.timestep_high_noise_prob + self.timestep_low_noise_prob > 1.0:
            raise ValueError(
                "High-noise and low-noise timestep probabilities must sum to <= 1"
            )
        self.train_cfg_scale = env_float(
            "DIFFSYNTH_TRAIN_CFG_SCALE", default_cfg_scale, 1.0
        )
        self.cfg_train_prob = env_float(
            "DIFFSYNTH_CFG_TRAIN_PROB", 0.0, 0.0, 1.0
        )
        self.train_negative_prompt = os.environ.get("DIFFSYNTH_NEGATIVE_PROMPT", "")
        self.trajectory_kd_steps = env_int(
            "DIFFSYNTH_TRAJECTORY_KD_STEPS",
            default_trajectory_kd_steps,
            1,
        )
        self.trajectory_kd_inference_steps = env_int(
            "DIFFSYNTH_TRAJECTORY_KD_INFERENCE_STEPS",
            50,
            1,
        )
        self.trajectory_kd_weight = env_float(
            "DIFFSYNTH_TRAJECTORY_KD_WEIGHT", 0.0, 0.0
        )
        self.trajectory_kd_detach = os.environ.get(
            "DIFFSYNTH_TRAJECTORY_KD_DETACH", "1"
        ) == "1"
        print(
            "Training loss config: "
            f"high_noise_prob={self.timestep_high_noise_prob:.3f} "
            f"high_noise_min_t={self.timestep_high_noise_min:.1f} "
            f"low_noise_prob={self.timestep_low_noise_prob:.3f} "
            f"low_noise_max_t={self.timestep_low_noise_max:.1f} "
            f"cfg_scale={self.train_cfg_scale:.3f} "
            f"cfg_prob={self.cfg_train_prob:.3f} "
            f"trajectory_kd_steps={self.trajectory_kd_steps} "
            f"trajectory_kd_inference_steps={self.trajectory_kd_inference_steps} "
            f"trajectory_kd_weight={self.trajectory_kd_weight:.6g} "
            f"trajectory_kd_detach={int(self.trajectory_kd_detach)}"
        )

    def _sample_train_cfg_scale(self):
        if self.train_cfg_scale <= 1.0:
            return 1.0
        if self.cfg_train_prob >= 1.0 or random.random() < self.cfg_train_prob:
            return self.train_cfg_scale
        return 1.0

    def _sample_scheduler_timestep(self, pipe, device):
        num_timesteps = len(pipe.scheduler.timesteps)
        min_timestep_id = int(self.min_timestep_boundary * num_timesteps)
        max_timestep_id = int(self.max_timestep_boundary * num_timesteps)
        if not 0 <= min_timestep_id < max_timestep_id <= num_timesteps:
            raise ValueError(
                "Invalid timestep boundaries: "
                f"min={self.min_timestep_boundary}, "
                f"max={self.max_timestep_boundary}"
            )

        full_ids = torch.arange(min_timestep_id, max_timestep_id)
        selector = float(torch.rand(()).item())
        use_low_noise = (
            self.timestep_low_noise_prob > 0.0
            and selector < self.timestep_low_noise_prob
        )
        use_high_noise = (
            self.timestep_high_noise_prob > 0.0
            and self.timestep_low_noise_prob
            <= selector
            < self.timestep_low_noise_prob + self.timestep_high_noise_prob
        )
        candidate_ids = full_ids
        if use_low_noise:
            timesteps_cpu = pipe.scheduler.timesteps[full_ids]
            low_noise_ids = full_ids[
                timesteps_cpu <= self.timestep_low_noise_max
            ]
            if len(low_noise_ids) > 0:
                candidate_ids = low_noise_ids
        elif use_high_noise:
            timesteps_cpu = pipe.scheduler.timesteps[full_ids]
            high_noise_ids = full_ids[timesteps_cpu >= self.timestep_high_noise_min]
            if len(high_noise_ids) > 0:
                candidate_ids = high_noise_ids

        offset = torch.randint(0, len(candidate_ids), (1,))
        timestep_id = candidate_ids[offset]
        timesteps = pipe.scheduler.timesteps[timestep_id].to(
            dtype=pipe.torch_dtype,
            device=device,
        )
        high_noise = (
            pipe.scheduler.timesteps[timestep_id].float().mean().item()
            >= self.timestep_high_noise_min
        )
        return timesteps, int(timestep_id.item()), bool(high_noise)

    def _scheduler_timestep_from_id(self, pipe, timestep_id, device):
        timestep = pipe.scheduler.timesteps[timestep_id:timestep_id + 1]
        return timestep.to(dtype=pipe.torch_dtype, device=device)

    def _trajectory_timestep_ids(self, pipe, start_timestep_id):
        num_timesteps = len(pipe.scheduler.timesteps)
        max_timestep_id = min(
            num_timesteps,
            int(self.max_timestep_boundary * num_timesteps),
        )
        stride = max(
            1,
            round(num_timesteps / self.trajectory_kd_inference_steps),
        )
        timestep_ids = []
        timestep_id = start_timestep_id
        while (
            len(timestep_ids) < self.trajectory_kd_steps
            and timestep_id < max_timestep_id
        ):
            timestep_ids.append(timestep_id)
            timestep_id += stride
        if not timestep_ids:
            timestep_ids.append(start_timestep_id)
        return timestep_ids, stride, max_timestep_id

    def _scheduler_step_between_ids(
        self,
        pipe,
        model_output,
        sample,
        current_timestep_id,
        next_timestep_id,
    ):
        if not hasattr(pipe.scheduler, "sigmas"):
            raise AttributeError(
                "Trajectory KD requires a scheduler with an explicit sigma schedule."
            )
        sigmas = pipe.scheduler.sigmas
        sigma = sigmas[current_timestep_id]
        sigma_next = (
            sigmas[next_timestep_id]
            if next_timestep_id < len(sigmas)
            else 0.0
        )
        delta_sigma = torch.as_tensor(
            sigma_next - sigma,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        return sample + model_output * delta_sigma

    def _training_weight(self, pipe, timestep, device, dtype):
        return pipe.scheduler.training_weight(timestep).to(device=device, dtype=dtype)

    def _maybe_restore_first_frame(self, latents, first_frame_latents):
        if first_frame_latents is None:
            return latents
        latents = latents.clone()
        latents[:, :, 0:1] = first_frame_latents
        return latents

    def _prediction_for_loss(self, pred, loss_mask):
        return pred[loss_mask] if loss_mask is not None else pred

    def _model_prediction_with_cfg(
        self,
        pipe,
        dit,
        latents,
        timestep,
        inputs_shared,
        inputs_posi,
        inputs_nega,
        cfg_scale,
        fuse_vae_embedding_in_latents,
        use_checkpointing=False,
    ):
        if (
            not getattr(self, "memory_tokens_enabled", True)
            and "memory_tokens" in inputs_shared
        ):
            raise RuntimeError(
                "Memory-token conditioning is forbidden for this training run"
            )
        common_kwargs = {
            "latents": latents,
            "timestep": timestep,
            "y": inputs_shared.get("y", None),
            "clip_feature": inputs_shared.get("clip_feature", None),
            "fuse_vae_embedding_in_latents": fuse_vae_embedding_in_latents,
        }
        memory_tokens = inputs_shared.get("memory_tokens", None)
        if memory_tokens is not None:
            common_kwargs["memory_tokens"] = memory_tokens
        if use_checkpointing:
            common_kwargs.update({
                "use_gradient_checkpointing": self.use_gradient_checkpointing,
                "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            })

        pred_pos = pipe.model_fn(
            dit=dit,
            context=inputs_posi.get("context", None),
            **common_kwargs,
        )
        if cfg_scale == 1.0 or inputs_nega.get("context", None) is None:
            return pred_pos

        pred_neg = pipe.model_fn(
            dit=dit,
            context=inputs_nega.get("context", None),
            **common_kwargs,
        )
        return pred_neg + cfg_scale * (pred_pos - pred_neg)
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        cfg_scale = self._sample_train_cfg_scale()
        inputs_nega = {"negative_prompt": self.train_negative_prompt}
        force_t2v_training = os.environ.get("DIFFSYNTH_T2V_TRAINING", "0") == "1"
        input_image = None if force_t2v_training else data.get("input_image", None)
        if not force_t2v_training and input_image is None and "video" in data and len(data["video"]) > 0:
            input_image = data["video"][0]
        inputs_shared = {
            "input_video": data["video"],
            "input_image": input_image,
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": cfg_scale,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def custom_distillation_loss(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        clean_latents = inputs_shared.get("input_latents", inputs_shared["latents"])

        noise = pipe.generate_noise(clean_latents.shape, rand_device=clean_latents.device).to(clean_latents.dtype)
        timesteps, timestep_id, high_noise = self._sample_scheduler_timestep(
            pipe, clean_latents.device
        )
        noisy_latents = pipe.scheduler.add_noise(clean_latents, noise, timestep=timesteps)
        if noisy_latents.dtype != clean_latents.dtype:
            noisy_latents = noisy_latents.to(clean_latents.dtype)

        fuse_vae_embedding_in_latents = inputs_shared.get("fuse_vae_embedding_in_latents", False)
        first_frame_latents = inputs_shared.get("first_frame_latents", None)
        loss_mask = None
        if fuse_vae_embedding_in_latents and first_frame_latents is not None:
            noisy_latents = self._maybe_restore_first_frame(
                noisy_latents, first_frame_latents
            )
            loss_mask = torch.ones_like(noisy_latents, dtype=torch.bool)
            loss_mask[:, :, 0:1] = False

        cfg_scale = float(inputs_shared.get("cfg_scale", 1.0))
        target = pipe.scheduler.training_target(clean_latents, noise, timesteps)
        training_weight = self._training_weight(
            pipe, timesteps, clean_latents.device, torch.float32
        )

        rollout_timestep_ids, rollout_stride, max_timestep_id = (
            self._trajectory_timestep_ids(pipe, timestep_id)
        )
        trajectory_loss_enabled = self.trajectory_kd_weight > 0.0
        if self.trajectory_kd_steps <= 1 and not trajectory_loss_enabled:
            # The baseline objective is one-step task loss plus teacher KD.
            # Avoid building an unused scheduler-rollout graph when trajectory
            # KD is explicitly disabled.
            rollout_timestep_ids = rollout_timestep_ids[:1]
        rollout_steps = len(rollout_timestep_ids)
        current_latents = noisy_latents
        kd_losses = []
        trajectory_losses = []
        task_mse = None
        task_loss = None
        repair_feature_loss = clean_latents.new_zeros((), dtype=torch.float32)
        repair_feature_metrics = {}
        repair_feature_weight = float(
            getattr(self, "repair_feature_weight", 0.0)
        )
        repair_parent_kd_weight = float(
            getattr(self, "repair_parent_kd_weight", 0.0)
        )
        repair_parent_losses = []

        for rollout_offset, current_timestep_id in enumerate(rollout_timestep_ids):
            current_timestep = self._scheduler_timestep_from_id(
                pipe, current_timestep_id, clean_latents.device
            )
            step_weight = self._training_weight(
                pipe, current_timestep, clean_latents.device, torch.float32
            )

            repair_captures = None
            with torch.no_grad():
                capture_method = getattr(
                    self, "_teacher_prediction_with_repair_features", None
                )
                if (
                    rollout_offset == 0
                    and repair_feature_weight > 0.0
                    and capture_method is not None
                ):
                    teacher_pred, repair_captures = capture_method(
                        pipe,
                        current_latents,
                        current_timestep,
                        inputs_shared,
                        inputs_posi,
                        inputs_nega,
                        cfg_scale,
                        fuse_vae_embedding_in_latents,
                    )
                else:
                    teacher_pred = self._model_prediction_with_cfg(
                        pipe,
                        self.teacher_dit,
                        current_latents,
                        current_timestep,
                        inputs_shared,
                        inputs_posi,
                        inputs_nega,
                        cfg_scale,
                        fuse_vae_embedding_in_latents,
                        use_checkpointing=False,
                    )

            if repair_captures is not None:
                repair_feature_loss, repair_feature_metrics = (
                    self._repair_feature_distillation_loss(
                        pipe.dit, repair_captures
                    )
                )
                del repair_captures

            student_pred = self._model_prediction_with_cfg(
                pipe,
                pipe.dit,
                current_latents,
                current_timestep,
                inputs_shared,
                inputs_posi,
                inputs_nega,
                cfg_scale,
                fuse_vae_embedding_in_latents,
                use_checkpointing=True,
            )

            if repair_parent_kd_weight > 0.0:
                set_adapters = getattr(self, "_set_lora_adapters_enabled", None)
                if set_adapters is None:
                    raise RuntimeError(
                        "Repair parent KD requires adapter enable/disable support"
                    )
                set_adapters(pipe.dit, False)
                try:
                    with torch.no_grad():
                        parent_pred = self._model_prediction_with_cfg(
                            pipe,
                            pipe.dit,
                            current_latents,
                            current_timestep,
                            inputs_shared,
                            inputs_posi,
                            inputs_nega,
                            cfg_scale,
                            fuse_vae_embedding_in_latents,
                            use_checkpointing=False,
                        )
                finally:
                    set_adapters(pipe.dit, True)
                repair_parent_losses.append(
                    torch.nn.functional.mse_loss(
                        self._prediction_for_loss(student_pred, loss_mask).float(),
                        self._prediction_for_loss(parent_pred, loss_mask).float(),
                    )
                    * step_weight
                )

            student_for_loss = self._prediction_for_loss(student_pred, loss_mask)
            teacher_for_loss = self._prediction_for_loss(teacher_pred.detach(), loss_mask)
            kd_loss_fn = (
                torch.nn.functional.l1_loss
                if self.kd_loss_type == "l1"
                else torch.nn.functional.mse_loss
            )
            kd_losses.append(
                kd_loss_fn(student_for_loss.float(), teacher_for_loss.float())
                * (step_weight if self.kd_use_training_weight else 1.0)
            )

            if rollout_offset == 0:
                target_for_loss = self._prediction_for_loss(target, loss_mask)
                task_mse = torch.nn.functional.mse_loss(
                    student_for_loss.float(), target_for_loss.float()
                )
                task_loss = task_mse * training_weight

            needs_next_kd_step = rollout_offset + 1 < rollout_steps
            if not needs_next_kd_step and not trajectory_loss_enabled:
                continue

            next_timestep_id = min(
                current_timestep_id + rollout_stride,
                max_timestep_id,
            )
            with torch.no_grad():
                next_teacher_latents = self._scheduler_step_between_ids(
                    pipe,
                    teacher_pred,
                    current_latents,
                    current_timestep_id,
                    next_timestep_id,
                )
                next_teacher_latents = self._maybe_restore_first_frame(
                    next_teacher_latents, first_frame_latents
                )
            next_student_latents = self._scheduler_step_between_ids(
                pipe,
                student_pred,
                current_latents,
                current_timestep_id,
                next_timestep_id,
            )
            next_student_latents = self._maybe_restore_first_frame(
                next_student_latents, first_frame_latents
            )
            if trajectory_loss_enabled:
                trajectory_losses.append(
                    torch.nn.functional.mse_loss(
                        self._prediction_for_loss(next_student_latents, loss_mask).float(),
                        self._prediction_for_loss(next_teacher_latents.detach(), loss_mask).float(),
                    ) * step_weight
                )
            if needs_next_kd_step:
                current_latents = (
                    next_student_latents.detach()
                    if self.trajectory_kd_detach
                    else next_student_latents
                )

        kd_loss = torch.stack(kd_losses).mean()
        trajectory_kd_loss = (
            torch.stack(trajectory_losses).mean()
            if trajectory_losses
            else task_loss.new_zeros(())
        )
        repair_parent_kd_loss = (
            torch.stack(repair_parent_losses).mean()
            if repair_parent_losses
            else task_loss.new_zeros(())
        )
        self.latest_loss_metrics = {
            "task_loss": task_loss.detach(),
            "kd_loss": kd_loss.detach(),
            "trajectory_kd_loss": trajectory_kd_loss.detach(),
            "raw_mse": task_mse.detach(),
            "training_weight": training_weight.detach(),
            "timestep": timesteps.detach().float().mean(),
            "cfg_scale": torch.tensor(cfg_scale, device=task_loss.device),
            "rollout_steps": torch.tensor(rollout_steps, device=task_loss.device),
            "rollout_stride": torch.tensor(rollout_stride, device=task_loss.device),
            "high_noise": torch.tensor(float(high_noise), device=task_loss.device),
            "repair_feature_loss": repair_feature_loss.detach(),
            "repair_parent_kd_loss": repair_parent_kd_loss.detach(),
        }
        self.latest_loss_metrics.update(repair_feature_metrics)
        loss = (
            self.task_loss_weight * task_loss
            + self.kd_loss_weight * kd_loss
        )
        if trajectory_loss_enabled:
            loss = loss + self.trajectory_kd_weight * trajectory_kd_loss
        if repair_feature_weight > 0.0:
            loss = loss + repair_feature_weight * repair_feature_loss
        if repair_parent_kd_weight > 0.0:
            loss = loss + repair_parent_kd_weight * repair_parent_kd_loss
        return loss

    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        with torch.no_grad():
            for unit in self.pipe.units:
                inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        inputs_shared, inputs_posi, inputs_nega = inputs

        memory_tokens = None
        memory_tokens_enabled = getattr(self, "memory_tokens_enabled", True)
        if not memory_tokens_enabled and "memory_tokens" in inputs_shared:
            raise RuntimeError(
                "Memory-token conditioning is forbidden for this training run"
            )
        if memory_tokens_enabled and self.memory_token_dropout < 1.0:
            video_latents = inputs_shared.get(
                "input_latents", inputs_shared["latents"]
            ).detach()

            with torch.no_grad():
                seq_tokens_5d = self.pipe.dit.patch_embedding(video_latents)

            _, _, f, _, _ = seq_tokens_5d.shape
            nk = random.randint(1, 3)
            if f > 0:
                selected_frames = sorted(random.sample(range(f), min(f, nk)))
                memory_tokens = seq_tokens_5d[:, :, selected_frames, :, :]
                memory_tokens = memory_tokens.flatten(2).transpose(1, 2)

            if memory_tokens is not None and random.random() < self.memory_token_dropout:
                memory_tokens = None

        if memory_tokens is None:
            inputs_shared.pop("memory_tokens", None)
        else:
            inputs_shared["memory_tokens"] = memory_tokens

        loss = self.task_to_loss[self.task](self.pipe, inputs_shared, inputs_posi, inputs_nega)
        return loss


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--student_init_ckpt", type=str, default=None, help="Optional student DiT checkpoint for stage-2 KD.")
    parser.add_argument(
        "--kd_loss_weight",
        type=float,
        default=float(os.environ.get("DIFFSYNTH_KD_LOSS_WEIGHT", 0.05)),
        help=(
            "Coefficient applied to frozen-teacher output distillation. "
            "Set DIFFSYNTH_KD_LOSS_TYPE=mse|l1 to choose the loss."
        ),
    )
    return parser


if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # NCCL 超时扩展：必须在 accelerate/DeepSpeed 初始化任何进程组之前完成。
    #
    # 为什么 InitProcessGroupKwargs 不够用？
    #   - DeepSpeed 通过 torchrun 启动时，torch.distributed 已经由 launcher
    #     用默认 600s 初始化完毕；accelerate 见到 is_initialized()==True 就复用。
    #   - ZeRO2 还会调用 dist.new_group() 建立 PG 1，默认同样是 600s。
    #
    # 解法：monkey-patch init_process_group + new_group，用 setdefault 注入
    # 自定义 timeout，保证所有进程组（包括 DeepSpeed 内部创建的）都使用新值。
    # -----------------------------------------------------------------------
    _nccl_timeout_sec = int(os.environ.get("DIFFSYNTH_NCCL_TIMEOUT_SEC", "7200"))
    _nccl_timeout = timedelta(seconds=_nccl_timeout_sec)

    import torch.distributed as _td
    _orig_init_pg  = _td.init_process_group
    _orig_new_group = _td.new_group

    def _init_pg_patched(*args, **kwargs):
        kwargs.setdefault("timeout", _nccl_timeout)
        print(f"[NCCL patch] init_process_group timeout={kwargs['timeout']}", flush=True)
        return _orig_init_pg(*args, **kwargs)

    def _new_group_patched(*args, **kwargs):
        kwargs.setdefault("timeout", _nccl_timeout)
        return _orig_new_group(*args, **kwargs)

    _td.init_process_group = _init_pg_patched
    _td.new_group          = _new_group_patched

    parser = wan_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters),
            InitProcessGroupKwargs(timeout=_nccl_timeout),
        ],
    )

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
        )
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        student_init_ckpt=args.student_init_ckpt,
        kd_loss_weight=args.kd_loss_weight,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )    

    launch_training_task(accelerator, dataset, model, model_logger, args=args)
