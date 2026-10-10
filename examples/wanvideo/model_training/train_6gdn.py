import argparse
import glob
import json
import math
import os
import sys
from datetime import timedelta

import accelerate
import torch
import warnings
from accelerate import InitProcessGroupKwargs
from accelerate.utils import set_seed
from peft.tuners.tuners_utils import BaseTunerLayer

from diffsynth.core import load_state_dict
from diffsynth.diffusion import *
from diffsynth.diffusion.logger import ModelLogger
from diffsynth.diffusion.runner import launch_training_task
from diffsynth.core.data.uni_dataset import UnifiedDataset
from diffsynth.models.wan_video_dit_linear import SelfAttention as GDN_SelfAttention
from diffsynth.pipelines.pipeline_cross import ModelConfig, WanVideoPipeline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train import WanTrainingModule as BaseWanTrainingModule
from train import env_float
from train import memory_token_dropout_from_env
from train import wan_parser as base_wan_parser

os.environ["TOKENIZERS_PARALLELISM"] = "false"


_FORBIDDEN_MEMORY_COMPONENTS = (
    "memory_attn",
    "norm_mem",
    "mem_scale",
    "q_former",
)


def _strict_nonmemory_wan_enabled():
    raw = os.environ.get("DIFFSYNTH_STRICT_NONMEMORY_WAN", "0").strip().lower()
    if raw not in {"0", "1", "false", "true"}:
        raise ValueError(
            "DIFFSYNTH_STRICT_NONMEMORY_WAN must be 0/1/false/true, "
            f"got {raw!r}"
        )
    return raw in {"1", "true"}


def _assert_standard_nonmemory_wan(model, label):
    if not _strict_nonmemory_wan_enabled():
        return

    class_name = f"{model.__class__.__module__}.{model.__class__.__name__}"
    expected_class = "diffsynth.models.wan_video_dit.WanModel"
    if class_name != expected_class:
        raise RuntimeError(
            f"{label} must use the checkpoint-native non-memory WanModel; "
            f"expected={expected_class}, actual={class_name}"
        )

    named_objects = set(name for name, _ in model.named_modules())
    named_objects.update(name for name, _ in model.named_parameters())
    named_objects.update(name for name, _ in model.named_buffers())
    forbidden_names = sorted(
        name
        for name in named_objects
        if any(
            component in name.lower()
            for component in _FORBIDDEN_MEMORY_COMPONENTS
        )
    )
    forbidden_attributes = sorted(
        f"blocks.{block_index}.{component}"
        for block_index, block in enumerate(model.blocks)
        for component in _FORBIDDEN_MEMORY_COMPONENTS
        if hasattr(block, component)
    )
    if forbidden_names or forbidden_attributes:
        raise RuntimeError(
            f"{label} contains forbidden memory-design components: "
            f"names={forbidden_names[:8]} attributes={forbidden_attributes[:8]}"
        )

    print(
        f"Strict non-memory Wan audit passed: {label} | "
        f"class={class_name} blocks={len(model.blocks)} "
        "forbidden_components=0"
    )


def _validate_deepspeed_runtime_config(accelerator):
    config_path = os.environ.get("DIFFSYNTH_DEEPSPEED_CONFIG_PATH", "").strip()
    if not config_path:
        return

    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if plugin is None:
        raise RuntimeError(
            "DIFFSYNTH_DEEPSPEED_CONFIG_PATH is set, but Accelerate did not "
            "initialize a DeepSpeed plugin"
        )

    with open(config_path, encoding="utf-8") as handle:
        expected_config = json.load(handle)
    expected_zero = expected_config.get("zero_optimization", {})
    actual_zero = plugin.deepspeed_config.get("zero_optimization", {})
    checked_keys = (
        "stage",
        "overlap_comm",
        "contiguous_gradients",
        "reduce_bucket_size",
        "allgather_bucket_size",
    )
    mismatches = {
        key: (expected_zero.get(key), actual_zero.get(key))
        for key in checked_keys
        if expected_zero.get(key) != actual_zero.get(key)
    }
    if mismatches:
        raise RuntimeError(
            "Accelerate did not apply the generated DeepSpeed ZeRO config: "
            f"{mismatches}"
        )

    accelerator.print(
        "DeepSpeed runtime config verified: "
        f"stage={actual_zero['stage']} "
        f"overlap_comm={actual_zero['overlap_comm']} "
        f"contiguous_gradients={actual_zero['contiguous_gradients']} "
        f"reduce_bucket_size={actual_zero['reduce_bucket_size']} "
        f"allgather_bucket_size={actual_zero['allgather_bucket_size']} "
        f"source={config_path}"
    )


def _parse_block_indices(block_indices):
    if block_indices is None:
        return None
    if isinstance(block_indices, (list, tuple)):
        indices = [int(i) for i in block_indices]
    else:
        text = str(block_indices).strip()
        if not text:
            return None
        indices = [int(item.strip()) for item in text.split(",") if item.strip()]

    seen = set()
    deduped = []
    for idx in indices:
        if idx in seen:
            continue
        seen.add(idx)
        deduped.append(idx)
    return deduped


def _gdn_block_indices(num_layers: int, group_size: int = 5, block_indices=None):
    explicit_indices = _parse_block_indices(block_indices)
    if explicit_indices is not None:
        invalid = [idx for idx in explicit_indices if idx < 0 or idx >= num_layers]
        if invalid:
            raise ValueError(
                f"gdn_block_indices contains out-of-range blocks {invalid}; num_layers={num_layers}"
            )
        return explicit_indices

    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")
    return list(range(0, num_layers, group_size))


def _teacher_model_kwargs(teacher_dit):
    first_block = teacher_dit.blocks[0]
    # The checkpoint-native WanModel keeps ffn_dim on each DiTBlock, while
    # some local WanModel variants also mirror it on the top-level model.
    # Derive it from the block so both layouts can construct the hybrid student.
    ffn_dim = getattr(teacher_dit, "ffn_dim", None)
    if ffn_dim is None:
        ffn_dim = getattr(first_block, "ffn_dim", None)
    if ffn_dim is None:
        ffn_dim = first_block.ffn[0].out_features

    return dict(
        dim=teacher_dit.dim,
        in_dim=teacher_dit.in_dim,
        ffn_dim=ffn_dim,
        out_dim=teacher_dit.head.head.out_features // math.prod(teacher_dit.patch_size),
        text_dim=teacher_dit.text_embedding[0].in_features,
        freq_dim=teacher_dit.freq_dim,
        eps=first_block.norm1.eps,
        patch_size=teacher_dit.patch_size,
        num_heads=first_block.num_heads,
        num_layers=len(teacher_dit.blocks),
        has_image_input=teacher_dit.has_image_input,
        has_image_pos_emb=getattr(teacher_dit, "has_image_pos_emb", False),
        has_ref_conv=getattr(teacher_dit, "has_ref_conv", False),
        add_control_adapter=teacher_dit.control_adapter is not None,
        in_dim_control_adapter=getattr(teacher_dit.control_adapter, "in_dim", 24) if teacher_dit.control_adapter is not None else 24,
        seperated_timestep=teacher_dit.seperated_timestep,
        require_vae_embedding=teacher_dit.require_vae_embedding,
        require_clip_embedding=teacher_dit.require_clip_embedding,
        fuse_vae_embedding_in_latents=teacher_dit.fuse_vae_embedding_in_latents,
    )


def _replace_selected_self_attn(
    student_dit,
    block_indices,
    device,
    torch_dtype,
    gdn_expand_v=2.0,
    gdn_head_dim=256,
    gdn_hidden_size=0,
    gdn_num_heads=0,
):
    for idx in block_indices:
        block = student_dit.blocks[idx]
        block.self_attn = GDN_SelfAttention(
            dim=block.dim,
            num_heads=block.num_heads,
            eps=block.norm1.eps,
            expand_v=gdn_expand_v,
            head_dim=gdn_head_dim,
            layer_idx=idx,
            gdn_hidden_size=gdn_hidden_size,
            gdn_num_heads=gdn_num_heads,
        ).to(device=device, dtype=torch_dtype)
    return student_dit


def build_default_trainable_models(num_layers: int = 30, group_size: int = 5, block_indices=None):
    return ",".join(
        f"dit.blocks.{i}.self_attn" for i in _gdn_block_indices(num_layers, group_size, block_indices)
    )


def _load_gdn_student_init(student_dit, checkpoint_path, block_indices):
    loaded_state = load_state_dict(checkpoint_path, torch_dtype=None, device="cpu")
    lora_keys = {
        name
        for name in loaded_state
        if ".lora_A." in name or ".lora_B." in name
    }
    student_state = {
        name: value for name, value in loaded_state.items() if name not in lora_keys
    }
    if lora_keys:
        print(
            "Student initialization ignored LoRA adapter keys; pass the same mixed "
            f"checkpoint through --lora_checkpoint to restore them: count={len(lora_keys)}"
        )
    checkpoint_keys = set(student_state)
    expected_gdn_keys_by_block = {
        block_index: {
            f"blocks.{block_index}.self_attn.{name}"
            for name in student_dit.blocks[block_index].self_attn.state_dict()
        }
        for block_index in block_indices
    }
    expected_gdn_keys = set().union(*expected_gdn_keys_by_block.values())

    allow_partial_raw = os.environ.get(
        "DIFFSYNTH_ALLOW_PARTIAL_GDN_INIT", "0"
    ).strip().lower()
    if allow_partial_raw not in {"0", "1", "false", "true"}:
        raise ValueError(
            "DIFFSYNTH_ALLOW_PARTIAL_GDN_INIT must be 0/1/false/true, "
            f"got {allow_partial_raw!r}"
        )
    allow_partial = allow_partial_raw in {"1", "true"}

    covered_blocks = []
    partial_blocks = []
    for block_index, block_keys in expected_gdn_keys_by_block.items():
        present = block_keys & checkpoint_keys
        if present == block_keys:
            covered_blocks.append(block_index)
        elif present:
            partial_blocks.append(
                (block_index, len(present), len(block_keys), sorted(block_keys - present)[:4])
            )
    if partial_blocks:
        raise RuntimeError(
            "Student initialization partially covers one or more GDN blocks; "
            f"each covered block must be complete: {partial_blocks[:4]}"
        )

    missing_gdn_keys = sorted(expected_gdn_keys - checkpoint_keys)
    if missing_gdn_keys and not allow_partial:
        raise RuntimeError(
            "Student initialization does not fully cover the selected GDN blocks: "
            f"missing {len(missing_gdn_keys)}/{len(expected_gdn_keys)} keys; "
            f"sample={missing_gdn_keys[:8]}"
        )
    if allow_partial and not covered_blocks:
        raise RuntimeError(
            "Partial student initialization did not fully cover any selected GDN block"
        )

    model_state = student_dit.state_dict()
    unexpected_keys = sorted(checkpoint_keys - set(model_state))
    if unexpected_keys:
        raise RuntimeError(
            "Student initialization contains keys that do not exist in the "
            f"configured GDN model: count={len(unexpected_keys)}; "
            f"sample={unexpected_keys[:8]}"
        )

    shape_mismatches = [
        (name, tuple(student_state[name].shape), tuple(model_state[name].shape))
        for name in checkpoint_keys
        if student_state[name].shape != model_state[name].shape
    ]
    if shape_mismatches:
        raise RuntimeError(
            "Student initialization tensor shapes do not match the configured "
            f"GDN model: count={len(shape_mismatches)}; "
            f"sample={shape_mismatches[:4]}"
        )

    load_res = student_dit.load_state_dict(student_state, strict=False)
    if load_res.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys remained after validated student initialization: "
            f"{load_res.unexpected_keys[:8]}"
        )
    print(
        f"Loaded student init ckpt: {checkpoint_path} | "
        f"loaded_keys={len(checkpoint_keys)} "
        f"gdn_coverage={len(expected_gdn_keys) - len(missing_gdn_keys)}/"
        f"{len(expected_gdn_keys)} "
        f"covered_blocks={covered_blocks} "
        f"random_blocks={[idx for idx in block_indices if idx not in covered_blocks]} "
        f"missing_base_keys={len(load_res.missing_keys)} unexpected_keys=0"
    )
    del loaded_state, student_state, model_state


class WanTrainingModule6GDN(BaseWanTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
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
        gdn_group_size=5,
        gdn_block_indices=None,
        gdn_expand_v=2.0,
        gdn_head_dim=256,
        gdn_hidden_size=0,
        gdn_num_heads=0,
        no_teacher=False,
        kd_loss_weight=0.05,
    ):
        if no_teacher:
            self._init_without_teacher(
                model_paths=model_paths,
                model_id_with_origin_paths=model_id_with_origin_paths,
                tokenizer_path=tokenizer_path,
                audio_processor_path=audio_processor_path,
                trainable_models=trainable_models,
                lora_base_model=lora_base_model,
                lora_target_modules=lora_target_modules,
                lora_rank=lora_rank,
                lora_checkpoint=lora_checkpoint,
                preset_lora_path=preset_lora_path,
                preset_lora_model=preset_lora_model,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                extra_inputs=extra_inputs,
                fp8_models=fp8_models,
                offload_models=offload_models,
                device=device,
                task=task,
                max_timestep_boundary=max_timestep_boundary,
                min_timestep_boundary=min_timestep_boundary,
                student_init_ckpt=student_init_ckpt,
                gdn_group_size=gdn_group_size,
                gdn_block_indices=gdn_block_indices,
                gdn_expand_v=gdn_expand_v,
                gdn_head_dim=gdn_head_dim,
                gdn_hidden_size=gdn_hidden_size,
                gdn_num_heads=gdn_num_heads,
            )
            return

        # Build the teacher pipeline directly. Calling BaseWanTrainingModule
        # here would first allocate a temporary 30-layer all-GDN student, then
        # discard it. That transient model is especially expensive for the
        # format-v9 3072-hidden GDN and can OOM before the requested 7/14/21
        # hybrid is even constructed.
        DiffusionTrainingModule.__init__(self)
        if not use_gradient_checkpointing:
            warnings.warn(
                "Gradient checkpointing is detected as disabled. "
                "To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing."
            )
            use_gradient_checkpointing = True

        self.pipe = self._load_gdn_pipeline(
            model_paths=model_paths,
            model_id_with_origin_paths=model_id_with_origin_paths,
            tokenizer_path=tokenizer_path,
            audio_processor_path=audio_processor_path,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        if self.pipe.dit is None:
            raise RuntimeError(
                "WanVideoPipeline did not load pipe.dit; cannot distill GDN model."
            )
        self.teacher_dit = self.pipe.dit
        self.teacher_dit.eval()
        self.teacher_dit.requires_grad_(False)
        _assert_standard_nonmemory_wan(self.teacher_dit, "teacher")

        block_indices = _gdn_block_indices(
            len(self.teacher_dit.blocks),
            group_size=gdn_group_size,
            block_indices=gdn_block_indices,
        )
        print(f"Using grouped GDN blocks: {block_indices}")

        student_class = type(self.teacher_dit)
        student = student_class(**_teacher_model_kwargs(self.teacher_dit)).to(
            device=device,
            dtype=torch.bfloat16,
        )
        student = _replace_selected_self_attn(
            student,
            block_indices,
            device,
            torch.bfloat16,
            gdn_expand_v=gdn_expand_v,
            gdn_head_dim=gdn_head_dim,
            gdn_hidden_size=gdn_hidden_size,
            gdn_num_heads=gdn_num_heads,
        )

        teacher_state = self.teacher_dit.state_dict()
        load_res = student.load_state_dict(teacher_state, strict=False)
        print(
            "Loaded teacher weights into hybrid student | "
            f"missing_keys={len(load_res.missing_keys)} unexpected_keys={len(load_res.unexpected_keys)}"
        )

        if student_init_ckpt:
            _load_gdn_student_init(student, student_init_ckpt, block_indices)

        _assert_standard_nonmemory_wan(student, "hybrid student")

        self.student_dit = student
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

        self.pipe = self.split_pipeline_units(
            task, self.pipe, trainable_models, lora_base_model
        )
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.memory_tokens_enabled = not _strict_nonmemory_wan_enabled()
        self.memory_token_dropout = (
            memory_token_dropout_from_env(default=1.0)
            if self.memory_tokens_enabled
            else 1.0
        )
        if not self.memory_tokens_enabled:
            print("Memory-token conditioning is strictly disabled.")
        self._init_distillation_objective(kd_loss_weight)
        self._init_training_loss_config(
            default_cfg_scale=1.0,
            default_trajectory_kd_steps=1,
        )
        self._init_repair_feature_config(block_indices, lora_base_model)
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": self.custom_distillation_loss,
            "sft:train": self.custom_distillation_loss,
            "kd_sft:data_process": lambda pipe, *args: args,
            "kd_sft": self.custom_distillation_loss,
            "kd_sft:train": self.custom_distillation_loss,
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    def _init_repair_feature_config(self, gdn_block_indices, lora_base_model):
        self.repair_feature_weight = env_float(
            "DIFFSYNTH_REPAIR_FEATURE_WEIGHT", 0.0, min_value=0.0
        )
        self.repair_parent_kd_weight = env_float(
            "DIFFSYNTH_REPAIR_PARENT_KD_WEIGHT", 0.0, min_value=0.0
        )
        if self.repair_parent_kd_weight > 0.0 and lora_base_model is None:
            raise ValueError(
                "DIFFSYNTH_REPAIR_PARENT_KD_WEIGHT requires --lora_base_model"
            )
        configured = os.environ.get(
            "DIFFSYNTH_REPAIR_FEATURE_BLOCKS", ""
        ).strip()
        self.repair_feature_blocks = (
            _parse_block_indices(configured) if configured else []
        )
        if self.repair_feature_weight > 0.0 and not self.repair_feature_blocks:
            raise ValueError(
                "DIFFSYNTH_REPAIR_FEATURE_WEIGHT>0 requires "
                "DIFFSYNTH_REPAIR_FEATURE_BLOCKS"
            )
        invalid = [
            index
            for index in self.repair_feature_blocks
            if index not in gdn_block_indices
        ]
        if invalid:
            raise ValueError(
                "Repair feature KD must target converted GDN blocks; "
                f"invalid={invalid}, gdn_blocks={list(gdn_block_indices)}"
            )
        checkpoint_raw = os.environ.get(
            "DIFFSYNTH_REPAIR_FEATURE_CHECKPOINT", "1"
        ).strip().lower()
        if checkpoint_raw not in {"0", "1", "false", "true"}:
            raise ValueError(
                "DIFFSYNTH_REPAIR_FEATURE_CHECKPOINT must be 0/1/false/true"
            )
        self.repair_feature_checkpoint = checkpoint_raw in {"1", "true"}
        if self.repair_feature_blocks:
            print(
                "GDN repair feature KD: "
                f"blocks={self.repair_feature_blocks} "
                f"weight={self.repair_feature_weight:.6g} "
                f"parent_kd={self.repair_parent_kd_weight:.6g} "
                f"checkpoint={int(self.repair_feature_checkpoint)}"
            )

    def _teacher_prediction_with_repair_features(
        self,
        pipe,
        latents,
        timestep,
        inputs_shared,
        inputs_posi,
        inputs_nega,
        cfg_scale,
        fuse_vae_embedding_in_latents,
    ):
        """Run the softmax teacher and retain inputs/targets for isolated KD.

        The student GDN attention is later evaluated on the *same* normalized
        hidden state and RoPE tensor seen by the teacher. This avoids conflating
        the local attention replacement error with errors accumulated by all
        preceding hybrid blocks.
        """

        if cfg_scale != 1.0:
            raise ValueError(
                "Attention feature repair currently requires cfg_scale=1; "
                "a CFG teacher performs two forwards and would make hook "
                "captures ambiguous"
            )
        captures = {}
        handles = []

        def make_hook(block_index):
            def capture(_module, args, output):
                if len(args) < 2:
                    raise RuntimeError(
                        f"Teacher self-attention block {block_index} did not "
                        "receive the expected (hidden, freqs) inputs"
                    )
                if isinstance(output, tuple):
                    output = output[0]
                captures[block_index] = (
                    args[0].detach(),
                    args[1].detach(),
                    output.detach(),
                )

            return capture

        for block_index in self.repair_feature_blocks:
            handles.append(
                self.teacher_dit.blocks[block_index].self_attn.register_forward_hook(
                    make_hook(block_index)
                )
            )
        try:
            prediction = self._model_prediction_with_cfg(
                pipe,
                self.teacher_dit,
                latents,
                timestep,
                inputs_shared,
                inputs_posi,
                inputs_nega,
                cfg_scale,
                fuse_vae_embedding_in_latents,
                use_checkpointing=False,
            )
        finally:
            for handle in handles:
                handle.remove()

        missing = sorted(set(self.repair_feature_blocks) - set(captures))
        if missing:
            raise RuntimeError(
                f"Teacher attention hooks did not capture blocks: {missing}"
            )
        return prediction, captures

    def _repair_feature_distillation_loss(self, student_dit, captures):
        losses = []
        metrics = {}
        for block_index in self.repair_feature_blocks:
            hidden, freqs, teacher_output = captures[block_index]
            student_attention = student_dit.blocks[block_index].self_attn
            if self.repair_feature_checkpoint:
                student_output = torch.utils.checkpoint.checkpoint(
                    student_attention,
                    hidden,
                    freqs,
                    use_reentrant=False,
                )
            else:
                student_output = student_attention(hidden, freqs)
            if isinstance(student_output, tuple):
                student_output = student_output[0]

            # ReHyAt uses an activation-matching objective. Normalize by the
            # teacher magnitude so a few high-energy blocks cannot dominate.
            numerator = torch.nn.functional.l1_loss(
                student_output.float(), teacher_output.float()
            )
            denominator = teacher_output.float().abs().mean().clamp_min(1e-6)
            block_loss = numerator / denominator
            losses.append(block_loss)
            metrics[f"repair_feature_block_{block_index}"] = block_loss.detach()

        if not losses:
            reference = next(student_dit.parameters())
            return reference.new_zeros((), dtype=torch.float32), metrics
        return torch.stack(losses).mean(), metrics

    def _load_gdn_pipeline(
        self,
        model_paths,
        model_id_with_origin_paths,
        tokenizer_path,
        audio_processor_path,
        fp8_models,
        offload_models,
        device,
    ):
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
            t5_path = os.path.join(
                local_dir, "models_t5_umt5-xxl-enc-bf16.pth"
            )
            vae_candidates = (
                os.path.join(local_dir, "Wan2.2_VAE.pth"),
                os.path.join(local_dir, "Wan2.1_VAE.pth"),
            )
            vae_path = next(
                (path for path in vae_candidates if os.path.isfile(path)), None
            )
            dit_shards = sorted(
                glob.glob(
                    os.path.join(local_dir, "diffusion_pytorch_model*.safetensors")
                )
            )
            if not dit_shards or vae_path is None:
                raise FileNotFoundError(
                    f"Incomplete Wan model under {local_dir}: "
                    f"dit_files={len(dit_shards)} vae={vae_path}"
                )
            if tokenizer_path is None:
                tokenizer_path = os.path.join(local_dir, "google/umt5-xxl")
            model_configs = [
                ModelConfig(path=t5_path),
                ModelConfig(path=dit_shards),
                ModelConfig(path=vae_path),
            ]
            warnings.warn(
                "Using local fallback model paths. Pass --model_paths or "
                "--model_id_with_origin_paths for explicit control."
            )
        tokenizer_config = (
            ModelConfig(path=tokenizer_path)
            if tokenizer_path is not None
            else None
        )
        audio_processor_config = (
            ModelConfig(path=audio_processor_path)
            if audio_processor_path is not None
            else None
        )
        return WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            audio_processor_config=audio_processor_config,
        )

    def _init_without_teacher(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
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
        gdn_group_size=5,
        gdn_block_indices=None,
        gdn_expand_v=2.0,
        gdn_head_dim=256,
        gdn_hidden_size=0,
        gdn_num_heads=0,
    ):
        DiffusionTrainingModule.__init__(self)
        if not use_gradient_checkpointing:
            warnings.warn(
                "Gradient checkpointing is detected as disabled. "
                "To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing."
            )
            use_gradient_checkpointing = True

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
            dit_shards = sorted(glob.glob(os.path.join(local_dir, "diffusion_pytorch_model-*.safetensors")))
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

        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )

        if self.pipe.dit is None:
            raise RuntimeError("WanVideoPipeline did not load pipe.dit; cannot train GDN model.")
        _assert_standard_nonmemory_wan(self.pipe.dit, "stage-2 base student")

        block_indices = _gdn_block_indices(
            len(self.pipe.dit.blocks),
            group_size=gdn_group_size,
            block_indices=gdn_block_indices,
        )
        print(f"Using grouped GDN blocks: {block_indices}")
        print("No-teacher mode enabled: training hybrid GDN DiT directly on dataset targets.")

        self.pipe.dit = _replace_selected_self_attn(
            self.pipe.dit,
            block_indices,
            device,
            torch.bfloat16,
            gdn_expand_v=gdn_expand_v,
            gdn_head_dim=gdn_head_dim,
            gdn_hidden_size=gdn_hidden_size,
            gdn_num_heads=gdn_num_heads,
        )
        self.student_dit = self.pipe.dit
        self.teacher_dit = None

        if student_init_ckpt:
            _load_gdn_student_init(
                self.pipe.dit, student_init_ckpt, block_indices
            )
        _assert_standard_nonmemory_wan(self.pipe.dit, "stage-2 hybrid student")
        torch.cuda.empty_cache()

        if os.environ.get("DIFFSYNTH_FORCE_FP32_TIME_MLP", "1") == "1":
            for dit_module in [self.pipe.dit, self.pipe.dit2]:
                if dit_module is None:
                    continue
                if hasattr(dit_module, "time_embedding") and hasattr(dit_module, "time_projection"):
                    dit_module.time_embedding = dit_module.time_embedding.float()
                    dit_module.time_projection = dit_module.time_projection.float()
            print("Enabled fp32 timestep MLP for Student.")

        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        self.switch_pipe_to_training_mode(
            self.pipe,
            trainable_models,
            lora_base_model,
            lora_target_modules,
            lora_rank,
            lora_checkpoint,
            preset_lora_path,
            preset_lora_model,
            task=task,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.memory_tokens_enabled = not _strict_nonmemory_wan_enabled()
        self.memory_token_dropout = (
            memory_token_dropout_from_env(default=1.0)
            if self.memory_tokens_enabled
            else 1.0
        )
        if not self.memory_tokens_enabled:
            print("Memory-token conditioning is strictly disabled.")
        self._init_training_loss_config(
            default_cfg_scale=1.0,
            default_trajectory_kd_steps=1,
        )
        self.lora_base_kd_weight = env_float(
            "DIFFSYNTH_LORA_BASE_KD_WEIGHT", 0.0, min_value=0.0
        )
        self.detail_edge_weight = env_float(
            "DIFFSYNTH_DETAIL_EDGE_WEIGHT", 0.0, min_value=0.0
        )
        self.replay_kd_weight = env_float(
            "DIFFSYNTH_REPLAY_KD_WEIGHT",
            self.lora_base_kd_weight,
            min_value=0.0,
        )
        self.detail_outside_kd_weight = env_float(
            "DIFFSYNTH_DETAIL_OUTSIDE_KD_WEIGHT",
            self.lora_base_kd_weight,
            min_value=0.0,
        )
        self.face_roi_weight = env_float(
            "DIFFSYNTH_FACE_ROI_WEIGHT", 0.0, min_value=0.0
        )
        self.hand_roi_weight = env_float(
            "DIFFSYNTH_HAND_ROI_WEIGHT", 0.0, min_value=0.0
        )
        self.temporal_roi_weight = env_float(
            "DIFFSYNTH_TEMPORAL_ROI_WEIGHT", 0.0, min_value=0.0
        )
        self.roi_laplacian_weight = env_float(
            "DIFFSYNTH_ROI_LAPLACIAN_WEIGHT", 0.0, min_value=0.0
        )
        if max(
            self.lora_base_kd_weight,
            self.replay_kd_weight,
            self.detail_outside_kd_weight,
        ) > 0.0 and lora_base_model is None:
            raise ValueError(
                "Base-preservation KD requires --lora_base_model"
            )
        print(
            "Detail-preserving SFT: "
            f"lora_base_kd_weight={self.lora_base_kd_weight:.6g} "
            f"replay_kd={self.replay_kd_weight:.6g} "
            f"detail_outside_kd={self.detail_outside_kd_weight:.6g} "
            f"face_roi={self.face_roi_weight:.6g} "
            f"hand_roi={self.hand_roi_weight:.6g} "
            f"temporal_roi={self.temporal_roi_weight:.6g} "
            f"roi_laplacian={self.roi_laplacian_weight:.6g} "
            f"legacy_edge={self.detail_edge_weight:.6g}"
        )
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": self.custom_sft_loss,
            "sft:train": self.custom_sft_loss,
            "kd_sft:data_process": lambda pipe, *args: args,
            "kd_sft": self.custom_sft_loss,
            "kd_sft:train": self.custom_sft_loss,
            "direct_sft:data_process": lambda pipe, *args: args,
            "direct_sft": self.custom_sft_loss,
            "direct_sft:train": self.custom_sft_loss,
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary

    @staticmethod
    def _set_lora_adapters_enabled(model, enabled):
        adapter_layers = [
            module for module in model.modules()
            if isinstance(module, BaseTunerLayer)
        ]
        if not adapter_layers:
            raise RuntimeError(
                "Base-preservation KD requested, but no PEFT adapter layers were found"
            )
        for layer in adapter_layers:
            layer.enable_adapters(enabled)
        return len(adapter_layers)

    @staticmethod
    def _spatial_edge_loss(predicted_clean, clean_latents):
        predicted_clean = predicted_clean.float()
        clean_latents = clean_latents.float()
        pred_dx = predicted_clean[..., 1:] - predicted_clean[..., :-1]
        clean_dx = clean_latents[..., 1:] - clean_latents[..., :-1]
        pred_dy = predicted_clean[..., 1:, :] - predicted_clean[..., :-1, :]
        clean_dy = clean_latents[..., 1:, :] - clean_latents[..., :-1, :]
        return 0.5 * (
            torch.nn.functional.smooth_l1_loss(pred_dx, clean_dx)
            + torch.nn.functional.smooth_l1_loss(pred_dy, clean_dy)
        )

    @staticmethod
    def _boxes_to_latent_mask(boxes, sampled_frame_indices, reference):
        """Rasterize sparse normalized xyxy boxes on a latent video grid."""

        if boxes is None or sampled_frame_indices is None:
            return reference.new_zeros(
                (reference.shape[0], 1, *reference.shape[2:]),
                dtype=torch.float32,
            )
        boxes = torch.as_tensor(boxes, device=reference.device, dtype=torch.float32)
        sampled_frame_indices = torch.as_tensor(
            sampled_frame_indices, device=reference.device, dtype=torch.float32
        ).flatten()
        if boxes.ndim == 4 and boxes.shape[0] == 1:
            boxes = boxes[0]
        if boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError(f"ROI boxes must have shape [K,M,4], got {tuple(boxes.shape)}")
        if sampled_frame_indices.numel() != boxes.shape[0]:
            raise ValueError(
                "ROI sampled frame count does not match boxes: "
                f"{sampled_frame_indices.numel()} != {boxes.shape[0]}"
            )
        batch, _, frames, height, width = reference.shape
        if batch != 1:
            raise ValueError("ROI replay currently requires per-device batch size 1")
        original_last = sampled_frame_indices.max().clamp_min(1.0)
        latent_positions = torch.linspace(
            0.0, original_last, frames, device=reference.device
        )
        nearest = (
            latent_positions[:, None] - sampled_frame_indices[None, :]
        ).abs().argmin(dim=1)
        selected = boxes[nearest]
        mask = reference.new_zeros((1, 1, frames, height, width), dtype=torch.float32)
        for frame_index in range(frames):
            for box in selected[frame_index]:
                if box[0] < 0 or box[2] <= box[0] or box[3] <= box[1]:
                    continue
                x1 = max(0, min(width - 1, int(torch.floor(box[0] * width).item())))
                y1 = max(0, min(height - 1, int(torch.floor(box[1] * height).item())))
                x2 = max(x1 + 1, min(width, int(torch.ceil(box[2] * width).item())))
                y2 = max(y1 + 1, min(height, int(torch.ceil(box[3] * height).item())))
                mask[:, :, frame_index, y1:y2, x1:x2] = 1.0
        # One-pixel feathering prevents a hard optimization seam at ROI edges.
        if height >= 3 and width >= 3:
            mask = torch.nn.functional.avg_pool3d(
                mask, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)
            ).clamp_(0.0, 1.0)
        return mask

    @staticmethod
    def _masked_mse(prediction, target, mask):
        mask = mask.to(device=prediction.device, dtype=torch.float32)
        denominator = mask.sum() * prediction.shape[1]
        if denominator.item() <= 0:
            return prediction.new_zeros((), dtype=torch.float32)
        return (
            (prediction.float() - target.float()).square() * mask
        ).sum() / denominator

    @staticmethod
    def _masked_temporal_loss(prediction, target, mask):
        if prediction.shape[2] <= 1:
            return prediction.new_zeros((), dtype=torch.float32)
        pred_delta = torch.diff(prediction.float(), dim=2)
        target_delta = torch.diff(target.float(), dim=2)
        temporal_mask = torch.minimum(mask[:, :, 1:], mask[:, :, :-1])
        denominator = temporal_mask.sum() * prediction.shape[1]
        if denominator.item() <= 0:
            return prediction.new_zeros((), dtype=torch.float32)
        elementwise = torch.nn.functional.smooth_l1_loss(
            pred_delta, target_delta, reduction="none"
        )
        return (elementwise * temporal_mask).sum() / denominator

    @staticmethod
    def _masked_spatial_gradient_loss(prediction, target, mask):
        losses = []
        for dimension in (-1, -2):
            pred_gradient = torch.diff(prediction.float(), dim=dimension)
            target_gradient = torch.diff(target.float(), dim=dimension)
            if dimension == -1:
                gradient_mask = torch.minimum(mask[..., 1:], mask[..., :-1])
            else:
                gradient_mask = torch.minimum(
                    mask[..., 1:, :], mask[..., :-1, :]
                )
            denominator = gradient_mask.sum() * prediction.shape[1]
            if denominator.item() > 0:
                elementwise = torch.nn.functional.smooth_l1_loss(
                    pred_gradient, target_gradient, reduction="none"
                )
                losses.append((elementwise * gradient_mask).sum() / denominator)
        if not losses:
            return prediction.new_zeros((), dtype=torch.float32)
        return torch.stack(losses).mean()

    def custom_sft_loss(self, pipe, inputs_shared, inputs_posi, inputs_nega):
        clean_latents = inputs_shared.get("input_latents")
        if clean_latents is None:
            clean_latents = inputs_shared["latents"]

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
        detail_kind_value = inputs_shared.get("detail_kind", 0)
        if hasattr(detail_kind_value, "item"):
            detail_kind_value = detail_kind_value.item()
        detail_kind = int(detail_kind_value)
        if detail_kind not in {0, 1, 2}:
            raise ValueError(f"detail_kind must be 0/1/2, got {detail_kind}")
        base_pred = None
        parent_kd_requested = max(
            self.lora_base_kd_weight,
            self.replay_kd_weight,
            self.detail_outside_kd_weight,
        ) > 0.0
        if parent_kd_requested:
            self._set_lora_adapters_enabled(pipe.dit, False)
            try:
                with torch.no_grad():
                    base_pred = self._model_prediction_with_cfg(
                        pipe,
                        pipe.dit,
                        noisy_latents,
                        timesteps,
                        inputs_shared,
                        inputs_posi,
                        inputs_nega,
                        cfg_scale,
                        fuse_vae_embedding_in_latents,
                        use_checkpointing=False,
                    )
            finally:
                self._set_lora_adapters_enabled(pipe.dit, True)

        pred = self._model_prediction_with_cfg(
            pipe,
            pipe.dit,
            noisy_latents,
            timesteps,
            inputs_shared,
            inputs_posi,
            inputs_nega,
            cfg_scale,
            fuse_vae_embedding_in_latents,
            use_checkpointing=True,
        )
        target = pipe.scheduler.training_target(clean_latents, noise, timesteps)
        pred_for_loss = self._prediction_for_loss(pred, loss_mask)
        target_for_loss = self._prediction_for_loss(target, loss_mask)
        task_mse = torch.nn.functional.mse_loss(
            pred_for_loss.float(), target_for_loss.float()
        )
        training_weight = self._training_weight(
            pipe, timesteps, task_mse.device, task_mse.dtype
        )
        task_loss = task_mse * training_weight

        sampled_frame_indices = inputs_shared.get(
            "roi_sampled_frame_indices", None
        )
        face_mask = self._boxes_to_latent_mask(
            inputs_shared.get("face_roi_boxes", None),
            sampled_frame_indices,
            pred,
        )
        hand_mask = self._boxes_to_latent_mask(
            inputs_shared.get("hand_roi_boxes", None),
            sampled_frame_indices,
            pred,
        )
        active_roi_mask = (
            face_mask if detail_kind == 1 else hand_mask if detail_kind == 2 else face_mask.new_zeros(face_mask.shape)
        )
        if detail_kind and active_roi_mask.sum().item() <= 0:
            raise RuntimeError(
                f"detail_kind={detail_kind} sample has no valid latent ROI"
            )

        face_roi_loss = task_loss.new_zeros(())
        hand_roi_loss = task_loss.new_zeros(())
        if detail_kind == 1 and self.face_roi_weight > 0.0:
            face_roi_loss = self._masked_mse(pred, target, face_mask) * training_weight
        if detail_kind == 2 and self.hand_roi_weight > 0.0:
            hand_roi_loss = self._masked_mse(pred, target, hand_mask) * training_weight

        base_kd_loss = task_loss.new_zeros(())
        applied_base_kd_weight = (
            self.detail_outside_kd_weight
            if detail_kind
            else self.replay_kd_weight
        )
        if base_pred is not None:
            if detail_kind:
                base_kd_loss = self._masked_mse(
                    pred,
                    base_pred,
                    (1.0 - active_roi_mask).clamp_(0.0, 1.0),
                ) * training_weight
            else:
                base_for_loss = self._prediction_for_loss(base_pred, loss_mask)
                base_kd_loss = torch.nn.functional.mse_loss(
                    pred_for_loss.float(), base_for_loss.float()
                ) * training_weight

        detail_sample_value = inputs_shared.get("detail_sample", True)
        if hasattr(detail_sample_value, "item"):
            detail_sample_value = detail_sample_value.item()
        detail_sample = bool(detail_sample_value)
        detail_edge_loss = task_loss.new_zeros(())
        needs_predicted_clean = (
            (self.detail_edge_weight > 0.0 and detail_sample)
            or (self.temporal_roi_weight > 0.0 and detail_kind)
            or (self.roi_laplacian_weight > 0.0 and detail_kind)
        )
        predicted_clean = None
        if needs_predicted_clean:
            sigma = torch.as_tensor(
                pipe.scheduler.sigmas[timestep_id],
                device=pred.device,
                dtype=torch.float32,
            )
            predicted_clean = noisy_latents.float() - sigma * pred.float()
        if self.detail_edge_weight > 0.0 and detail_sample:
            detail_edge_loss = self._spatial_edge_loss(
                predicted_clean, clean_latents
            )

        temporal_roi_loss = task_loss.new_zeros(())
        roi_laplacian_loss = task_loss.new_zeros(())
        if detail_kind and self.temporal_roi_weight > 0.0:
            temporal_roi_loss = self._masked_temporal_loss(
                predicted_clean, clean_latents, active_roi_mask
            )
        if detail_kind and self.roi_laplacian_weight > 0.0:
            roi_laplacian_loss = self._masked_spatial_gradient_loss(
                predicted_clean, clean_latents, active_roi_mask
            )

        self.latest_loss_metrics = {
            "task_loss": task_loss.detach(),
            "raw_mse": task_mse.detach(),
            "base_kd_loss": base_kd_loss.detach(),
            "detail_edge_loss": detail_edge_loss.detach(),
            "face_roi_loss": face_roi_loss.detach(),
            "hand_roi_loss": hand_roi_loss.detach(),
            "temporal_roi_loss": temporal_roi_loss.detach(),
            "roi_laplacian_loss": roi_laplacian_loss.detach(),
            "roi_fraction": active_roi_mask.mean().detach(),
            "detail_kind": torch.tensor(float(detail_kind), device=task_loss.device),
            "base_kd_weight": torch.tensor(
                applied_base_kd_weight, device=task_loss.device
            ),
            "detail_sample": torch.tensor(
                float(detail_sample), device=task_loss.device
            ),
            "training_weight": training_weight.detach(),
            "timestep": timesteps.detach().float().mean(),
            "cfg_scale": torch.tensor(cfg_scale, device=task_loss.device),
            "high_noise": torch.tensor(float(high_noise), device=task_loss.device),
        }
        return (
            task_loss
            + applied_base_kd_weight * base_kd_loss
            + self.face_roi_weight * face_roi_loss
            + self.hand_roi_weight * hand_roi_loss
            + self.temporal_roi_weight * temporal_roi_loss
            + self.roi_laplacian_weight * roi_laplacian_loss
            + self.detail_edge_weight * detail_edge_loss
        )


def wan_gdn_parser():
    parser = base_wan_parser()
    parser.add_argument(
        "--gdn_group_size",
        type=int,
        default=5,
        help="Every N blocks, replace the first self-attn with GDN. Default: 5",
    )
    parser.add_argument(
        "--gdn_block_indices",
        type=str,
        default="",
        help=(
            "Comma-separated DiT block indices whose self-attn is replaced by GDN. "
            "When set, this overrides --gdn_group_size."
        ),
    )
    parser.add_argument(
        "--no_teacher",
        action="store_true",
        help="Do not instantiate a teacher DiT. Train the hybrid GDN DiT directly against dataset diffusion targets.",
    )
    parser.add_argument(
        "--gdn_expand_v",
        type=float,
        default=float(os.environ.get("GDN_EXPAND_V", 2.0)),
        help="FLA GatedDeltaNet value expansion. Use 1.0 for mapping-compatible GDN.",
    )
    parser.add_argument(
        "--gdn_head_dim",
        type=int,
        default=int(os.environ.get("GDN_HEAD_DIM", 256)),
        help="FLA head dimension; non-positive selects dim/num_heads.",
    )
    parser.add_argument(
        "--gdn_hidden_size",
        type=int,
        default=int(os.environ.get("GDN_HIDDEN_SIZE", 0)),
        help=(
            "Internal projected GDN hidden size; non-positive keeps the Wan "
            "block width. Format-v9 exact-large mapping uses 3072."
        ),
    )
    parser.add_argument(
        "--gdn_num_heads",
        type=int,
        default=int(os.environ.get("GDN_NUM_HEADS", 0)),
        help=(
            "Internal projected GDN head count; non-positive keeps the Wan "
            "head count. Format-v9 exact-large mapping uses 24."
        ),
    )
    parser.add_argument(
        "--training_seed",
        type=int,
        default=int(os.environ.get("TRAINING_SEED", -1)),
        help="Reproducible data/noise seed; negative keeps the legacy random behavior.",
    )
    return parser


if __name__ == "__main__":
    os.environ.setdefault("DIFFSYNTH_T2V_TRAINING", "1")

    _nccl_timeout_sec = int(os.environ.get("DIFFSYNTH_NCCL_TIMEOUT_SEC", "7200"))
    _nccl_timeout = timedelta(seconds=_nccl_timeout_sec)

    import torch.distributed as _td

    _orig_init_pg = _td.init_process_group
    _orig_new_group = _td.new_group

    def _init_pg_patched(*args, **kwargs):
        kwargs.setdefault("timeout", _nccl_timeout)
        print(f"[NCCL patch] init_process_group timeout={kwargs['timeout']}", flush=True)
        return _orig_init_pg(*args, **kwargs)

    def _new_group_patched(*args, **kwargs):
        kwargs.setdefault("timeout", _nccl_timeout)
        return _orig_new_group(*args, **kwargs)

    _td.init_process_group = _init_pg_patched
    _td.new_group = _new_group_patched

    parser = wan_gdn_parser()
    args = parser.parse_args()
    if args.training_seed >= 0:
        set_seed(args.training_seed, device_specific=False)
        print(f"Training seed: {args.training_seed}", flush=True)

    lora_only = os.environ.get("DIFFSYNTH_LORA_ONLY", "0").strip() == "1"
    if lora_only:
        if not args.lora_base_model:
            raise ValueError(
                "DIFFSYNTH_LORA_ONLY=1 requires --lora_base_model"
            )
        args.trainable_models = None
    elif not args.trainable_models:
        args.trainable_models = build_default_trainable_models(
            group_size=args.gdn_group_size,
            block_indices=args.gdn_block_indices,
        )

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters),
            InitProcessGroupKwargs(timeout=_nccl_timeout),
        ],
    )
    _validate_deepspeed_runtime_config(accelerator)

    qworld_common_config = os.environ.get(
        "DIFFSYNTH_QWORLD_COMMON_CONFIG", ""
    ).strip()
    if qworld_common_config:
        if args.batch_size != 1:
            raise ValueError(
                "QWorld common adapter requires --batch_size 1 because the "
                "DiffSynth runner uses an identity collator"
            )
        from diffsynth.core.data.qworld_common_dataset import (
            build_qworld_common_dataset_from_env,
        )

        dataset = build_qworld_common_dataset_from_env(
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            repeat=args.dataset_repeat,
            seed=args.training_seed if args.training_seed >= 0 else 0,
        )
        accelerator.print(
            "Dataset backend: QWorld common | "
            f"config={qworld_common_config} | samples={len(dataset)}"
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

    model = WanTrainingModule6GDN(
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
        gdn_group_size=args.gdn_group_size,
        gdn_block_indices=args.gdn_block_indices,
        gdn_expand_v=args.gdn_expand_v,
        gdn_head_dim=args.gdn_head_dim,
        gdn_hidden_size=args.gdn_hidden_size,
        gdn_num_heads=args.gdn_num_heads,
        no_teacher=args.no_teacher,
        kd_loss_weight=args.kd_loss_weight,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )

    launch_training_task(accelerator, dataset, model, model_logger, args=args)
