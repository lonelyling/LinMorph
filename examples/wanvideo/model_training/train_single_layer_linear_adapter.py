import argparse
import math
import os
import random
import sys
import types
from datetime import timedelta

import accelerate
import torch
import torch.nn.functional as F
from accelerate import InitProcessGroupKwargs
from accelerate.utils import set_seed
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from diffsynth.core import load_state_dict
from diffsynth.core.data.uni_dataset import UnifiedDataset
from diffsynth.diffusion import DiffusionTrainingModule
from diffsynth.diffusion.logger import ModelLogger
from diffsynth.diffusion.runner import launch_training_task
if os.environ.get("DIFFSYNTH_STANDARD_WAN_TEACHER", "0") == "1":
    # Use the checkpoint-native Softmax Wan implementation for mapping.  The
    # target GDN architecture is constructed independently below, so this does
    # not reduce its hidden size or head count.
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
else:
    from diffsynth.pipelines.pipeline_cross import ModelConfig, WanVideoPipeline


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from train import WanTrainingModule as BaseWanTrainingModule  # noqa: E402
from train import wan_parser as base_wan_parser  # noqa: E402


os.environ["TOKENIZERS_PARALLELISM"] = "false"


SOURCE_NAMES = ("q", "k", "v", "o")
TARGET_NAMES = ("q", "k", "v", "o", "g", "beta")
TARGET_PARAMETER_NAMES = {
    "q": "q_proj.weight",
    "k": "k_proj.weight",
    "v": "v_proj.weight",
    "o": "o_proj.weight",
    "g": "g_proj.weight",
    "beta": "b_proj.weight",
}
SHAPE_ADAPTER_NAMES = ("subheadmix",)
SHAPE_ADAPTER_IDS = {"subheadmix": 2}
AUXILIARY_TARGET_PARAMETER_NAMES = {
    "a": "a_proj.weight",
    "A_log": "A_log",
    "dt_bias": "dt_bias",
    "q_conv1d": "q_conv1d.weight",
    "k_conv1d": "k_conv1d.weight",
    "v_conv1d": "v_conv1d.weight",
    "o_norm": "o_norm.weight",
}


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


class MetaXSafeRMSNormGated(torch.nn.Module):
    """Mathematically equivalent fallback for FLA's fused gated RMSNorm.

    MetaX's Triton runtime limits private memory to 4 KiB/thread.  The fused
    kernel for a 512-wide value head requests 5 KiB and fails before either
    forward or backward can begin.  Keeping the reduction in ordinary PyTorch
    operations avoids that kernel while preserving
    ``RMSNorm(x) * silu(g)`` and the checkpoint-compatible ``weight`` name.
    """

    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.eps = float(eps)
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
        if residual is not None or prenorm or residual_in_fp32:
            raise ValueError("MetaXSafeRMSNormGated only supports the GatedDeltaNet call signature")
        x_fp32 = x.float()
        normalized = x_fp32 * torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float() * F.silu(g.float())).to(dtype=x.dtype)


def _metax_safe_gdn_forward(self, hidden_states, attention_mask=None,
                             past_key_values=None, use_cache=False,
                             output_attentions=False, **kwargs):
    """GatedDeltaNet forward with L2 normalization outside FLA's Triton kernel."""
    from einops import rearrange
    from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    del output_attentions
    batch_size, q_len, _ = hidden_states.shape
    last_state = None
    if past_key_values is not None and len(past_key_values) > self.layer_idx:
        last_state = past_key_values[self.layer_idx]
    cu_seqlens = kwargs.get("cu_seqlens", None)
    if attention_mask is not None:
        indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
        hidden_states = index_first_axis(
            rearrange(hidden_states, "b s ... -> (b s) ..."), indices
        ).unsqueeze(0)
    conv_state_q = conv_state_k = conv_state_v = None
    if last_state is not None:
        conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]
    q, conv_state_q = self.q_conv1d(self.q_proj(hidden_states), cache=conv_state_q,
                                     output_final_state=use_cache, cu_seqlens=cu_seqlens)
    k, conv_state_k = self.k_conv1d(self.k_proj(hidden_states), cache=conv_state_k,
                                     output_final_state=use_cache, cu_seqlens=cu_seqlens)
    v, conv_state_v = self.v_conv1d(self.v_proj(hidden_states), cache=conv_state_v,
                                     output_final_state=use_cache, cu_seqlens=cu_seqlens)
    q = rearrange(q, "... (h d) -> ... h d", d=self.head_k_dim)
    k = rearrange(k, "... (h d) -> ... h d", d=self.head_k_dim)
    v = rearrange(v, "... (h d) -> ... h d", d=self.head_v_dim)
    q_fp32, k_fp32 = q.float(), k.float()
    q = (q_fp32 * torch.rsqrt(q_fp32.square().sum(dim=-1, keepdim=True) + 1e-6)).to(dtype=q.dtype)
    k = (k_fp32 * torch.rsqrt(k_fp32.square().sum(dim=-1, keepdim=True) + 1e-6)).to(dtype=k.dtype)
    beta = self.b_proj(hidden_states).sigmoid()
    decay = -self.A_log.float().exp() * F.softplus(
        self.a_proj(hidden_states).float() + self.dt_bias
    )
    recurrent_state = last_state["recurrent_state"] if last_state is not None else None
    o, recurrent_state = chunk_gated_delta_rule(
        q=q, k=k, v=v, g=decay, beta=beta, initial_state=recurrent_state,
        output_final_state=use_cache, cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )
    if past_key_values is not None:
        past_key_values.update(
            recurrent_state=recurrent_state,
            conv_state=(conv_state_q, conv_state_k, conv_state_v),
            layer_idx=self.layer_idx,
            offset=q_len,
        )
    gate = rearrange(self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_v_dim)
    o = self.o_norm(o, gate)
    o = self.o_proj(rearrange(o, "b t h d -> b t (h d)"))
    if attention_mask is not None:
        o = pad_input(o.squeeze(0), indices, batch_size, q_len)
    return o, None, past_key_values


def _uniform_block_indices(num_layers, count):
    if count <= 0:
        raise ValueError(f"mapping_num_layers must be positive, got {count}")
    if count > num_layers:
        raise ValueError(
            f"mapping_num_layers={count} exceeds the model depth {num_layers}"
        )
    if count == 1:
        return [0]
    indices = [
        round(position * (num_layers - 1) / (count - 1))
        for position in range(count)
    ]
    if len(set(indices)) != count:
        raise RuntimeError(
            f"Uniform selection produced duplicate blocks: {indices}"
        )
    return indices


def _parse_mapping_block_indices(
    block_index,
    block_indices,
    num_layers,
    num_mapping_layers,
    seed,
    middle_start,
    middle_end,
):
    explicit = []
    if block_indices:
        explicit = [
            int(item.strip()) for item in str(block_indices).split(",") if item.strip()
        ]
        if len(set(explicit)) != len(explicit):
            raise ValueError(
                f"gdn_block_indices contains duplicate block indices: {explicit}"
            )
        invalid = [index for index in explicit if index < 0 or index >= num_layers]
        if invalid:
            raise ValueError(
                f"gdn_block_indices contains out-of-range blocks {invalid}; "
                f"num_layers={num_layers}"
            )
        if num_mapping_layers > 1 and len(explicit) != num_mapping_layers:
            raise ValueError(
                f"mapping_num_layers={num_mapping_layers}, but explicit "
                f"gdn_block_indices contains {len(explicit)} blocks: {explicit}"
            )
        return sorted(explicit)

    if num_mapping_layers > 1:
        return _uniform_block_indices(num_layers, num_mapping_layers)

    if block_index < 0:
        start = max(0, middle_start)
        end = min(num_layers - 1, middle_end)
        if start > end:
            raise ValueError(
                f"Invalid middle-layer interval [{middle_start}, {middle_end}] for {num_layers} layers"
            )
        block_index = random.Random(seed).choice(list(range(start, end + 1)))

    if block_index < 0 or block_index >= num_layers:
        raise ValueError(f"block_index={block_index} is outside [0, {num_layers - 1}]")
    return [block_index]


def _parse_single_block_index(
    block_index, block_indices, num_layers, seed, middle_start, middle_end
):
    indices = _parse_mapping_block_indices(
        block_index=block_index,
        block_indices=block_indices,
        num_layers=num_layers,
        num_mapping_layers=1,
        seed=seed,
        middle_start=middle_start,
        middle_end=middle_end,
    )
    if len(indices) != 1:
        raise ValueError(
            "Single-layer mapping requires exactly one --gdn_block_indices value, "
            f"got {indices}"
        )
    return indices[0]


def _manual_copy_matrix(init_mode):
    matrix = torch.zeros(len(TARGET_NAMES), len(SOURCE_NAMES), dtype=torch.float32)
    matrix[:4, :4] = torch.eye(4, dtype=torch.float32)
    if init_mode == "method2_average":
        matrix[4] = torch.tensor([1.0 / 3, 1.0 / 3, 0.0, 1.0 / 3])
        matrix[5] = torch.tensor([0.0, 0.5, 0.5, 0.0])
    elif init_mode != "manual_copy":
        raise ValueError(f"Unsupported mixing initialization: {init_mode}")
    return matrix


def _standardize_source_statistics(values):
    """Keep shape information while putting reduced source statistics on a usable scale."""
    values = values.float()
    centered = values - values.mean(dim=-1, keepdim=True)
    scale = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    return centered / scale


class MappedGDNWeightTransform(torch.nn.Module):
    """Generate one GDN layer from four frozen softmax projection matrices.

    Full mapping uses fixed shape adapters around four-way mixing: optional
    per-head repeat widening for rectangular GDN projections, head pooling for
    a_proj, per-head statistics for A_log/dt_bias, per-output-channel statistics
    for short convolutions, and within-head statistics for o_norm. These are
    experimental bases, not claims of semantic equivalence between Softmax and
    GDN parameters.
    """

    def __init__(
        self,
        source_attention,
        block_index,
        training_block_indices=None,
        low_rank=0,
        full_residual=False,
        post_full_residual=False,
        postmix=False,
        mix_granularity="head",
        mix_order="pre",
        source_specific_expansion=False,
        mixing_init="manual_copy",
        low_rank_init_std=0.02,
        gdn_expand_v=2.0,
        gdn_head_dim=256,
        gdn_hidden_size=0,
        gdn_num_heads=0,
        shape_adapter="subheadmix",
        gdn_use_short_conv=True,
        identity_short_conv=True,
        map_auxiliary_parameters=False,
        capture_random_initialization=False,
        seed=42,
    ):
        super().__init__()
        from fla.layers.gated_deltanet import GatedDeltaNet

        dim = int(source_attention.dim)
        source_num_heads = int(source_attention.num_heads)
        target_hidden_size = int(gdn_hidden_size) if gdn_hidden_size > 0 else dim
        target_num_heads = int(gdn_num_heads) if gdn_num_heads > 0 else source_num_heads
        gdn_norm_eps = float(
            getattr(getattr(source_attention, "norm_q", None), "eps", 1e-6)
        )
        if dim % source_num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by source_num_heads={source_num_heads}"
            )
        if target_hidden_size < dim or target_hidden_size % dim != 0:
            raise ValueError(
                "The projected large-GDN mapper requires an integer hidden-size "
                f"expansion: source={dim}, target={target_hidden_size}"
            )
        if target_num_heads < source_num_heads or target_num_heads % source_num_heads != 0:
            raise ValueError(
                "The projected large-GDN mapper requires an integer head-count "
                f"expansion: source={source_num_heads}, target={target_num_heads}"
            )
        source_head_dim = dim // source_num_heads
        head_dim = int(gdn_head_dim) if gdn_head_dim > 0 else source_head_dim
        if shape_adapter not in SHAPE_ADAPTER_IDS:
            raise ValueError(
                f"shape_adapter must be one of {SHAPE_ADAPTER_NAMES}, got {shape_adapter!r}"
            )
        head_v_dim = int(head_dim * gdn_expand_v)
        if not math.isclose(head_dim * gdn_expand_v, head_v_dim, rel_tol=1e-5):
            raise ValueError(
                f"gdn_expand_v={gdn_expand_v} does not produce an integer value "
                f"for head_dim={head_dim}"
            )
        for target_name, target_dim in (
            ("head_dim", head_dim),
            ("head_v_dim", head_v_dim),
        ):
            if target_dim < source_head_dim or target_dim % source_head_dim != 0:
                raise ValueError(
                    "SubheadMix requires integer widening relative to the teacher "
                    f"head width: {target_name}={target_dim}, "
                    f"teacher_head_dim={source_head_dim}"
                )
        self.shape_adapter = shape_adapter
        self.dim = dim
        self.target_dim = target_hidden_size
        self.source_num_heads = source_num_heads
        self.num_heads = target_num_heads
        self.target_num_heads = target_num_heads
        self.hidden_repeat = target_hidden_size // dim
        self.head_repeat = target_num_heads // source_num_heads
        self.source_head_dim = source_head_dim
        self.target_head_dim = head_dim
        self.target_head_v_dim = head_v_dim

        source_linears = [
            source_attention.q,
            source_attention.k,
            source_attention.v,
            source_attention.o,
        ]
        source_shapes = {tuple(layer.weight.shape) for layer in source_linears}
        if source_shapes != {(dim, dim)}:
            raise ValueError(
                "Expected four square softmax projections with a shared shape, got "
                f"{sorted(source_shapes)}"
            )

        source_weights = torch.stack(
            [layer.weight.detach().clone() for layer in source_linears], dim=0
        )
        source_weights_4d = source_weights.reshape(
            len(SOURCE_NAMES), source_num_heads, source_head_dim, dim
        )
        source_weights_by_head = source_weights_4d.mean(dim=2)
        source_head_statistics = _standardize_source_statistics(
            source_weights_4d.square().mean(dim=(2, 3)).sqrt()
        )
        source_channel_statistics = _standardize_source_statistics(
            source_weights.square().mean(dim=2).sqrt()
        )
        source_norm_statistics = _standardize_source_statistics(
            source_weights_4d.square().mean(dim=(1, 3)).sqrt()
        )
        self.register_buffer("source_weights", source_weights, persistent=False)
        self.register_buffer(
            "source_weights_by_head", source_weights_by_head, persistent=False
        )
        self.register_buffer(
            "source_head_statistics", source_head_statistics, persistent=False
        )
        self.register_buffer(
            "source_channel_statistics", source_channel_statistics, persistent=False
        )
        self.register_buffer(
            "source_norm_statistics", source_norm_statistics, persistent=False
        )
        self.register_buffer(
            "selected_block_index",
            torch.tensor(block_index, dtype=torch.int64),
            persistent=True,
        )
        if training_block_indices is None:
            training_block_indices = [block_index]
        self.register_buffer(
            "selected_block_indices",
            torch.tensor(tuple(training_block_indices), dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "mapping_format_version",
            torch.tensor(9, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "mapping_gdn_expand_v",
            torch.tensor(float(gdn_expand_v), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "mapping_gdn_head_dim",
            torch.tensor(head_dim, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "mapping_gdn_hidden_size",
            torch.tensor(target_hidden_size, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "mapping_gdn_num_heads",
            torch.tensor(target_num_heads, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "mapping_shape_adapter",
            torch.tensor(SHAPE_ADAPTER_IDS[shape_adapter], dtype=torch.int64),
            persistent=True,
        )

        cpu_rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        try:
            self.gdn = GatedDeltaNet(
                hidden_size=target_hidden_size,
                expand_v=gdn_expand_v,
                head_dim=head_dim,
                num_heads=target_num_heads,
                num_v_heads=target_num_heads,
                mode="chunk",
                use_gate=True,
                use_short_conv=gdn_use_short_conv,
                layer_idx=block_index,
                norm_eps=gdn_norm_eps,
            )
            if _env_bool("MAPPING_SAFE_GATED_NORM", True) and self.target_head_v_dim >= 256:
                fused_norm = self.gdn.o_norm
                safe_norm = MetaXSafeRMSNormGated(self.target_head_v_dim, eps=fused_norm.eps)
                safe_norm.weight.data.copy_(fused_norm.weight.detach())
                self.gdn.o_norm = safe_norm
                print(
                    "Using MetaX-safe gated RMSNorm fallback "
                    f"for head_v_dim={self.target_head_v_dim}."
                )
            if _env_bool("MAPPING_SAFE_GDN_RULE", True) and self.target_head_dim >= 256:
                self.gdn.forward = types.MethodType(_metax_safe_gdn_forward, self.gdn)
                print(
                    "Using MetaX-safe Q/K L2 normalization outside the "
                    f"GatedDeltaRule Triton kernel for head_dim={self.target_head_dim}."
                )
            self.random_initial_parameters = (
                {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in self.gdn.named_parameters()
                }
                if capture_random_initialization
                else {}
            )
            self._initialize_gdn_auxiliary_parameters(identity_short_conv)
        finally:
            torch.random.set_rng_state(cpu_rng_state)

        self.gdn.to(device=source_weights.device, dtype=source_weights.dtype)
        if hasattr(self.gdn, "A_log"):
            self.gdn.A_log.data = self.gdn.A_log.data.float()
        if hasattr(self.gdn, "dt_bias"):
            self.gdn.dt_bias.data = self.gdn.dt_bias.data.float()
        self.gdn.requires_grad_(False)

        # Wan2.1-1.3B blocks are 1536-wide, while the established Wan2.2 GDN is
        # 3072-wide.  Keep that GDN byte-for-byte shape-compatible and bridge
        # only at its boundary.  The repeat/average pair starts as a left
        # inverse: hidden_output_adapter @ hidden_input_adapter == I.
        hidden_input = torch.eye(
            dim, dtype=source_weights.dtype, device=source_weights.device
        ).repeat(self.hidden_repeat, 1)
        hidden_output = hidden_input.transpose(0, 1).contiguous() / self.hidden_repeat
        self.hidden_input_adapter = torch.nn.Parameter(hidden_input)
        self.hidden_output_adapter = torch.nn.Parameter(hidden_output)
        target_to_source_head = torch.arange(
            target_num_heads, dtype=torch.long, device=source_weights.device
        ) // self.head_repeat
        self.register_buffer(
            "target_to_source_head", target_to_source_head, persistent=False
        )

        self.mixing = torch.nn.Parameter(
            _manual_copy_matrix(mixing_init).to(device=source_weights.device)
        )
        if mix_granularity not in {"full", "head", "channel"}:
            raise ValueError(
                "mix_granularity must be 'full', 'head', or 'channel', "
                f"got {mix_granularity!r}"
            )
        if mix_order not in {"pre", "post"}:
            raise ValueError(f"mix_order must be 'pre' or 'post', got {mix_order!r}")
        # MAPPING_POSTMIX is the legacy spelling for post-headmix-base.
        if postmix and mix_order == "pre":
            mix_order = "post"
        if source_specific_expansion and mix_order != "post":
            raise ValueError("source_specific_expansion is defined only for post-mix")
        self.mix_granularity = mix_granularity
        self.mix_order = mix_order
        self.source_specific_expansion = bool(source_specific_expansion)
        self.postmix_enabled = self.mix_order == "post"
        # PreMix: first form one teacher-width HeadMix block per logical head,
        # then widen it through a dense A/B adapter.  PostMix: first widen every
        # source projection and then mix independently in every widened 128-dim
        # subblock.  The latter therefore has 2/4 alpha groups per GDN head.
        subhead_targets = (
            {
                "q": self.target_head_dim // self.source_head_dim,
                "k": self.target_head_dim // self.source_head_dim,
                "v": self.target_head_v_dim // self.source_head_dim,
                "o": self.target_head_v_dim // self.source_head_dim,
                "g": self.target_head_v_dim // self.source_head_dim,
            }
            if self.postmix_enabled
            else {"q": 1, "k": 1, "v": 1, "o": 1, "g": 1}
        )
        self.subhead_mixing = torch.nn.ParameterDict(
            {
                name: torch.nn.Parameter(
                    self.mixing.detach()[TARGET_NAMES.index(name)]
                    .view(1, 1, len(SOURCE_NAMES))
                    .expand(self.target_num_heads, count, len(SOURCE_NAMES))
                    .clone()
                )
                for name, count in subhead_targets.items()
            }
        )
        self.register_buffer(
            "mapping_subheadmix_enabled", torch.tensor(1, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "mapping_postmix_enabled",
            torch.tensor(int(self.postmix_enabled), dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "mapping_mix_granularity",
            torch.tensor(
                {"head": 0, "channel": 1, "full": 2}[self.mix_granularity],
                dtype=torch.int64,
            ),
            persistent=True,
        )
        self.register_buffer(
            "mapping_mix_order",
            torch.tensor(0 if self.mix_order == "pre" else 1, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "mapping_source_specific_expansion",
            torch.tensor(int(self.source_specific_expansion), dtype=torch.int64),
            persistent=True,
        )
        def row_expansion(repeat):
            value = torch.zeros(
                self.target_num_heads * repeat * self.source_head_dim, self.dim,
                dtype=source_weights.dtype, device=source_weights.device,
            )
            for head in range(self.target_num_heads):
                source_head = head // self.head_repeat
                for part in range(repeat):
                    row = (head * repeat + part) * self.source_head_dim
                    col = source_head * self.source_head_dim
                    value[row:row + self.source_head_dim, col:col + self.source_head_dim] = torch.eye(
                        self.source_head_dim, dtype=value.dtype, device=value.device
                    )
            return value
        def column_expansion(repeat):
            value = torch.zeros(
                self.dim, self.target_num_heads * repeat * self.source_head_dim,
                dtype=source_weights.dtype, device=source_weights.device,
            )
            total_repeat = self.head_repeat * repeat
            for head in range(self.target_num_heads):
                source_head = head // self.head_repeat
                for part in range(repeat):
                    row = source_head * self.source_head_dim
                    col = (head * repeat + part) * self.source_head_dim
                    value[row:row + self.source_head_dim, col:col + self.source_head_dim] = torch.eye(
                        self.source_head_dim, dtype=value.dtype, device=value.device
                    ) / total_repeat
            return value
        self.linear_adapter = torch.nn.ParameterDict(
            {
                "q": torch.nn.Parameter(row_expansion(self.target_head_dim // self.source_head_dim)),
                "k": torch.nn.Parameter(row_expansion(self.target_head_dim // self.source_head_dim)),
                "v": torch.nn.Parameter(row_expansion(self.target_head_v_dim // self.source_head_dim)),
                "g": torch.nn.Parameter(row_expansion(self.target_head_v_dim // self.source_head_dim)),
                "o": torch.nn.Parameter(column_expansion(self.target_head_v_dim // self.source_head_dim)),
            }
        )
        # ChannelMix assigns one source-mixture vector to every target channel.
        # For o, "channel" means its input-column channel, consistent with the
        # orientation of its multi-head value representation.
        channel_sizes = {
            "q": self.target_num_heads * self.target_head_dim if self.mix_order == "post" else self.dim,
            "k": self.target_num_heads * self.target_head_dim if self.mix_order == "post" else self.dim,
            "v": self.target_num_heads * self.target_head_v_dim if self.mix_order == "post" else self.dim,
            "g": self.target_num_heads * self.target_head_v_dim if self.mix_order == "post" else self.dim,
            "o": self.target_num_heads * self.target_head_v_dim if self.mix_order == "post" else self.dim,
        }
        self.channel_mixing = torch.nn.ParameterDict(
            {
                name: torch.nn.Parameter(
                    self.mixing.detach()[TARGET_NAMES.index(name)]
                    .view(1, len(SOURCE_NAMES))
                    .expand(size, len(SOURCE_NAMES))
                    .clone()
                )
                for name, size in channel_sizes.items()
            }
        )
        # PostMix can optionally give every (target, source) pair its own dense
        # expansion.  PreMix intentionally has only one target adapter because
        # mixing occurs before expansion.
        self.source_linear_adapter = torch.nn.ParameterDict()
        if self.source_specific_expansion:
            for target_name, adapter in self.linear_adapter.items():
                for source_name in SOURCE_NAMES:
                    self.source_linear_adapter[f"{target_name}__{source_name}"] = torch.nn.Parameter(
                        adapter.detach().clone()
                    )
        self.headwise_mixing = torch.nn.Parameter(
            self.mixing.detach()[TARGET_NAMES.index("beta")]
            .view(1, len(SOURCE_NAMES))
            .expand(self.target_num_heads, len(SOURCE_NAMES))
            .clone()
        )
        self.map_auxiliary_parameters = bool(map_auxiliary_parameters)
        # Auxiliary parameters do not all expose the same axes.  Tie their
        # source-routing granularity to the main mapping: FullMix has one
        # vector/target; HeadMix uses a vector/head; ChannelMix uses a vector
        # per real auxiliary channel.  A_log/dt_bias only have a head axis, so
        # their natural finest representation remains per-head.
        self.auxiliary_mix_granularity = mix_granularity
        self.register_buffer(
            "mapping_auxiliary_enabled",
            torch.tensor(int(self.map_auxiliary_parameters), dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "mapping_auxiliary_mix_granularity",
            torch.tensor({"head": 0, "channel": 1, "full": 2}[mix_granularity], dtype=torch.int64),
            persistent=True,
        )
        if not getattr(self.gdn, "use_short_conv", False):
            raise ValueError("Full GDN mapping requires q/k/v short convolutions")
        conv_size = int(self.gdn.q_conv1d.weight.shape[-1])
        self.auxiliary_mixing = torch.nn.ParameterDict(
            {
                "a": torch.nn.Parameter(torch.zeros(len(SOURCE_NAMES))),
                "A_log": torch.nn.Parameter(torch.zeros(len(SOURCE_NAMES))),
                "dt_bias": torch.nn.Parameter(torch.zeros(len(SOURCE_NAMES))),
                "q_conv1d": torch.nn.Parameter(
                    torch.zeros(conv_size, len(SOURCE_NAMES))
                ),
                "k_conv1d": torch.nn.Parameter(
                    torch.zeros(conv_size, len(SOURCE_NAMES))
                ),
                "v_conv1d": torch.nn.Parameter(
                    torch.zeros(conv_size, len(SOURCE_NAMES))
                ),
                "o_norm": torch.nn.Parameter(torch.zeros(len(SOURCE_NAMES))),
            }
        )
        self.auxiliary_mixing.to(device=source_weights.device)
        self.auxiliary_headwise_mixing = torch.nn.ParameterDict(
            {
                "a": torch.nn.Parameter(torch.zeros(self.target_num_heads, len(SOURCE_NAMES))),
                "A_log": torch.nn.Parameter(torch.zeros(self.target_num_heads, len(SOURCE_NAMES))),
                "dt_bias": torch.nn.Parameter(torch.zeros(self.target_num_heads, len(SOURCE_NAMES))),
                "q_conv1d": torch.nn.Parameter(torch.zeros(conv_size, self.target_num_heads, len(SOURCE_NAMES))),
                "k_conv1d": torch.nn.Parameter(torch.zeros(conv_size, self.target_num_heads, len(SOURCE_NAMES))),
                "v_conv1d": torch.nn.Parameter(torch.zeros(conv_size, self.target_num_heads, len(SOURCE_NAMES))),
            }
        )
        self.auxiliary_headwise_mixing.to(device=source_weights.device)
        self.auxiliary_full_mixing = torch.nn.ParameterDict(
            {
                name: torch.nn.Parameter(torch.zeros(len(SOURCE_NAMES)))
                for name in AUXILIARY_TARGET_PARAMETER_NAMES
            }
        )
        self.auxiliary_full_mixing.to(device=source_weights.device)
        self.auxiliary_channel_mixing = torch.nn.ParameterDict(
            {
                # a_proj is [head, input-channel]; route its input channels.
                "a": torch.nn.Parameter(torch.zeros(self.dim, len(SOURCE_NAMES))),
                # These vectors have one scalar per logical head.
                "A_log": torch.nn.Parameter(torch.zeros(self.target_num_heads, len(SOURCE_NAMES))),
                "dt_bias": torch.nn.Parameter(torch.zeros(self.target_num_heads, len(SOURCE_NAMES))),
                # [kernel tap, teacher-width output channel, source].
                "q_conv1d": torch.nn.Parameter(torch.zeros(conv_size, self.dim, len(SOURCE_NAMES))),
                "k_conv1d": torch.nn.Parameter(torch.zeros(conv_size, self.dim, len(SOURCE_NAMES))),
                "v_conv1d": torch.nn.Parameter(torch.zeros(conv_size, self.dim, len(SOURCE_NAMES))),
                # o_norm is shared across heads; its real channel width is 128
                # before the existing repeat-to-256 dimension adapter.
                "o_norm": torch.nn.Parameter(torch.zeros(self.source_head_dim, len(SOURCE_NAMES))),
            }
        )
        self.auxiliary_channel_mixing.to(device=source_weights.device)
        auxiliary_trainable = self.map_auxiliary_parameters
        self.auxiliary_mixing.requires_grad_(
            auxiliary_trainable and self.auxiliary_mix_granularity == "head"
        )
        self.auxiliary_headwise_mixing.requires_grad_(
            auxiliary_trainable and self.auxiliary_mix_granularity == "head"
        )
        self.auxiliary_full_mixing.requires_grad_(
            auxiliary_trainable and self.auxiliary_mix_granularity == "full"
        )
        self.auxiliary_channel_mixing.requires_grad_(
            auxiliary_trainable and self.auxiliary_mix_granularity == "channel"
        )
        self.low_rank = int(low_rank)
        self.full_residual_enabled = bool(full_residual)
        self.post_full_residual_enabled = bool(post_full_residual)
        if self.full_residual_enabled and self.post_full_residual_enabled:
            raise ValueError("narrow full_residual and post_full_residual are mutually exclusive")
        if self.postmix_enabled and self.full_residual_enabled:
            raise ValueError("PostMix requires a native post_full_residual, not narrow full_residual")
        if self.full_residual_enabled and self.low_rank:
            raise ValueError("SubheadMix full residual is mutually exclusive with low_rank")
        if self.low_rank < 0:
            raise ValueError(f"low_rank must be non-negative, got {self.low_rank}")
        self.register_buffer(
            "mapping_low_rank",
            torch.tensor(self.low_rank, dtype=torch.int64),
            persistent=True,
        )
        self.low_rank_scale = 1.0
        self.low_rank_u = torch.nn.ParameterDict()
        self.low_rank_v = torch.nn.ParameterDict()
        if self.low_rank > 0:
            gdn_parameters = dict(self.gdn.named_parameters())
            target_shapes = {
                name: tuple(gdn_parameters[parameter_name].shape)
                for name, parameter_name in TARGET_PARAMETER_NAMES.items()
            }
            cpu_rng_state = torch.random.get_rng_state()
            torch.manual_seed(seed + 1)
            try:
                for name, (out_features, in_features) in target_shapes.items():
                    u = torch.empty(
                        out_features,
                        self.low_rank,
                        dtype=torch.float32,
                    )
                    v = torch.zeros(
                        in_features,
                        self.low_rank,
                        dtype=torch.float32,
                    )
                    torch.nn.init.normal_(u, mean=0.0, std=low_rank_init_std)
                    self.low_rank_u[name] = torch.nn.Parameter(
                        u.to(device=source_weights.device, dtype=source_weights.dtype)
                    )
                    self.low_rank_v[name] = torch.nn.Parameter(
                        v.to(device=source_weights.device, dtype=source_weights.dtype)
                    )
            finally:
                torch.random.set_rng_state(cpu_rng_state)
        self.register_buffer(
            "mapping_full_residual_enabled",
            torch.tensor(int(self.full_residual_enabled), dtype=torch.int64),
            persistent=True,
        )
        self.full_residual = torch.nn.ParameterDict()
        if self.full_residual_enabled:
            narrow_shapes = {
                "q": (self.dim, self.dim), "k": (self.dim, self.dim),
                "v": (self.dim, self.dim), "o": (self.dim, self.dim),
                "g": (self.dim, self.dim), "beta": (self.num_heads, self.dim),
            }
            for name, shape in narrow_shapes.items():
                self.full_residual[name] = torch.nn.Parameter(
                    torch.zeros(shape, dtype=source_weights.dtype,
                                device=source_weights.device)
                )
        self.register_buffer(
            "mapping_post_full_residual_enabled",
            torch.tensor(int(self.post_full_residual_enabled), dtype=torch.int64),
            persistent=True,
        )
        # These residuals live in the native, widened GDN parameter spaces and
        # are deliberately added *after* A/B linear expansion.
        self.post_full_residual = torch.nn.ParameterDict()
        if self.post_full_residual_enabled:
            for name, parameter_name in TARGET_PARAMETER_NAMES.items():
                self.post_full_residual[name] = torch.nn.Parameter(
                    torch.zeros(
                        tuple(dict(self.gdn.named_parameters())[parameter_name].shape),
                        dtype=source_weights.dtype, device=source_weights.device,
                    )
                )

        self.last_output = None
        self._validate_target_shapes()

    def share_mapping_parameters_from(self, other):
        if self.low_rank != other.low_rank:
            raise ValueError(
                f"Cannot share mapping parameters across low-rank {self.low_rank} "
                f"and {other.low_rank} transforms"
            )
        if self.map_auxiliary_parameters != other.map_auxiliary_parameters:
            raise ValueError("Auxiliary mapping settings must match before sharing")
        self.mixing = other.mixing
        for name in self.subhead_mixing:
            self.subhead_mixing[name] = other.subhead_mixing[name]
        for name in self.channel_mixing:
            self.channel_mixing[name] = other.channel_mixing[name]
        self.headwise_mixing = other.headwise_mixing
        for name in self.auxiliary_mixing:
            self.auxiliary_mixing[name] = other.auxiliary_mixing[name]
        for name in self.auxiliary_headwise_mixing:
            self.auxiliary_headwise_mixing[name] = other.auxiliary_headwise_mixing[name]
        for name in self.auxiliary_full_mixing:
            self.auxiliary_full_mixing[name] = other.auxiliary_full_mixing[name]
        for name in self.auxiliary_channel_mixing:
            self.auxiliary_channel_mixing[name] = other.auxiliary_channel_mixing[name]
        for name in self.low_rank_u:
            self.low_rank_u[name] = other.low_rank_u[name]
            self.low_rank_v[name] = other.low_rank_v[name]
        for name in self.full_residual:
            self.full_residual[name] = other.full_residual[name]
        for name in self.post_full_residual:
            self.post_full_residual[name] = other.post_full_residual[name]
        for name in self.linear_adapter:
            self.linear_adapter[name] = other.linear_adapter[name]
        for name in self.source_linear_adapter:
            self.source_linear_adapter[name] = other.source_linear_adapter[name]
        self.hidden_input_adapter = other.hidden_input_adapter
        self.hidden_output_adapter = other.hidden_output_adapter
        # Every block supplies its own materialized parameters through
        # functional_call.  The registered GDN here is therefore only a frozen
        # compute/shape template.  Sharing it is exact (past_key_values=None,
        # use_cache=False) and avoids retaining 30 identical 3072-wide GDNs.
        self.gdn = other.gdn

    def _initialize_gdn_auxiliary_parameters(self, identity_short_conv):
        with torch.no_grad():
            if hasattr(self.gdn, "a_proj"):
                self.gdn.a_proj.weight.zero_()
            if identity_short_conv and getattr(self.gdn, "use_short_conv", False):
                for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                    convolution = getattr(self.gdn, name)
                    convolution.weight.zero_()
                    convolution.weight[..., -1] = 1.0
                    if convolution.bias is not None:
                        convolution.bias.zero_()

    def _repeat_rows_per_head(self, weight, target_head_dim):
        repeat = target_head_dim // self.source_head_dim
        source = weight.reshape(
            self.source_num_heads, self.source_head_dim, weight.shape[-1]
        ).index_select(0, self.target_to_source_head)
        return source.repeat(1, repeat, 1).reshape(
            self.target_num_heads * target_head_dim, weight.shape[-1]
        )

    def _repeat_columns_per_head(self, weight, target_head_dim):
        repeat = target_head_dim // self.source_head_dim
        widened = weight.reshape(
            weight.shape[0], self.source_num_heads, self.source_head_dim
        ).index_select(1, self.target_to_source_head)
        widened = widened.repeat(1, 1, repeat).reshape(
            weight.shape[0], self.target_num_heads * target_head_dim
        )
        # Repeated value channels carry identical activations. Averaging their
        # output columns preserves the original projection at initialization.
        return widened / (self.head_repeat * repeat)

    def _repeat_channels_per_head(self, values, target_head_dim):
        repeat = target_head_dim // self.source_head_dim
        source = values.reshape(self.source_num_heads, self.source_head_dim)
        source = source.index_select(0, self.target_to_source_head)
        return source.repeat(1, repeat).reshape(
            self.target_num_heads * target_head_dim
        )

    def _source_heads_for_target(self, values):
        """Duplicate a source-head axis (axis 1) onto the exact GDN heads."""
        return values.index_select(1, self.target_to_source_head)

    def _expand_hidden_input_axis(self, weight):
        """Map a [..., 1536] input axis to the GDN's exact 3072 input axis."""
        return weight @ self.hidden_output_adapter.to(
            dtype=weight.dtype, device=weight.device
        )

    def _expand_hidden_output_axis(self, weight):
        """Map a [1536, ...] output axis to the GDN's exact 3072 output axis."""
        return self.hidden_input_adapter.to(
            dtype=weight.dtype, device=weight.device
        ) @ weight

    def _subhead_mixed_weight(self, target_name):
        alpha = self.subhead_mixing[target_name].to(
            dtype=self.source_weights.dtype, device=self.source_weights.device
        )
        source = self.source_weights.reshape(
            len(SOURCE_NAMES), self.source_num_heads, self.source_head_dim, self.dim
        )
        source = self._source_heads_for_target(source)
        if target_name == "o":
            # Output projection head channels live on its input-column axis.
            source = source.permute(0, 3, 1, 2)
            return torch.einsum("hrs,sdhi->dhri", alpha, source).reshape(
                self.dim, -1
            )
        return torch.einsum("hrs,shdi->hrdi", alpha, source).reshape(-1, self.dim)

    def _postmix_weight(self, target_name):
        """Expand every source with A/B, then HeadMix in widened subhead space."""
        alpha = self.subhead_mixing[target_name].to(
            dtype=self.source_weights.dtype, device=self.source_weights.device
        )
        expanded = self._expanded_source_weights(target_name)
        if target_name == "o":
            # Each source is expanded on o's input-column axis before its
            # widened (head, subblock) coefficients are applied.
            expanded = expanded.reshape(
                len(SOURCE_NAMES), self.dim, self.target_num_heads,
                alpha.shape[1], self.source_head_dim,
            )
            return torch.einsum("hrs,sdhri->dhri", alpha, expanded).reshape(
                self.dim, -1
            )
        expanded = expanded.reshape(
            len(SOURCE_NAMES), self.target_num_heads, alpha.shape[1],
            self.source_head_dim, self.dim,
        )
        return torch.einsum("hrs,shrdi->hrdi", alpha, expanded).reshape(-1, self.dim)

    def _expanded_source_weights(self, target_name):
        """Apply the shared or source-specific dense expansion to all sources."""
        if target_name == "o":
            if self.source_specific_expansion:
                return torch.stack(
                    [
                        self.source_weights[source_index]
                        @ self.source_linear_adapter[f"o__{source_name}"].to(
                            dtype=self.source_weights.dtype,
                            device=self.source_weights.device,
                        )
                        for source_index, source_name in enumerate(SOURCE_NAMES)
                    ],
                    dim=0,
                )
            return self.source_weights @ self.linear_adapter["o"].to(
                dtype=self.source_weights.dtype, device=self.source_weights.device
            )
        if self.source_specific_expansion:
            return torch.stack(
                [
                    self.source_linear_adapter[f"{target_name}__{source_name}"].to(
                        dtype=self.source_weights.dtype,
                        device=self.source_weights.device,
                    )
                    @ self.source_weights[source_index]
                    for source_index, source_name in enumerate(SOURCE_NAMES)
                ],
                dim=0,
            )
        adapter = self.linear_adapter[target_name].to(
            dtype=self.source_weights.dtype, device=self.source_weights.device
        )
        return adapter.unsqueeze(0) @ self.source_weights

    def _channel_mixed_weight(self, target_name):
        """Channel-wise four-source mixing before or after dense expansion."""
        source = (
            self._expanded_source_weights(target_name)
            if self.mix_order == "post"
            else self.source_weights
        )
        alpha = self.channel_mixing[target_name].to(
            dtype=source.dtype, device=source.device
        )
        if target_name == "o":
            return torch.einsum("js,sdj->dj", alpha, source)
        return torch.einsum("is,sij->ij", alpha, source)

    def _full_mixed_weight(self, target_index, target_name):
        """One four-source coefficient vector per target projection.

        In pre order this is the historical FullMix in teacher width followed
        by the shared target expansion.  In post order it first expands each
        source with that same shared target adapter, then applies the scalar
        four-source mix in native GDN width.  Thus it is the controlled
        FullMix counterpart of preHeadmix/postHeadmix_base.
        """
        source = (
            self._expanded_source_weights(target_name)
            if self.mix_order == "post"
            else self.source_weights
        )
        coefficients = self.mixing[target_index].to(
            dtype=source.dtype, device=source.device
        )
        return torch.tensordot(coefficients, source, dims=([-1], [0]))

    def _validate_target_shapes(self):
        parameters = dict(self.gdn.named_parameters())
        for target_name, parameter_name in TARGET_PARAMETER_NAMES.items():
            if parameter_name not in parameters:
                raise ValueError(
                    f"GDN does not expose required parameter {parameter_name!r}"
                )
            actual = tuple(parameters[parameter_name].shape)
            expected = {
                "q": (self.target_num_heads * self.target_head_dim, self.target_dim),
                "k": (self.target_num_heads * self.target_head_dim, self.target_dim),
                "v": (self.target_num_heads * self.target_head_v_dim, self.target_dim),
                "g": (self.target_num_heads * self.target_head_v_dim, self.target_dim),
                "o": (self.target_dim, self.target_num_heads * self.target_head_v_dim),
                "beta": (self.target_num_heads, self.target_dim),
            }[target_name]
            if actual != expected:
                raise ValueError(
                    f"Cannot map {target_name}: GDN {parameter_name} has shape {actual}, expected {expected}"
                )
        auxiliary_expected_shapes = {
            "a": (self.target_num_heads, self.target_dim),
            "A_log": (self.target_num_heads,),
            "dt_bias": (self.target_num_heads,),
            "q_conv1d": (
                self.target_num_heads * self.target_head_dim,
                1,
                self.auxiliary_mixing["q_conv1d"].shape[0],
            ),
            "k_conv1d": (
                self.target_num_heads * self.target_head_dim,
                1,
                self.auxiliary_mixing["k_conv1d"].shape[0],
            ),
            "v_conv1d": (
                self.target_num_heads * self.target_head_v_dim,
                1,
                self.auxiliary_mixing["v_conv1d"].shape[0],
            ),
            "o_norm": (self.target_head_v_dim,),
        }
        for target_name, parameter_name in AUXILIARY_TARGET_PARAMETER_NAMES.items():
            if parameter_name not in parameters:
                raise ValueError(
                    f"GDN does not expose required parameter {parameter_name!r}"
                )
            actual = tuple(parameters[parameter_name].shape)
            expected = tuple(auxiliary_expected_shapes[target_name])
            if actual != expected:
                raise ValueError(
                    f"Cannot map {target_name}: GDN {parameter_name} has shape {actual}, expected {expected}"
                )

    def _low_rank_residual(self, target_name):
        if self.low_rank <= 0:
            return None
        return self.low_rank_u[target_name] @ self.low_rank_v[target_name].transpose(
            0, 1
        )

    @staticmethod
    def _mixed_weight(coefficients, source, target):
        source = source.to(device=target.device, dtype=target.dtype)
        coefficients = coefficients.to(device=target.device, dtype=target.dtype)
        return torch.tensordot(coefficients, source, dims=([-1], [0]))

    def _materialized_auxiliary_parameters(self):
        parameters = dict(self.gdn.named_parameters())
        mapped = {}

        source_by_head = self._source_heads_for_target(self.source_weights_by_head)
        head_statistics = self._source_heads_for_target(self.source_head_statistics)
        source_channels = self.source_channel_statistics.reshape(
            len(SOURCE_NAMES), self.source_num_heads, self.source_head_dim
        )
        source_channels = self._source_heads_for_target(source_channels)

        if self.auxiliary_mix_granularity == "full":
            a_delta = torch.einsum(
                "s,shd->hd",
                self.auxiliary_full_mixing["a"].to(source_by_head.dtype),
                source_by_head,
            )
        elif self.auxiliary_mix_granularity == "channel":
            a_delta = torch.einsum(
                "ds,shd->hd",
                self.auxiliary_channel_mixing["a"].to(source_by_head.dtype),
                source_by_head,
            )
        else:
            a_delta = torch.einsum(
                "hs,shd->hd",
                (
                    self.auxiliary_mixing["a"].view(1, -1)
                    + self.auxiliary_headwise_mixing["a"]
                ).to(source_by_head.dtype),
                source_by_head,
            )
        a_base = parameters[AUXILIARY_TARGET_PARAMETER_NAMES["a"]]
        mapped[AUXILIARY_TARGET_PARAMETER_NAMES["a"]] = a_base + self._expand_hidden_input_axis(
            a_delta
        ).to(a_base.dtype)

        for target_name in ("A_log", "dt_bias"):
            parameter_name = AUXILIARY_TARGET_PARAMETER_NAMES[target_name]
            base = parameters[parameter_name]
            if self.auxiliary_mix_granularity == "full":
                delta = torch.einsum(
                    "s,sh->h",
                    self.auxiliary_full_mixing[target_name].to(head_statistics.dtype),
                    head_statistics,
                )
            elif self.auxiliary_mix_granularity == "channel":
                delta = torch.einsum(
                    "hs,sh->h",
                    self.auxiliary_channel_mixing[target_name].to(head_statistics.dtype),
                    head_statistics,
                )
            else:
                delta = torch.einsum(
                    "hs,sh->h",
                    (
                        self.auxiliary_mixing[target_name].view(1, -1)
                        + self.auxiliary_headwise_mixing[target_name]
                    ).to(head_statistics.dtype),
                    head_statistics,
                )
            mapped[parameter_name] = base + delta.to(base.dtype)

        for target_name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            parameter_name = AUXILIARY_TARGET_PARAMETER_NAMES[target_name]
            base = parameters[parameter_name]
            if self.auxiliary_mix_granularity == "full":
                channel_values = torch.einsum(
                    "s,shd->hd",
                    self.auxiliary_full_mixing[target_name].to(source_channels.dtype),
                    source_channels,
                ).unsqueeze(0).expand(base.shape[-1], -1, -1)
            elif self.auxiliary_mix_granularity == "channel":
                alpha = self.auxiliary_channel_mixing[target_name].reshape(
                    base.shape[-1], self.source_num_heads, self.source_head_dim,
                    len(SOURCE_NAMES),
                ).index_select(1, self.target_to_source_head).to(source_channels.dtype)
                channel_values = torch.einsum("khds,shd->khd", alpha, source_channels)
            else:
                channel_values = torch.einsum(
                    "khs,shd->khd",
                    self.auxiliary_mixing[target_name].unsqueeze(1)
                    + self.auxiliary_headwise_mixing[target_name],
                    source_channels,
                )
            target_head_dim = (
                self.target_head_v_dim
                if target_name == "v_conv1d"
                else self.target_head_dim
            )
            repeat = target_head_dim // self.source_head_dim
            channel_values = channel_values.repeat(1, 1, repeat)
            residual = channel_values.permute(1, 2, 0).reshape(
                self.target_num_heads * target_head_dim, 1, -1
            )
            mapped[parameter_name] = base + residual.to(base.dtype)

        norm_name = AUXILIARY_TARGET_PARAMETER_NAMES["o_norm"]
        norm_base = parameters[norm_name]
        if self.auxiliary_mix_granularity == "full":
            norm_residual = torch.einsum(
                "s,sd->d",
                self.auxiliary_full_mixing["o_norm"].to(self.source_norm_statistics.dtype),
                self.source_norm_statistics,
            )
        elif self.auxiliary_mix_granularity == "channel":
            norm_residual = torch.einsum(
                "ds,sd->d",
                self.auxiliary_channel_mixing["o_norm"].to(self.source_norm_statistics.dtype),
                self.source_norm_statistics,
            )
        else:
            norm_residual = self._mixed_weight(
                self.auxiliary_mixing["o_norm"], self.source_norm_statistics, norm_base
            )
        norm_residual = norm_residual.repeat(
            self.target_head_v_dim // self.source_head_dim
        )
        mapped[norm_name] = norm_base + norm_residual.to(norm_base.dtype)
        return mapped

    def materialized_parameters(self):
        mapped = {}
        for target_index, target_name in enumerate(TARGET_NAMES):
            source = (
                self._source_heads_for_target(self.source_weights_by_head)
                if target_name == "beta"
                else self.source_weights
            )
            coefficients = self.mixing[target_index].to(
                dtype=source.dtype, device=source.device
            )
            weight = (
                (
                    torch.einsum(
                        "s,shd->hd", coefficients, source,
                    )
                    if self.mix_granularity == "full"
                    else torch.einsum(
                        "hs,shd->hd",
                        self.headwise_mixing.to(dtype=source.dtype, device=source.device),
                        source,
                    )
                )
                if target_name == "beta"
                else (
                    self._full_mixed_weight(target_index, target_name)
                    if self.mix_granularity == "full"
                    else (
                        self._channel_mixed_weight(target_name)
                        if self.mix_granularity == "channel"
                        else (
                        self._postmix_weight(target_name)
                        if self.postmix_enabled
                        else self._subhead_mixed_weight(target_name)
                        )
                    )
                )
            )
            if self.full_residual_enabled:
                weight = weight + self.full_residual[target_name].to(weight.dtype)
            if target_name in self.linear_adapter and self.mix_order == "pre":
                weight = (
                    self.linear_adapter[target_name] @ weight
                    if target_name != "o"
                    else weight @ self.linear_adapter[target_name]
                )
            # The exact Wan2.2 GDN remains 3072-wide.  These two adapters map
            # only the hidden input/output axes that do not exist in the
            # 1536-wide Wan2.1 teacher matrices.
            weight = (
                self._expand_hidden_output_axis(weight)
                if target_name == "o"
                else self._expand_hidden_input_axis(weight)
            )
            if self.post_full_residual_enabled:
                weight = weight + self.post_full_residual[target_name].to(weight.dtype)
            residual = self._low_rank_residual(target_name)
            if residual is not None:
                weight = weight + residual.to(weight.dtype) * self.low_rank_scale
            mapped[TARGET_PARAMETER_NAMES[target_name]] = weight
        if self.map_auxiliary_parameters:
            mapped.update(self._materialized_auxiliary_parameters())
        return mapped


    def materialized_projection_weights(self):
        """Backward-compatible alias; format v2 can also return auxiliary parameters."""
        return self.materialized_parameters()

    def forward(self, x, freqs=None, mapped_parameters=None):
        del freqs
        if mapped_parameters is None:
            mapped_parameters = self.materialized_parameters()
        try:
            from torch.func import functional_call
        except ImportError:
            from torch.nn.utils.stateless import functional_call

        projected_x = F.linear(
            x,
            self.hidden_input_adapter.to(dtype=x.dtype, device=x.device),
        )
        output = functional_call(
            self.gdn,
            mapped_parameters,
            (projected_x,),
            strict=False,
        )
        if isinstance(output, tuple):
            output = output[0]
        output = F.linear(
            output,
            self.hidden_output_adapter.to(dtype=output.dtype, device=output.device),
        )
        # The caller receives the output directly. Keeping a second module
        # reference is unnecessary and would retain the recomputed long-video
        # tensor after activation-checkpoint backward.
        self.last_output = None
        return output.to(x.dtype)


class _StopAfterSelectedBlock(RuntimeError):
    pass


class WanSingleLayerWeightMappingModule(BaseWanTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        audio_processor_path=None,
        trainable_models=None,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="weight_mapping",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        student_init_ckpt=None,
        block_index=-1,
        gdn_block_indices="",
        mapping_num_layers=1,
        middle_layer_start=10,
        middle_layer_end=19,
        mapping_seed=42,
        mixing_init="manual_copy",
        low_rank=0,
        full_residual=False,
        post_full_residual=False,
        postmix=False,
        mix_granularity="head",
        mix_order="pre",
        source_specific_expansion=False,
        low_rank_init_std=0.02,
        gdn_expand_v=1.0,
        gdn_head_dim=0,
        gdn_hidden_size=0,
        gdn_num_heads=0,
        shape_adapter="subheadmix",
        gdn_use_short_conv=True,
        identity_short_conv=True,
        map_auxiliary_parameters=False,
        train_base_mapping=True,
        capture_random_initialization=False,
        attention_loss_weight=1.0,
        block_loss_weight=1.0,
        mapping_regularization=0.0,
        normalized_mse=False,
        mapping_log_interval=100,
        mapping_blocks_per_step=0,
        checkpoint_mapped_block=False,
        mapping_step_offset=0,
        **unused_kwargs,
    ):
        del audio_processor_path, task, unused_kwargs
        DiffusionTrainingModule.__init__(self)
        if trainable_models and trainable_models.strip() != "mapped_attention":
            raise ValueError(
                "Only mapped_attention may be trainable in this experiment, got "
                f"{trainable_models!r}"
            )
        if attention_loss_weight <= 0 and block_loss_weight <= 0:
            raise ValueError(
                "At least one of attention_loss_weight or block_loss_weight must be positive"
            )

        if model_paths is None and model_id_with_origin_paths is None:
            raise ValueError("Pass --model_paths or --model_id_with_origin_paths")
        model_configs = self.parse_model_configs(
            model_paths,
            model_id_with_origin_paths,
            fp8_models=fp8_models,
            offload_models=offload_models,
            device=device,
        )
        tokenizer_config = (
            ModelConfig(path=tokenizer_path) if tokenizer_path is not None else None
        )
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )
        if self.pipe.dit is None:
            raise RuntimeError("WanVideoPipeline did not load a DiT model")
        if self.pipe.dit2 is not None:
            print("Warning: pipe.dit2 is present; this MVP aligns only pipe.dit.")

        if os.environ.get("DIFFSYNTH_STANDARD_WAN_TEACHER", "0") == "1":
            teacher_class = (
                f"{self.pipe.dit.__class__.__module__}."
                f"{self.pipe.dit.__class__.__name__}"
            )
            if teacher_class != "diffsynth.models.wan_video_dit.WanModel":
                raise RuntimeError(
                    "Standard Wan teacher was requested but the loader returned "
                    f"{teacher_class}"
                )
            forbidden_memory_parameters = (
                "memory_attn",
                "mem_scale",
                "norm_mem",
            )
            teacher_memory_parameters = [
                name
                for name, _ in self.pipe.dit.named_parameters()
                if any(part in name for part in forbidden_memory_parameters)
            ]
            if teacher_memory_parameters:
                raise RuntimeError(
                    "Mapping teacher contains forbidden memory parameters: "
                    f"{teacher_memory_parameters[:8]}"
                )
            print(
                "Mapping teacher verified: standard Softmax WanModel "
                f"(hidden={self.pipe.dit.dim}, "
                f"heads={self.pipe.dit.blocks[0].num_heads}, "
                f"blocks={len(self.pipe.dit.blocks)})"
            )

        self.pipe.eval()
        self.pipe.requires_grad_(False)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.block_indices = _parse_mapping_block_indices(
            block_index=block_index,
            block_indices=gdn_block_indices,
            num_layers=len(self.pipe.dit.blocks),
            num_mapping_layers=int(mapping_num_layers),
            seed=mapping_seed,
            middle_start=middle_layer_start,
            middle_end=middle_layer_end,
        )
        self.block_index = self.block_indices[0]
        teacher_attention = self.pipe.dit.blocks[self.block_index].self_attn
        self.mapped_attention = MappedGDNWeightTransform(
            source_attention=teacher_attention,
            block_index=self.block_index,
            training_block_indices=self.block_indices,
            low_rank=low_rank,
            full_residual=full_residual,
            post_full_residual=post_full_residual,
            postmix=postmix,
            mix_granularity=mix_granularity,
            mix_order=mix_order,
            source_specific_expansion=source_specific_expansion,
            mixing_init=mixing_init,
            low_rank_init_std=low_rank_init_std,
            gdn_expand_v=gdn_expand_v,
            gdn_head_dim=gdn_head_dim,
            gdn_hidden_size=gdn_hidden_size,
            gdn_num_heads=gdn_num_heads,
            shape_adapter=shape_adapter,
            gdn_use_short_conv=gdn_use_short_conv,
            identity_short_conv=identity_short_conv,
            map_auxiliary_parameters=map_auxiliary_parameters,
            capture_random_initialization=capture_random_initialization,
            seed=mapping_seed,
        )
        self.additional_mapped_attentions = torch.nn.ModuleDict()
        for layer_index in self.block_indices[1:]:
            layer_mapper = MappedGDNWeightTransform(
                source_attention=self.pipe.dit.blocks[layer_index].self_attn,
                block_index=layer_index,
                training_block_indices=self.block_indices,
                low_rank=low_rank,
                full_residual=full_residual,
                post_full_residual=post_full_residual,
                postmix=postmix,
                mix_granularity=mix_granularity,
                mix_order=mix_order,
                source_specific_expansion=source_specific_expansion,
                mixing_init=mixing_init,
                low_rank_init_std=low_rank_init_std,
                gdn_expand_v=gdn_expand_v,
                gdn_head_dim=gdn_head_dim,
                gdn_hidden_size=gdn_hidden_size,
                gdn_num_heads=gdn_num_heads,
                shape_adapter=shape_adapter,
                gdn_use_short_conv=gdn_use_short_conv,
                identity_short_conv=identity_short_conv,
                map_auxiliary_parameters=map_auxiliary_parameters,
                capture_random_initialization=False,
                seed=mapping_seed,
            )
            layer_mapper.share_mapping_parameters_from(self.mapped_attention)
            self.additional_mapped_attentions[str(layer_index)] = layer_mapper
        self.mapped_attention.gdn.requires_grad_(False)
        for layer_mapper in self.additional_mapped_attentions.values():
            layer_mapper.gdn.requires_grad_(False)
        gdn_template_ids = {
            id(self.mapped_attention.gdn),
            *(id(mapper.gdn) for mapper in self.additional_mapped_attentions.values()),
        }
        if len(gdn_template_ids) != 1:
            raise RuntimeError(
                "All block mappers must share one frozen GDN compute template"
            )
        print(
            "GDN template audit: 1 shared frozen template for "
            f"{len(self.block_indices)} block mappers"
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if os.environ.get("DIFFSYNTH_STANDARD_WAN_TEACHER", "0") == "1":
            forbidden_memory_parameters = (
                "memory_attn",
                "mem_scale",
                "norm_mem",
            )
            model_memory_parameters = [
                name
                for name, _ in self.named_parameters()
                if any(part in name for part in forbidden_memory_parameters)
            ]
            if model_memory_parameters:
                raise RuntimeError(
                    "Weight-mapping model contains forbidden memory parameters: "
                    f"{model_memory_parameters[:8]}"
                )
            print("Memory-parameter audit: 0 forbidden parameters")
        self.train_base_mapping = bool(train_base_mapping)
        if not self.train_base_mapping and low_rank <= 0 and not full_residual and not post_full_residual:
            raise ValueError(
                "--train_base_mapping false requires --low_rank, --full_residual, "
                "or --post_full_residual"
            )
        self.mapped_attention.mixing.requires_grad_(
            self.train_base_mapping and mix_granularity == "full"
        )
        self.mapped_attention.subhead_mixing.requires_grad_(
            self.train_base_mapping and mix_granularity == "head"
        )
        self.mapped_attention.channel_mixing.requires_grad_(
            self.train_base_mapping and mix_granularity == "channel"
        )
        self.mapped_attention.headwise_mixing.requires_grad_(
            self.train_base_mapping and mix_granularity != "full"
        )
        self.mapped_attention.full_residual.requires_grad_(False)
        self.mapped_attention.post_full_residual.requires_grad_(bool(post_full_residual))
        self.mapped_attention.linear_adapter.requires_grad_(
            self.train_base_mapping and not source_specific_expansion
        )
        self.mapped_attention.source_linear_adapter.requires_grad_(
            self.train_base_mapping and bool(source_specific_expansion)
        )
        self.mapped_attention.hidden_input_adapter.requires_grad_(
            self.train_base_mapping
        )
        self.mapped_attention.hidden_output_adapter.requires_grad_(
            self.train_base_mapping
        )
        self.mapped_attention.auxiliary_mixing.requires_grad_(
            bool(map_auxiliary_parameters) and self.train_base_mapping and mix_granularity == "head"
        )
        self.mapped_attention.auxiliary_headwise_mixing.requires_grad_(
            bool(map_auxiliary_parameters) and self.train_base_mapping and mix_granularity == "head"
        )
        self.mapped_attention.auxiliary_full_mixing.requires_grad_(
            bool(map_auxiliary_parameters) and self.train_base_mapping and mix_granularity == "full"
        )
        self.mapped_attention.auxiliary_channel_mixing.requires_grad_(
            bool(map_auxiliary_parameters) and self.train_base_mapping and mix_granularity == "channel"
        )
        for parameter in self.mapped_attention.low_rank_u.parameters():
            parameter.requires_grad_(True)
        for parameter in self.mapped_attention.low_rank_v.parameters():
            parameter.requires_grad_(True)

        if student_init_ckpt:
            self._load_mapping_checkpoint(student_init_ckpt)

        self.use_gradient_checkpointing = False
        self.use_gradient_checkpointing_offload = False
        if use_gradient_checkpointing or use_gradient_checkpointing_offload:
            print("Gradient checkpointing is disabled for teacher feature capture.")
        self.extra_inputs = extra_inputs.split(",") if extra_inputs else []
        self.fp8_models = fp8_models
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        # This mapping module intentionally bypasses WanTrainingModule.__init__
        # to avoid constructing its memory-augmented student DiT, but it still
        # reuses WanTrainingModule.get_pipeline_inputs().  Initialize that
        # inherited method's complete, versioned configuration in one place so
        # new fields (CFG/high-noise/trajectory settings) cannot be missing.
        self._init_training_loss_config(
            default_cfg_scale=1.0,
            default_trajectory_kd_steps=1,
        )
        self.attention_loss_weight = float(attention_loss_weight)
        self.block_loss_weight = float(block_loss_weight)
        self.mapping_regularization = float(mapping_regularization)
        self.normalized_mse = bool(normalized_mse)
        self.mapping_log_interval = max(1, int(mapping_log_interval))
        requested_blocks_per_step = int(mapping_blocks_per_step)
        if requested_blocks_per_step < 0:
            raise ValueError(
                "mapping_blocks_per_step must be non-negative; "
                f"got {requested_blocks_per_step}"
            )
        if requested_blocks_per_step > len(self.block_indices):
            raise ValueError(
                "mapping_blocks_per_step cannot exceed the number of mapping "
                f"blocks ({len(self.block_indices)}); got "
                f"{requested_blocks_per_step}"
            )
        self.mapping_blocks_per_step = (
            len(self.block_indices)
            if requested_blocks_per_step == 0
            else requested_blocks_per_step
        )
        self.checkpoint_mapped_block = bool(checkpoint_mapped_block)
        self._mapping_steps = int(mapping_step_offset)
        if self._mapping_steps < 0:
            raise ValueError(
                f"mapping_step_offset must be non-negative; got {self._mapping_steps}"
            )
        self.register_buffer(
            "mixing_initial_value", _manual_copy_matrix(mixing_init), persistent=False
        )
        self.latest_loss_metrics = {}
        self._printed_initial_metrics = False

        trainable = [
            (name, parameter.numel())
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        ]
        if len(self.block_indices) == 1:
            print(f"Selected single replacement block: {self.block_index}")
        else:
            print(
                "Selected shared-mapping training blocks: "
                f"{self.block_indices} (uniform over {len(self.pipe.dit.blocks)} blocks)"
            )
        print(f"Mixing source order: {SOURCE_NAMES}; target order: {TARGET_NAMES}")
        print(f"Initial mixing matrix:\n{self.mapped_attention.mixing.detach().cpu()}")
        print(
            "Auxiliary mapping: "
            f"enabled={self.mapped_attention.map_auxiliary_parameters}; "
            f"targets={tuple(AUXILIARY_TARGET_PARAMETER_NAMES)}"
        )
        print(
            f"Base mapping trainable: {self.train_base_mapping}; "
            f"low-rank residual rank: {self.mapped_attention.low_rank}"
        )
        if self.mapping_blocks_per_step == len(self.block_indices):
            print(
                "Mapping block minibatch: all "
                f"{len(self.block_indices)} blocks per rank and step"
            )
        else:
            print(
                "Mapping block minibatch: "
                f"{self.mapping_blocks_per_step}/{len(self.block_indices)} blocks "
                "per rank and step (deterministic rank-strided schedule); "
                f"step offset={self._mapping_steps}"
            )
        print(
            "Mapped-block activation checkpoint: "
            f"{self.checkpoint_mapped_block}"
        )
        print(
            f"Trainable mapping parameters ({sum(size for _, size in trainable)} total): {trainable}"
        )

    @property
    def teacher_dit(self):
        return self.pipe.dit

    def train(self, mode=True):
        super().train(mode)
        self.pipe.eval()
        self.mapped_attention.train(mode)
        for mapper in self.additional_mapped_attentions.values():
            mapper.train(mode)
        return self

    def _mapper_for_block(self, block_index):
        if block_index == self.block_index:
            return self.mapped_attention
        try:
            return self.additional_mapped_attentions[str(block_index)]
        except KeyError as error:
            raise KeyError(
                f"Block {block_index} is not one of the mapping training blocks "
                f"{self.block_indices}"
            ) from error

    def _active_mapping_block_indices(self):
        """Select an unbiased, deterministic block minibatch for this rank.

        The full 30-block objective remains the default.  Long-video profiles
        can select a smaller number of blocks per rank so one optimizer update
        averages different block losses across DDP ranks without retaining 30
        large GDN autograd graphs on every GPU.
        """
        if self.mapping_blocks_per_step >= len(self.block_indices):
            return tuple(self.block_indices)

        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        rank = int(
            os.environ.get(
                "RANK",
                os.environ.get("LOCAL_RANK", "0"),
            )
        )
        rank %= world_size
        count = self.mapping_blocks_per_step
        start = (
            self._mapping_steps * world_size * count + rank * count
        ) % len(self.block_indices)
        selected = [
            self.block_indices[(start + offset) % len(self.block_indices)]
            for offset in range(count)
        ]
        # Teacher execution follows model depth and may stop after the deepest
        # selected block, so return cyclic selections in ascending depth order.
        return tuple(sorted(selected))

    def _load_mapping_checkpoint(self, checkpoint_path):
        state_dict = load_state_dict(checkpoint_path, torch_dtype=None, device="cpu")
        prefixes = (
            "mapped_attention.",
            "module.mapped_attention.",
        )
        normalized = {}
        for name, value in state_dict.items():
            normalized_name = name
            for prefix in prefixes:
                if normalized_name.startswith(prefix):
                    normalized_name = normalized_name[len(prefix) :]
                    break
            normalized[normalized_name] = value
        checkpoint_blocks = normalized.pop("selected_block_indices", None)
        checkpoint_block = normalized.pop("selected_block_index", None)
        source_blocks = (
            [int(value) for value in checkpoint_blocks.reshape(-1).tolist()]
            if checkpoint_blocks is not None
            else (
                [int(checkpoint_block.item())]
                if checkpoint_block is not None
                else []
            )
        )
        if len(self.block_indices) == 1 and source_blocks != self.block_indices:
            raise ValueError(
                f"Mapping checkpoint was trained for blocks {source_blocks}, "
                f"but this run selected block {self.block_index}"
            )
        if len(self.block_indices) > 1 and source_blocks != self.block_indices:
            print(
                "Initializing shared mapping coefficients from checkpoint trained "
                f"on blocks {source_blocks}; current blocks are {self.block_indices}."
            )
        checkpoint_version = normalized.pop("mapping_format_version", None)
        version = int(checkpoint_version.item()) if checkpoint_version is not None else 1
        legacy_fullmix = (
            "mixing" in normalized
            and "subhead_mixing.q" not in normalized
        )
        if version > 9 and not legacy_fullmix:
            raise ValueError(
                f"Unsupported mapping checkpoint format {version}"
            )
        checkpoint_expand_v = normalized.pop("mapping_gdn_expand_v", None)
        checkpoint_head_dim = normalized.pop("mapping_gdn_head_dim", None)
        checkpoint_hidden_size = normalized.pop("mapping_gdn_hidden_size", None)
        checkpoint_num_heads = normalized.pop("mapping_gdn_num_heads", None)
        checkpoint_adapter = normalized.pop("mapping_shape_adapter", None)
        if version >= 4 and not legacy_fullmix:
            if (
                checkpoint_expand_v is None
                or checkpoint_head_dim is None
                or checkpoint_adapter is None
            ):
                raise ValueError(
                    f"Format-v{version} mapping checkpoint is missing architecture metadata"
                )
            expected_expand_v = float(self.mapped_attention.mapping_gdn_expand_v.item())
            expected_head_dim = int(self.mapped_attention.mapping_gdn_head_dim.item())
            expected_adapter = int(self.mapped_attention.mapping_shape_adapter.item())
            actual = (
                float(checkpoint_expand_v.item()),
                int(checkpoint_head_dim.item()),
                int(checkpoint_adapter.item()),
            )
            expected = (expected_expand_v, expected_head_dim, expected_adapter)
            if actual != expected and not legacy_fullmix:
                raise ValueError(
                    "Mapping checkpoint architecture mismatch: "
                    f"checkpoint={actual}, current={expected}"
                )
            if version >= 9:
                if checkpoint_hidden_size is None or checkpoint_num_heads is None:
                    raise ValueError(
                        "Format-v9 mapping checkpoint is missing exact-GDN hidden/head metadata"
                    )
                actual_wide = (
                    int(checkpoint_hidden_size.item()), int(checkpoint_num_heads.item())
                )
                expected_wide = (
                    int(self.mapped_attention.mapping_gdn_hidden_size.item()),
                    int(self.mapped_attention.mapping_gdn_num_heads.item()),
                )
                if actual_wide != expected_wide:
                    raise ValueError(
                        "Mapping checkpoint exact-GDN size mismatch: "
                        f"checkpoint={actual_wide}, current={expected_wide}"
                    )
            elif self.mapped_attention.target_dim != self.mapped_attention.dim:
                raise ValueError(
                    "A pre-v9 checkpoint cannot initialize a projected exact-size GDN"
                )
        elif self.mapped_attention.shape_adapter != "subheadmix":
            raise ValueError(
                "Legacy mapping checkpoints have no architecture metadata and can "
                "only initialize the direct square mapper."
            )
        for metadata_name in (
            "mapping_auxiliary_enabled",
            "mapping_auxiliary_mix_granularity",
            "mapping_low_rank",
            "mapping_full_residual_enabled",
            "mapping_post_full_residual_enabled",
            "mapping_postmix_enabled",
            "mapping_mix_granularity",
            "mapping_mix_order",
            "mapping_source_specific_expansion",
        ):
            normalized.pop(metadata_name, None)
        if legacy_fullmix:
            source_mixing = normalized["mixing"]
            if tuple(source_mixing.shape) != tuple(self.mapped_attention.mixing.shape):
                raise ValueError(
                    f"Legacy FullMix coefficient shape {tuple(source_mixing.shape)} "
                    f"does not match {tuple(self.mapped_attention.mixing.shape)}"
                )
            with torch.no_grad():
                self.mapped_attention.mixing.copy_(source_mixing)
                for name, alpha in self.mapped_attention.subhead_mixing.items():
                    alpha.copy_(
                        source_mixing[TARGET_NAMES.index(name)]
                        .view(1, 1, -1)
                        .expand_as(alpha)
                    )
                self.mapped_attention.headwise_mixing.copy_(
                    source_mixing[TARGET_NAMES.index("beta")]
                    .view(1, -1)
                    .expand_as(self.mapped_attention.headwise_mixing)
                )
                checkpoint_headwise = normalized.get("headwise_mixing")
                if checkpoint_headwise is not None:
                    for name, alpha in self.mapped_attention.subhead_mixing.items():
                        alpha.copy_(
                            checkpoint_headwise[TARGET_NAMES.index(name)]
                            .unsqueeze(1)
                        )
                    self.mapped_attention.headwise_mixing.copy_(
                        checkpoint_headwise[TARGET_NAMES.index("beta")]
                    )
                for name, parameter in self.mapped_attention.auxiliary_mixing.items():
                    checkpoint_name = f"auxiliary_mixing.{name}"
                    if checkpoint_name in normalized:
                        parameter.copy_(normalized[checkpoint_name])
                for name, parameter in self.mapped_attention.auxiliary_headwise_mixing.items():
                    checkpoint_name = f"auxiliary_headwise_mixing.{name}"
                    if checkpoint_name in normalized:
                        parameter.copy_(normalized[checkpoint_name])
                for name, parameter in self.mapped_attention.full_residual.items():
                    checkpoint_name = f"full_residual.{name}"
                    if checkpoint_name not in normalized:
                        raise ValueError(
                            "Linear-adapter initialization requires a HeadMix+"
                            f"full-residual checkpoint missing {checkpoint_name}"
                        )
                    parameter.copy_(normalized[checkpoint_name])
            print(
                f"Imported FullMix coefficients into SubheadMix from {checkpoint_path}"
            )
            return
        if len(self.block_indices) > 1:
            normalized = {
                name: value
                for name, value in normalized.items()
                if not name.startswith("gdn.")
            }
        load_result = self.mapped_attention.load_state_dict(normalized, strict=False)
        if load_result.unexpected_keys:
            raise ValueError(
                f"Unexpected keys in mapping checkpoint {checkpoint_path}: "
                f"{load_result.unexpected_keys[:8]}"
            )
        print(
            f"Loaded mapping checkpoint: {checkpoint_path} | "
            f"missing_keys={len(load_result.missing_keys)}"
        )

    def checkpoint_state_dict_converter(self, state_dict):
        if len(self.block_indices) > 1:
            return {
                name: value
                for name, value in state_dict.items()
                if not name.startswith("gdn.")
            }
        if any(name.startswith("gdn.") for name in state_dict):
            with torch.no_grad():
                materialized = self.mapped_attention.materialized_parameters()
            for name, value in materialized.items():
                state_dict[f"gdn.{name}"] = value.detach()
        return state_dict

    def _capture_teacher_block_features(self, model_kwargs, block_indices=None):
        if block_indices is None:
            block_indices = self.block_indices
        block_indices = tuple(int(index) for index in block_indices)
        if not block_indices:
            raise ValueError("At least one teacher block must be captured")
        captured = {index: {} for index in block_indices}
        attention_handles = []
        block_handles = []
        last_block_index = max(block_indices)

        def make_attention_hook(layer_index):
            def attention_hook(module, args, output):
                del module, args
                captured[layer_index]["attention_output"] = output.detach()

            return attention_hook

        def make_block_hook(layer_index):
            def block_hook(module, args, output):
                del module
                captured[layer_index]["block_inputs"] = tuple(
                    value.detach() for value in args[:4]
                )
                captured[layer_index]["block_output"] = output.detach()
                if layer_index == last_block_index:
                    raise _StopAfterSelectedBlock()

            return block_hook

        for layer_index in block_indices:
            block = self.teacher_dit.blocks[layer_index]
            attention_handles.append(
                block.self_attn.register_forward_hook(
                    make_attention_hook(layer_index)
                )
            )
            block_handles.append(
                block.register_forward_hook(make_block_hook(layer_index))
            )
        try:
            with torch.no_grad():
                teacher_model_kwargs = dict(
                    dit=self.teacher_dit,
                    use_gradient_checkpointing=False,
                    use_gradient_checkpointing_offload=False,
                    **model_kwargs,
                )
                if os.environ.get("DIFFSYNTH_STANDARD_WAN_TEACHER", "0") != "1":
                    teacher_model_kwargs["memory_tokens"] = None
                self.pipe.model_fn(**teacher_model_kwargs)
        except _StopAfterSelectedBlock:
            pass
        finally:
            for handle in attention_handles + block_handles:
                handle.remove()

        required = {"attention_output", "block_inputs", "block_output"}
        for layer_index, layer_capture in captured.items():
            missing = required.difference(layer_capture)
            if missing:
                raise RuntimeError(
                    f"Failed to capture teacher block {layer_index} features: "
                    f"missing {sorted(missing)}"
                )
        return captured

    def _forward_mapped_block(
        self, block_inputs, mapped_parameters=None, block_index=None
    ):
        if block_index is None:
            block_index = self.block_index
        x, context, t_mod, freqs = block_inputs
        block = self.teacher_dit.blocks[block_index]
        mapper = self._mapper_for_block(block_index)
        has_sequence_timestep = len(t_mod.shape) == 4
        chunk_dim = 2 if has_sequence_timestep else 1
        modulation = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            modulation.chunk(6, dim=chunk_dim)
        )
        if has_sequence_timestep:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )

        attention_input = block.norm1(x) * (1 + scale_msa) + shift_msa
        attention_output = mapper(
            attention_input,
            freqs,
            mapped_parameters=mapped_parameters,
        )
        x = block.gate(x, gate_msa, attention_output)
        x = x + block.cross_attn(block.norm3(x), context)
        mlp_input = block.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return attention_output, x

    def _feature_mse(self, prediction, target):
        loss = F.mse_loss(prediction.float(), target.float())
        if self.normalized_mse:
            loss = loss / target.float().square().mean().clamp_min(1e-8)
        return loss

    def weight_mapping_loss(self, inputs_shared, inputs_posi):
        clean_latents = inputs_shared.get("input_latents")
        if clean_latents is None:
            clean_latents = inputs_shared["latents"]
        noise = self.pipe.generate_noise(
            clean_latents.shape, rand_device=clean_latents.device
        ).to(clean_latents.dtype)
        num_timesteps = len(self.pipe.scheduler.timesteps)
        min_timestep_id = int(self.min_timestep_boundary * num_timesteps)
        max_timestep_id = int(self.max_timestep_boundary * num_timesteps)
        if not 0 <= min_timestep_id < max_timestep_id <= num_timesteps:
            raise ValueError(
                "Invalid timestep boundaries: "
                f"min={self.min_timestep_boundary}, "
                f"max={self.max_timestep_boundary}"
            )
        timestep_id = torch.randint(min_timestep_id, max_timestep_id, (1,))
        timesteps = self.pipe.scheduler.timesteps[timestep_id].to(
            dtype=self.pipe.torch_dtype,
            device=clean_latents.device,
        )
        noisy_latents = self.pipe.scheduler.add_noise(
            clean_latents, noise, timestep=timesteps
        )
        noisy_latents = noisy_latents.to(clean_latents.dtype)

        if (
            inputs_shared.get("fuse_vae_embedding_in_latents", False)
            and inputs_shared.get("first_frame_latents") is not None
        ):
            noisy_latents = noisy_latents.clone()
            noisy_latents[:, :, 0:1] = inputs_shared["first_frame_latents"]

        model_kwargs = {
            "latents": noisy_latents,
            "timestep": timesteps,
            "context": inputs_posi.get("context"),
            "y": inputs_shared.get("y"),
            "clip_feature": inputs_shared.get("clip_feature"),
            "fuse_vae_embedding_in_latents": inputs_shared.get(
                "fuse_vae_embedding_in_latents", False
            ),
        }
        active_block_indices = self._active_mapping_block_indices()
        teacher_by_block = self._capture_teacher_block_features(
            model_kwargs,
            block_indices=active_block_indices,
        )
        attention_losses = []
        block_losses = []
        per_block_metrics = {}
        for layer_index in active_block_indices:
            teacher = teacher_by_block[layer_index]

            def mapped_block_forward(*block_inputs, current_layer=layer_index):
                return self._forward_mapped_block(
                    block_inputs,
                    block_index=current_layer,
                )

            if self.checkpoint_mapped_block:
                mapped_attention, mapped_block = activation_checkpoint(
                    mapped_block_forward,
                    *teacher["block_inputs"],
                    use_reentrant=False,
                )
            else:
                mapped_attention, mapped_block = mapped_block_forward(
                    *teacher["block_inputs"]
                )
            layer_attention_mse = self._feature_mse(
                mapped_attention, teacher["attention_output"]
            )
            layer_block_mse = self._feature_mse(
                mapped_block, teacher["block_output"]
            )
            attention_losses.append(layer_attention_mse)
            block_losses.append(layer_block_mse)
            per_block_metrics[layer_index] = (
                layer_block_mse.detach(),
                layer_attention_mse.detach(),
            )
            self._mapper_for_block(layer_index).last_output = None
            del mapped_attention, mapped_block

        attention_mse = torch.stack(attention_losses).mean()
        block_mse = torch.stack(block_losses).mean()
        mapping_delta = self.mapped_attention.mixing - self.mixing_initial_value.to(
            self.mapped_attention.mixing.device
        )
        regularization_deltas = [mapping_delta.reshape(-1)]
        if self.mapped_attention.map_auxiliary_parameters:
            auxiliary_parameters = {
                "head": self.mapped_attention.auxiliary_mixing,
                "full": self.mapped_attention.auxiliary_full_mixing,
                "channel": self.mapped_attention.auxiliary_channel_mixing,
            }[self.mapped_attention.auxiliary_mix_granularity]
            regularization_deltas.extend(
                parameter.reshape(-1)
                for parameter in auxiliary_parameters.values()
            )
        regularization = torch.cat(regularization_deltas).square().mean()
        loss = (
            self.attention_loss_weight * attention_mse
            + self.block_loss_weight * block_mse
            + self.mapping_regularization * regularization
        )
        self.latest_loss_metrics = {
            "task_loss": block_mse.detach(),
            "kd_loss": attention_mse.detach(),
            "mapping_regularization": regularization.detach(),
        }
        self._mapping_steps += 1
        if not self._printed_initial_metrics:
            print(
                "Initial feature alignment | "
                f"block_mse={block_mse.item():.6f} "
                f"attention_mse={attention_mse.item():.6f}"
            )
            if len(self.block_indices) > 1:
                formatted = {
                    layer_index: {
                        "block_mse": float(values[0].item()),
                        "attention_mse": float(values[1].item()),
                    }
                    for layer_index, values in per_block_metrics.items()
                }
                print(f"Initial per-block alignment: {formatted}")
            self._printed_initial_metrics = True
        if (
            self._mapping_steps % self.mapping_log_interval == 0
            and int(os.environ.get("LOCAL_RANK", "0")) == 0
        ):
            delta_norm = mapping_delta.detach().float().norm().item()
            print(
                f"T matrix at local step {self._mapping_steps} (delta_norm={delta_norm:.6f}):\n"
                f"{self.mapped_attention.mixing.detach().float().cpu()}"
            )
            if len(self.block_indices) > 1:
                formatted = {
                    layer_index: {
                        "block_mse": float(values[0].item()),
                        "attention_mse": float(values[1].item()),
                    }
                    for layer_index, values in per_block_metrics.items()
                }
                print(
                    f"Per-block alignment at local step {self._mapping_steps}: "
                    f"{formatted}"
                )
                print(
                    "Active mapping blocks at local step "
                    f"{self._mapping_steps}: {list(active_block_indices)}"
                )
            if self.mapped_attention.map_auxiliary_parameters:
                auxiliary = {
                    name: parameter.detach().float().cpu().tolist()
                    for name, parameter in {
                        "head": self.mapped_attention.auxiliary_mixing,
                        "full": self.mapped_attention.auxiliary_full_mixing,
                        "channel": self.mapped_attention.auxiliary_channel_mixing,
                    }[self.mapped_attention.auxiliary_mix_granularity].items()
                }
                print(
                    "Auxiliary "
                    f"{self.mapped_attention.auxiliary_mix_granularity} mixing "
                    f"at local step {self._mapping_steps}: {auxiliary}"
                )
            if self.mapped_attention.low_rank > 0:
                low_rank_norms = {
                    name: {
                        "u": self.mapped_attention.low_rank_u[name]
                        .detach()
                        .float()
                        .norm()
                        .item(),
                        "v": self.mapped_attention.low_rank_v[name]
                        .detach()
                        .float()
                        .norm()
                        .item(),
                        "residual": self.mapped_attention._low_rank_residual(name)
                        .detach()
                        .float()
                        .norm()
                        .item(),
                    }
                    for name in TARGET_NAMES
                }
                print(
                    f"Low-rank norms at local step {self._mapping_steps}: "
                    f"{low_rank_norms}"
                )
        return loss

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(
            inputs, self.pipe.device, self.pipe.torch_dtype
        )
        with torch.no_grad():
            for unit in self.pipe.units:
                inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        inputs_shared, inputs_posi, _ = inputs
        if os.environ.get("DIFFSYNTH_STANDARD_WAN_TEACHER", "0") != "1":
            inputs_shared["memory_tokens"] = None
        return self.weight_mapping_loss(inputs_shared, inputs_posi)


class WeightMappingLogger(ModelLogger):
    """Save only mapping/GDN state, never gather the frozen teacher checkpoint."""

    def save_model(
        self,
        accelerator,
        model,
        file_name,
        save_full_model=True,
        synchronize=True,
    ):
        if synchronize:
            accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            mapper = unwrapped.mapped_attention
            full_mapper_state = mapper.state_dict()
            if save_full_model:
                mapper_state = unwrapped.checkpoint_state_dict_converter(
                    full_mapper_state
                )
            else:
                mapper_state = {
                    name: value
                    for name, value in full_mapper_state.items()
                    if name == "mixing"
                    or name.startswith("subhead_mixing.")
                    or name.startswith("channel_mixing.")
                    or (
                        name.startswith("linear_adapter.")
                        and not mapper.source_specific_expansion
                    )
                    or (
                        name.startswith("source_linear_adapter.")
                        and mapper.source_specific_expansion
                    )
                    or name == "headwise_mixing"
                    or name.startswith("auxiliary_mixing.")
                    or name.startswith("auxiliary_headwise_mixing.")
                    or name.startswith("auxiliary_full_mixing.")
                    or name.startswith("auxiliary_channel_mixing.")
                    or name.startswith("low_rank_u.")
                    or name.startswith("low_rank_v.")
                    or name.startswith("full_residual.")
                    or name.startswith("post_full_residual.")
                    or name in {"hidden_input_adapter", "hidden_output_adapter"}
                }
                for metadata_name in (
                    "selected_block_index",
                    "selected_block_indices",
                    "mapping_format_version",
                    "mapping_gdn_expand_v",
                    "mapping_gdn_head_dim",
                    "mapping_gdn_hidden_size",
                    "mapping_gdn_num_heads",
                    "mapping_shape_adapter",
                    "mapping_subheadmix_enabled",
                    "mapping_auxiliary_enabled",
                    "mapping_auxiliary_mix_granularity",
                    "mapping_low_rank",
                    "mapping_full_residual_enabled",
                    "mapping_post_full_residual_enabled",
                    "mapping_postmix_enabled",
                    "mapping_mix_granularity",
                    "mapping_mix_order",
                    "mapping_source_specific_expansion",
                ):
                    mapper_state[metadata_name] = full_mapper_state[metadata_name]
            # DeepSpeed may expose parameters as views of flattened storage; GDN's
            # o_proj may also be a transposed non-contiguous view.  A default
            # clone preserves strides, which safetensors rejects, so force a
            # contiguous clone for every serialized tensor.
            mapper_state = {
                name: value.detach().clone(memory_format=torch.contiguous_format)
                for name, value in mapper_state.items()
            }
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(mapper_state, path, safe_serialization=True)
            print(
                f"Weight mapping checkpoint saved to: {path} ({len(mapper_state)} tensors)"
            )
        if synchronize:
            accelerator.wait_for_everyone()


class PrefixDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_samples):
        self.dataset = dataset
        self.max_samples = min(len(dataset), int(max_samples))
        self.load_from_cache = getattr(dataset, "load_from_cache", False)

    def __len__(self):
        return self.max_samples

    def __getitem__(self, index):
        return self.dataset[index]


def weight_mapping_parser():
    parser = base_wan_parser()
    parser.description = "Shared softmax-to-GDN weight mapping experiment"
    parser.add_argument(
        "--block_index",
        type=int,
        default=int(os.environ.get("MAPPING_BLOCK_INDEX", -1)),
    )
    parser.add_argument(
        "--gdn_block_indices",
        type=str,
        default=os.environ.get("GDN_BLOCK_INDICES", ""),
        help=(
            "Comma-separated training blocks. Multiple blocks share one mapping; "
            "when omitted, --mapping_num_layers controls uniform selection."
        ),
    )
    parser.add_argument(
        "--mapping_num_layers",
        type=int,
        default=int(os.environ.get("MAPPING_NUM_LAYERS", 1)),
        help=(
            "Number of uniformly spaced blocks that jointly train one shared mapping. "
            "The default 1 preserves the original single-layer behavior."
        ),
    )
    parser.add_argument("--gdn_group_size", type=int, default=5, help=argparse.SUPPRESS)
    parser.add_argument(
        "--middle_layer_start",
        type=int,
        default=int(os.environ.get("MIDDLE_LAYER_START", 10)),
    )
    parser.add_argument(
        "--middle_layer_end",
        type=int,
        default=int(os.environ.get("MIDDLE_LAYER_END", 19)),
    )
    parser.add_argument(
        "--mapping_seed", type=int, default=int(os.environ.get("MAPPING_SEED", 42))
    )
    parser.add_argument(
        "--training_seed",
        type=int,
        default=int(os.environ.get("TRAINING_SEED", -1)),
        help="Seed training randomness when non-negative; -1 keeps random behavior.",
    )
    parser.add_argument(
        "--mixing_init",
        choices=("manual_copy", "method2_average"),
        default=os.environ.get("MAPPING_INIT", "manual_copy"),
    )
    parser.add_argument(
        "--low_rank", type=int, default=int(os.environ.get("MAPPING_LOW_RANK", 0))
    )
    parser.add_argument(
        "--full_residual",
        type=_parse_bool,
        default=_env_bool("MAPPING_FULL_RESIDUAL", False),
        help="Train an unconstrained dense residual; mutually exclusive with --low_rank.",
    )
    parser.add_argument(
        "--post_full_residual",
        type=_parse_bool,
        default=_env_bool("MAPPING_POST_FULL_RESIDUAL", False),
        help="Train a native-width dense residual after linear expansion.",
    )
    parser.add_argument(
        "--postmix",
        type=_parse_bool,
        default=_env_bool("MAPPING_POSTMIX", False),
        help=(
            "Expand every source projection with dense A/B before applying "
            "HeadMix in the widened subhead space."
        ),
    )
    parser.add_argument(
        "--mix_granularity",
        choices=("full", "head", "channel"),
        default=os.environ.get("MAPPING_MIX_GRANULARITY", "head"),
    )
    parser.add_argument(
        "--mix_order",
        choices=("pre", "post"),
        default=os.environ.get(
            "MAPPING_MIX_ORDER",
            "post" if _env_bool("MAPPING_POSTMIX", False) else "pre",
        ),
    )
    parser.add_argument(
        "--source_specific_expansion",
        type=_parse_bool,
        default=_env_bool("MAPPING_SOURCE_SPECIFIC_EXPANSION", False),
        help="Use independent dense expansion matrices for every (target, source) pair.",
    )
    parser.add_argument(
        "--low_rank_init_std",
        type=float,
        default=float(os.environ.get("LOW_RANK_INIT_STD", 0.02)),
    )
    parser.add_argument(
        "--gdn_expand_v",
        type=float,
        default=float(os.environ.get("MAPPING_GDN_EXPAND_V", 2.0)),
    )
    parser.add_argument(
        "--gdn_head_dim",
        type=int,
        default=int(os.environ.get("MAPPING_GDN_HEAD_DIM", 256)),
    )
    parser.add_argument(
        "--gdn_hidden_size",
        type=int,
        default=int(os.environ.get("MAPPING_GDN_HIDDEN_SIZE", 0)),
        help="Internal GDN hidden size; 0 follows the teacher hidden size.",
    )
    parser.add_argument(
        "--gdn_num_heads",
        type=int,
        default=int(os.environ.get("MAPPING_GDN_NUM_HEADS", 0)),
        help="Internal GDN head count; 0 follows the teacher head count.",
    )
    parser.add_argument(
        "--mapping_shape_adapter",
        choices=SHAPE_ADAPTER_NAMES,
        default=os.environ.get("MAPPING_SHAPE_ADAPTER", "subheadmix"),
        help=(
            "SubheadMix is the only mapping mode in this isolated entry point. "
            "It uses independent alpha coefficients for every (head, subblock)."
        ),
    )
    parser.add_argument(
        "--gdn_use_short_conv",
        type=_parse_bool,
        default=_env_bool("MAPPING_GDN_USE_SHORT_CONV", True),
    )
    parser.add_argument(
        "--identity_short_conv",
        type=_parse_bool,
        default=_env_bool("MAPPING_IDENTITY_SHORT_CONV", True),
    )
    parser.add_argument(
        "--map_auxiliary_parameters",
        type=_parse_bool,
        default=_env_bool("MAPPING_MAP_AUXILIARY", False),
        help=(
            "Learn mixing adapters for a_proj, A_log, dt_bias, q/k/v short conv, "
            "and o_norm in addition to q/k/v/o/g/b."
        ),
    )
    parser.add_argument(
        "--train_base_mapping",
        type=_parse_bool,
        default=_env_bool("MAPPING_TRAIN_BASE", True),
        help=(
            "Train mixing and auxiliary-mixing coefficients. Disable this for a "
            "controlled low-rank-only residual experiment."
        ),
    )
    parser.add_argument(
        "--attention_loss_weight",
        type=float,
        default=float(os.environ.get("ATTENTION_LOSS_WEIGHT", 1.0)),
    )
    parser.add_argument(
        "--block_loss_weight",
        type=float,
        default=float(os.environ.get("BLOCK_LOSS_WEIGHT", 1.0)),
    )
    parser.add_argument(
        "--mapping_regularization",
        type=float,
        default=float(os.environ.get("MAPPING_REGULARIZATION", 0.0)),
    )
    parser.add_argument(
        "--normalized_mse",
        type=_parse_bool,
        default=_env_bool("MAPPING_NORMALIZED_MSE", False),
    )
    parser.add_argument(
        "--mapping_log_interval",
        type=int,
        default=int(os.environ.get("MAPPING_MATRIX_LOG_INTERVAL", 100)),
    )
    parser.add_argument(
        "--mapping_max_samples",
        type=int,
        default=int(os.environ.get("MAPPING_MAX_SAMPLES", 4096)),
        help="Limit the global dataset prefix for the transform-search experiment; 0 keeps all samples.",
    )
    parser.add_argument(
        "--mapping_blocks_per_step",
        type=int,
        default=int(os.environ.get("MAPPING_BLOCKS_PER_STEP", 0)),
        help=(
            "Number of mapping blocks evaluated per rank and step. "
            "0 evaluates all selected blocks; a smaller value uses a "
            "deterministic rank-strided unbiased block minibatch."
        ),
    )
    parser.add_argument(
        "--checkpoint_mapped_block",
        type=_parse_bool,
        default=_env_bool("MAPPING_CHECKPOINT_MAPPED_BLOCK", False),
        help=(
            "Activation-checkpoint the mapped GDN plus the remainder of its "
            "Wan block. This does not checkpoint the frozen teacher pass."
        ),
    )
    return parser


def main():
    os.environ.setdefault("DIFFSYNTH_T2V_TRAINING", "1")
    timeout_seconds = int(os.environ.get("DIFFSYNTH_NCCL_TIMEOUT_SEC", "7200"))
    process_group_timeout = timedelta(seconds=timeout_seconds)

    import torch.distributed as distributed

    original_init_process_group = distributed.init_process_group
    original_new_group = distributed.new_group

    def init_process_group(*args, **kwargs):
        kwargs.setdefault("timeout", process_group_timeout)
        print(
            f"[NCCL patch] init_process_group timeout={kwargs['timeout']}", flush=True
        )
        return original_init_process_group(*args, **kwargs)

    def new_group(*args, **kwargs):
        kwargs.setdefault("timeout", process_group_timeout)
        return original_new_group(*args, **kwargs)

    distributed.init_process_group = init_process_group
    distributed.new_group = new_group

    parser = weight_mapping_parser()
    args = parser.parse_args()
    if args.training_seed >= 0:
        set_seed(args.training_seed, device_specific=False)
        print(f"Training seed: {args.training_seed}", flush=True)
    ddp_kwargs = accelerate.DistributedDataParallelKwargs(
        find_unused_parameters=args.find_unused_parameters,
        broadcast_buffers=False,
    )
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            ddp_kwargs,
            InitProcessGroupKwargs(timeout=process_group_timeout),
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
        ),
    )
    if args.mapping_max_samples < 0:
        raise ValueError("--mapping_max_samples must be non-negative")
    if args.mapping_max_samples > 0 and args.mapping_max_samples < len(dataset):
        dataset = PrefixDataset(dataset, args.mapping_max_samples)
        accelerator.print(
            f"Weight mapping dataset limited to {len(dataset)} global samples"
        )
    model = WanSingleLayerWeightMappingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        task=args.task,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        student_init_ckpt=args.student_init_ckpt,
        block_index=args.block_index,
        gdn_block_indices=args.gdn_block_indices,
        mapping_num_layers=args.mapping_num_layers,
        middle_layer_start=args.middle_layer_start,
        middle_layer_end=args.middle_layer_end,
        mapping_seed=args.mapping_seed,
        mixing_init=args.mixing_init,
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
        attention_loss_weight=args.attention_loss_weight,
        block_loss_weight=args.block_loss_weight,
        mapping_regularization=args.mapping_regularization,
        normalized_mse=args.normalized_mse,
        mapping_log_interval=args.mapping_log_interval,
        mapping_blocks_per_step=args.mapping_blocks_per_step,
        checkpoint_mapped_block=args.checkpoint_mapped_block,
        mapping_step_offset=args.resume_step,
    )
    model_logger = WeightMappingLogger(args.output_path)
    launch_training_task(accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
