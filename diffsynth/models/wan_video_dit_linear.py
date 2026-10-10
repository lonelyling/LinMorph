import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange, repeat
from .wan_video_camera_controller import SimpleAdapter
from .fla_compat import patch_fla_l2norm_for_metax

patch_fla_l2norm_for_metax()

from fla.layers.gated_deltanet import GatedDeltaNet
from fla.modules.convolution import causal_conv1d
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if os.environ.get("DIFFSYNTH_DISABLE_FLASH_ATTN", "0") == "1":
        compatibility_mode = True
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x

class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        expand_v: float = 2.0,
        head_dim: int = 256,
        layer_idx: int = None,
        gdn_hidden_size: int = 0,
        gdn_num_heads: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        target_hidden_size = int(gdn_hidden_size) if gdn_hidden_size > 0 else dim
        target_num_heads = int(gdn_num_heads) if gdn_num_heads > 0 else num_heads
        if target_hidden_size < dim or target_hidden_size % dim != 0:
            raise ValueError(
                "Projected GDN hidden size must be an integer expansion of the "
                f"Wan block width: source={dim}, target={target_hidden_size}"
            )
        if target_num_heads < num_heads or target_num_heads % num_heads != 0:
            raise ValueError(
                "Projected GDN head count must be an integer expansion of the "
                f"Wan head count: source={num_heads}, target={target_num_heads}"
            )
        self.gdn_hidden_size = target_hidden_size
        self.gdn_num_heads = target_num_heads
        self.gdn_inference_impl = os.environ.get(
            "DIFFSYNTH_GDN_INFERENCE_IMPL", "fla"
        ).strip().lower()
        if self.gdn_inference_impl not in {"fla", "cuda_gates", "fla_folded", "fla_folded_out", "packed", "packed_stream", "packed_fused", "packed_fused_stream"}:
            raise ValueError(
                "DIFFSYNTH_GDN_INFERENCE_IMPL must be one of "
                f"fla/cuda_gates/fla_folded/fla_folded_out/packed/packed_stream/packed_fused/packed_fused_stream, got {self.gdn_inference_impl!r}"
            )
        self.gdn_stream_chunk_size = int(
            os.environ.get("DIFFSYNTH_GDN_STREAM_CHUNK_SIZE", "10240")
        )
        if self.gdn_stream_chunk_size <= 0:
            raise ValueError("DIFFSYNTH_GDN_STREAM_CHUNK_SIZE must be positive")
        self.gdn_release_unpacked_projections = os.environ.get(
            "DIFFSYNTH_GDN_RELEASE_UNPACKED_PROJECTIONS", "1"
        ) == "1"
        self.register_parameter("packed_qkvg_weight", None)
        self.register_parameter("packed_ab_weight", None)
        self.register_parameter("packed_qkv_conv_weight", None)
        self.register_parameter("packed_qkv_conv_bias", None)
        for name in ("q", "k", "v", "g", "a", "b", "o"):
            self.register_parameter(f"folded_{name}_weight", None)
        self.gdn_conv_backend = None
        self.gdn_conv_activation = None
        if head_dim <= 0:
            if target_hidden_size % target_num_heads != 0:
                raise ValueError(
                    f"gdn_hidden_size={target_hidden_size} must be divisible by "
                    f"gdn_num_heads={target_num_heads}"
                )
            head_dim = target_hidden_size // target_num_heads

        self.attn = GatedDeltaNet(
            hidden_size=target_hidden_size,
            expand_v=expand_v,
            head_dim=head_dim,
            num_heads=target_num_heads,
            num_v_heads=target_num_heads,
            mode="chunk",
            use_gate=True,
            use_short_conv=True,
            layer_idx=layer_idx,
            norm_eps=eps,
        )

        # The format-v9 Wan2.1 mapping keeps the target GDN at the exact
        # 3072-hidden/24-head shape and bridges the surrounding 1536-wide Wan
        # block at its boundary.  Repeat/average starts as a left inverse and
        # is overwritten by the learned mapping initialization when supplied.
        if target_hidden_size != dim:
            hidden_repeat = target_hidden_size // dim
            hidden_input = torch.eye(dim).repeat(hidden_repeat, 1)
            hidden_output = hidden_input.transpose(0, 1).contiguous() / hidden_repeat
            self.hidden_input_adapter = nn.Parameter(hidden_input)
            self.hidden_output_adapter = nn.Parameter(hidden_output)
        else:
            self.register_parameter("hidden_input_adapter", None)
            self.register_parameter("hidden_output_adapter", None)


    def enable_inference_optimization(
        self,
        implementation: str = "packed",
        stream_chunk_size: int = 10240,
        release_unpacked_projections: bool = True,
    ):
        """Enable the Qwen-style packed and Kairos-style streamed inference path.

        Call this after loading the checkpoint. Training keeps using the original
        FLA path. LoRA wrappers must be merged into plain nn.Linear modules first.
        """
        implementation = implementation.strip().lower()
        if implementation not in {"fla", "cuda_gates", "fla_folded", "fla_folded_out", "packed", "packed_stream", "packed_fused", "packed_fused_stream"}:
            raise ValueError(f"Unsupported GDN inference implementation: {implementation}")
        if stream_chunk_size <= 0:
            raise ValueError("stream_chunk_size must be positive")
        self.gdn_inference_impl = implementation
        self.gdn_stream_chunk_size = int(stream_chunk_size)
        self.gdn_release_unpacked_projections = bool(release_unpacked_projections)
        if implementation in {"fla_folded", "fla_folded_out"}:
            if implementation == "fla_folded":
                self._materialize_folded_projections()
            else:
                self._materialize_folded_output_projection()
            if self.gdn_release_unpacked_projections:
                self.release_unpacked_inference_components()
        elif implementation not in {"fla", "cuda_gates"}:
            self._materialize_packed_projections()
        if implementation in {"packed_fused", "packed_fused_stream"}:
            self._materialize_packed_qkv_convolution()
            if self.gdn_release_unpacked_projections:
                self.release_unpacked_inference_components()
        return self

    @torch.no_grad()
    def _materialize_folded_projections(self):
        """Experimental real-arithmetic rewrite; BF16 rounding is NOT preserved."""
        if self.folded_q_weight is not None:
            return
        if self.hidden_input_adapter is None or self.hidden_output_adapter is None:
            raise RuntimeError("Folded GDN requires learned input/output adapters")
        projection_names = ("q_proj", "k_proj", "v_proj", "g_proj", "a_proj", "b_proj")
        projections = [getattr(self.attn, name, None) for name in projection_names]
        if any(type(projection) is not nn.Linear for projection in projections):
            raise TypeError("Folded GDN requires materialized nn.Linear projections")
        if type(getattr(self.attn, "o_proj", None)) is not nn.Linear:
            raise TypeError("Folded GDN requires a materialized nn.Linear output projection")

        if any(projection.bias is not None for projection in projections):
            raise ValueError("Folded GDN requires bias-free input projections")

        input_adapter = self.hidden_input_adapter

        def fold_input(weight):
            # Build once in fp32, then store in the model dtype used by GEMM.
            return torch.mm(weight.float(), input_adapter.float()).to(weight.dtype).contiguous()

        for short_name, projection in zip(("q", "k", "v", "g", "a", "b"), projections):
            setattr(
                self,
                f"folded_{short_name}_weight",
                nn.Parameter(fold_input(projection.weight), requires_grad=False),
            )
        self._materialize_folded_output_projection()

    @torch.no_grad()
    def _materialize_folded_output_projection(self):
        if self.folded_o_weight is not None:
            return
        if self.hidden_output_adapter is None:
            raise RuntimeError("Folded output projection requires the learned output adapter")
        if type(getattr(self.attn, "o_proj", None)) is not nn.Linear:
            raise TypeError("Folded output projection requires a materialized nn.Linear o_proj")
        if self.attn.o_proj.bias is not None:
            raise ValueError("Folded GDN requires a bias-free output projection")
        folded_o = torch.mm(
            self.hidden_output_adapter.float(), self.attn.o_proj.weight.float()
        ).to(self.attn.o_proj.weight.dtype).contiguous()
        self.folded_o_weight = nn.Parameter(folded_o, requires_grad=False)

    @torch.no_grad()
    def _materialize_packed_projections(self):
        if self.packed_qkvg_weight is not None:
            return
        projection_names = ("q_proj", "k_proj", "v_proj", "g_proj", "a_proj", "b_proj")
        projections = [getattr(self.attn, name, None) for name in projection_names]
        unsupported = [
            name for name, projection in zip(projection_names, projections)
            if type(projection) is not nn.Linear
        ]
        if unsupported:
            raise TypeError(
                "Packed GDN inference requires materialized nn.Linear projections; "
                "merge LoRA/adapters first. Unsupported: " + ", ".join(unsupported)
            )
        q_proj, k_proj, v_proj, g_proj, a_proj, b_proj = projections
        self.packed_qkvg_weight = nn.Parameter(
            torch.cat(
                [q_proj.weight, k_proj.weight, v_proj.weight, g_proj.weight], dim=0
            ).contiguous(),
            requires_grad=False,
        )
        self.packed_ab_weight = nn.Parameter(
            torch.cat([b_proj.weight, a_proj.weight], dim=0).contiguous(),
            requires_grad=False,
        )
        if self.gdn_release_unpacked_projections:
            self.release_unpacked_inference_projections()

    def release_unpacked_inference_projections(self):
        if self.packed_qkvg_weight is None or self.packed_ab_weight is None:
            raise RuntimeError("Materialize packed projections before releasing originals")
        for name in ("q_proj", "k_proj", "v_proj", "g_proj", "a_proj", "b_proj"):
            setattr(self.attn, name, None)
        self.gdn_release_unpacked_projections = True

    @torch.no_grad()
    def _materialize_packed_qkv_convolution(self):
        if self.packed_qkv_conv_weight is not None:
            return
        names = ("q_conv1d", "k_conv1d", "v_conv1d")
        convolutions = [getattr(self.attn, name, None) for name in names]
        if any(convolution is None for convolution in convolutions):
            raise RuntimeError(
                "Fused QKV convolution requires the original convolution modules"
            )
        q_conv, k_conv, v_conv = convolutions
        signatures = {
            (conv.kernel_size, conv.backend, conv.activation, conv.bias is not None)
            for conv in convolutions
        }
        if len(signatures) != 1:
            raise RuntimeError("Q/K/V convolution settings differ and cannot be fused")
        self.packed_qkv_conv_weight = nn.Parameter(
            torch.cat(
                [conv.weight[:, 0, :] for conv in convolutions], dim=0
            ).contiguous(),
            requires_grad=False,
        )
        if q_conv.bias is not None:
            self.packed_qkv_conv_bias = nn.Parameter(
                torch.cat([conv.bias for conv in convolutions], dim=0).contiguous(),
                requires_grad=False,
            )
        self.gdn_conv_backend = q_conv.backend
        self.gdn_conv_activation = q_conv.activation

    def release_unpacked_inference_components(self):
        if self.gdn_inference_impl == "cuda_gates":
            return  # Original projections are needed by this exact-formula candidate.
        if self.gdn_inference_impl in {"fla_folded", "fla_folded_out"}:
            if self.folded_o_weight is None:
                raise RuntimeError("Materialize folded output projection before release")
            if self.gdn_inference_impl == "fla_folded":
                if self.folded_q_weight is None:
                    raise RuntimeError("Materialize folded input projections before release")
                for name in ("q_proj", "k_proj", "v_proj", "g_proj", "a_proj", "b_proj"):
                    setattr(self.attn, name, None)
                self.hidden_input_adapter = None
            self.attn.o_proj = None
            self.hidden_output_adapter = None
            self.gdn_release_unpacked_projections = True
            return
        self.release_unpacked_inference_projections()
        if self.gdn_inference_impl in {"packed_fused", "packed_fused_stream"}:
            if self.packed_qkv_conv_weight is None:
                raise RuntimeError("Materialize fused QKV convolution before release")
            for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
                setattr(self.attn, name, None)

    def _folded_gdn_chunk(self, hidden_states, fold_input=True, fold_output=True, cuda_gates=False):
        attn = self.attn
        q_weight = self.folded_q_weight if fold_input else attn.q_proj.weight
        k_weight = self.folded_k_weight if fold_input else attn.k_proj.weight
        v_weight = self.folded_v_weight if fold_input else attn.v_proj.weight
        g_weight = self.folded_g_weight if fold_input else attn.g_proj.weight
        a_weight = self.folded_a_weight if fold_input else attn.a_proj.weight
        b_weight = self.folded_b_weight if fold_input else attn.b_proj.weight
        q, _ = attn.q_conv1d(
            x=F.linear(hidden_states, q_weight),
            cache=None,
            output_final_state=False,
        )
        k, _ = attn.k_conv1d(
            x=F.linear(hidden_states, k_weight),
            cache=None,
            output_final_state=False,
        )
        v, _ = attn.v_conv1d(
            x=F.linear(hidden_states, v_weight),
            cache=None,
            output_final_state=False,
        )
        q, k = map(
            lambda tensor: rearrange(tensor, "... (h d) -> ... h d", d=attn.head_k_dim),
            (q, k),
        )
        v = rearrange(v, "... (h d) -> ... h d", d=attn.head_v_dim)
        if attn.num_v_heads > attn.num_heads:
            q, k = map(
                lambda tensor: repeat(
                    tensor,
                    "... h d -> ... (h g) d",
                    g=attn.num_v_heads // attn.num_heads,
                ),
                (q, k),
            )
        beta_logits = F.linear(hidden_states, b_weight)
        decay_logits = F.linear(hidden_states, a_weight)
        if cuda_gates:
            from .gdn_cuda_gates import native_gates
            beta, decay = native_gates(decay_logits, beta_logits, attn.A_log, attn.dt_bias)
        else:
            beta = beta_logits.sigmoid()
            decay = -attn.A_log.float().exp() * F.softplus(
                decay_logits.float() + attn.dt_bias
            )
        if attn.allow_neg_eigval:
            beta = beta * 2.0
        output, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=decay,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
        output_gate = rearrange(
            F.linear(hidden_states, g_weight),
            "... (h d) -> ... h d",
            d=attn.head_v_dim,
        )
        output = attn.o_norm(output, output_gate)
        output = rearrange(output, "b t h d -> b t (h d)")
        return F.linear(output, self.folded_o_weight) if fold_output else attn.o_proj(output)

    def _packed_gdn_chunk(self, hidden_states, states, output_final_state):
        attn = self.attn
        key_dim, value_dim = attn.key_dim, attn.value_dim
        qkvg = F.linear(hidden_states, self.packed_qkvg_weight)
        fused_conv = self.gdn_inference_impl in {
            "packed_fused", "packed_fused_stream"
        }
        if fused_conv:
            conv_qkv, recurrent_state = states
            qkv = qkvg[..., :2 * key_dim + value_dim]
            output_gate = qkvg[..., 2 * key_dim + value_dim:]
            qkv, conv_qkv = causal_conv1d(
                x=qkv,
                weight=self.packed_qkv_conv_weight,
                bias=self.packed_qkv_conv_bias,
                initial_state=conv_qkv,
                output_final_state=output_final_state,
                activation=self.gdn_conv_activation,
                backend=self.gdn_conv_backend,
            )
            q, k, v = torch.split(qkv, [key_dim, key_dim, value_dim], dim=-1)
        else:
            q, k, v, output_gate = torch.split(
                qkvg, [key_dim, key_dim, value_dim, value_dim], dim=-1
            )
            conv_q, conv_k, conv_v, recurrent_state = states
            q, conv_q = attn.q_conv1d(
                x=q, cache=conv_q, output_final_state=output_final_state
            )
            k, conv_k = attn.k_conv1d(
                x=k, cache=conv_k, output_final_state=output_final_state
            )
            v, conv_v = attn.v_conv1d(
                x=v, cache=conv_v, output_final_state=output_final_state
            )

        q, k = map(
            lambda tensor: rearrange(
                tensor, "... (h d) -> ... h d", d=attn.head_k_dim
            ),
            (q, k),
        )
        v = rearrange(v, "... (h d) -> ... h d", d=attn.head_v_dim)
        if attn.num_v_heads > attn.num_heads:
            q, k = map(
                lambda tensor: repeat(
                    tensor,
                    "... h d -> ... (h g) d",
                    g=attn.num_v_heads // attn.num_heads,
                ),
                (q, k),
            )

        ba = F.linear(hidden_states, self.packed_ab_weight)
        beta_logits, decay_logits = ba.split(attn.num_v_heads, dim=-1)
        beta = beta_logits.sigmoid()
        if attn.allow_neg_eigval:
            beta = beta * 2.0
        decay = -attn.A_log.float().exp() * F.softplus(
            decay_logits.float() + attn.dt_bias
        )
        output, recurrent_state = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=decay,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=True,
        )
        output_gate = rearrange(
            output_gate, "... (h d) -> ... h d", d=attn.head_v_dim
        )
        output = attn.o_norm(output, output_gate)
        output = rearrange(output, "b t h d -> b t (h d)")
        output = attn.o_proj(output)
        if fused_conv:
            return output, (conv_qkv, recurrent_state)
        return output, (conv_q, conv_k, conv_v, recurrent_state)

    def _forward_packed_inference(self, hidden_states):
        self._materialize_packed_projections()
        if not self.attn.use_short_conv or not self.attn.use_gate:
            raise RuntimeError(
                "Packed GDN inference requires use_short_conv=True and use_gate=True"
            )
        fused_conv = self.gdn_inference_impl in {
            "packed_fused", "packed_fused_stream"
        }
        if self.gdn_inference_impl in {"packed", "packed_fused"}:
            initial_states = (None, None) if fused_conv else (None, None, None, None)
            output, _ = self._packed_gdn_chunk(
                hidden_states, initial_states, output_final_state=False
            )
            return output

        states = (None, None) if fused_conv else (None, None, None, None)
        outputs = []
        chunk_size = self.gdn_stream_chunk_size
        for start in range(0, hidden_states.shape[1], chunk_size):
            chunk = hidden_states[:, start:start + chunk_size]
            output, states = self._packed_gdn_chunk(
                chunk, states, output_final_state=True
            )
            outputs.append(output)
        return torch.cat(outputs, dim=1)

    def inference_optimization_status(self):
        return {
            "implementation": self.gdn_inference_impl,
            "packed": self.packed_qkvg_weight is not None,
            "native_cuda_gates": self.gdn_inference_impl == "cuda_gates",
            "adapter_folded": self.folded_q_weight is not None,
            "output_adapter_folded": self.folded_o_weight is not None,
            "preserves_rounding": None if self.gdn_inference_impl == "cuda_gates" else (self.folded_o_weight is None),
            "fused_qkv_convolution": self.packed_qkv_conv_weight is not None,
            "stream_chunk_size": self.gdn_stream_chunk_size,
            "release_unpacked_projections": self.gdn_release_unpacked_projections,
        }

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # FLA computes these recurrence parameters in fp32. Keep their storage
        # in fp32 as well so low-LR updates are not rounded away in bf16.
        for name in ("A_log", "dt_bias"):
            parameter = getattr(self.attn, name, None)
            if parameter is not None:
                parameter.data = parameter.data.float()
        return result

    def forward(self, x, freqs):
        original_dtype = x.dtype
        use_optimized_inference = (
            self.gdn_inference_impl != "fla"
            and not self.training
            and not torch.is_grad_enabled()
        )
        use_folded_inference = (
            use_optimized_inference
            and self.gdn_inference_impl in {"fla_folded", "fla_folded_out"}
        )
        if use_folded_inference and self.gdn_inference_impl == "fla_folded":
            self._materialize_folded_projections()
            out = self._folded_gdn_chunk(x, fold_input=True)
        else:
            if self.hidden_input_adapter is not None:
                x = F.linear(
                    x,
                    self.hidden_input_adapter.to(dtype=x.dtype, device=x.device),
                )
            if use_folded_inference:
                self._materialize_folded_output_projection()
                out = self._folded_gdn_chunk(x, fold_input=False)
            elif use_optimized_inference and self.gdn_inference_impl == "cuda_gates":
                if x.shape[1] <= 64:
                    # FLA selects the recurrent kernel for short sequences.
                    out = self.attn(x)
                else:
                    out = self._folded_gdn_chunk(x, fold_input=False, fold_output=False, cuda_gates=True)
            elif use_optimized_inference:
                out = self._forward_packed_inference(x)
            else:
                if self.attn.q_proj is None or self.attn.o_proj is None:
                    raise RuntimeError(
                        "Unpacked GDN projections were released for inference; "
                        "this module can no longer run the training/FLA path."
                    )
                out = self.attn(x)

            if isinstance(out, tuple):
                out = out[0]
            if self.hidden_output_adapter is not None and not use_folded_inference:
                out = F.linear(
                    out,
                    self.hidden_output_adapter.to(dtype=out.dtype, device=out.device),
                )

        return out.to(original_dtype)

class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual


class ScaleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.full((1,), 0.1))

    def forward(self, x):
        return x * self.scale


class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()
        self.memory_attn = CrossAttention(dim, num_heads, eps, has_image_input=False)
        self.norm_mem = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.mem_scale = ScaleLayer()

    def forward(self, x, context, t_mod, freqs, memory_tokens=None, **kwargs):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)
        if memory_tokens is not None:
            x = x + self.mem_scale(self.memory_attn(self.norm_mem(x), memory_tokens))
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t_embed_dtype = self.time_embedding[0].weight.dtype
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(t_embed_dtype))
        t_proj_dtype = self.time_projection[1].weight.dtype
        t_mod = self.time_projection(t.to(t_proj_dtype)).unflatten(1, (6, self.dim)).to(x.dtype)
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs, kwargs.get("memory_tokens", None),
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs, kwargs.get("memory_tokens", None),
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs, memory_tokens=kwargs.get("memory_tokens", None))

        x = self.head(x, t.to(x.dtype))
        x = self.unpatchify(x, (f, h, w))
        return x
