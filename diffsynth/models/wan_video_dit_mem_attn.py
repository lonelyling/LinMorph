import re
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .wan_video_camera_controller import SimpleAdapter
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

    def forward(self, q, k, v, require_attention_map=False):
        if not require_attention_map:
            x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
            return x
        else:
            # 计算attention map
            B, S, _ = q.shape
            d_k = q.shape[-1] // self.num_heads
            q_ = rearrange(q, "b s1 (n d) -> b n s1 d", n=self.num_heads)
            k_ = rearrange(k, "b s2 (n d) -> b n s2 d", n=self.num_heads)
            x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)

            attn_weights = torch.matmul(q_, k_.transpose(-2, -1)) / math.sqrt(d_k)  # [B, num_heads, S, S]
            attn_weights = F.softmax(attn_weights, dim=-1)
            return x, attn_weights


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
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
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


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

        # ========attention map存储========
        self.store_attn_map = False  # 统一使用这个名称
        self.attn_map_buffer = None  # 存储attention map的buffer
        self.key_token_indices = None  # 关键物体token索引字典
        self.spatial_shape = None  # 存储空间维度信息 (F, H, W)

    def enable_attn_tracking(self, key_token_indices=None, spatial_shape=None):
        """启用attention map追踪

        Args:
            key_token_indices: Dict[str, List[int]]
                关键物体的token索引，例如 {"白猫": [6, 7]}
            spatial_shape: Tuple[int, int, int] or None
                潜空间的空间维度 (F, H, W)，用于分割每一帧
        """
        self.store_attn_map = True
        self.key_token_indices = key_token_indices
        self.spatial_shape = spatial_shape
        self.attn_map_buffer = []
        
    def disable_attn_tracking(self):
        """禁用tracking并释放内存"""
        self.store_attn_map = False
        self.attn_map_buffer = []
        self.key_token_indices = None
        self.spatial_shape = None
    
    def get_attn_maps(self):
        """获取收集的attention maps"""
        return self.attn_map_buffer

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # 分离图像和文本context（如果有图像输入）
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y

        # Q, K, V投影
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)

        # 根据tracking模式选择attention计算方式
        if self.store_attn_map:
            # Tracking模式：获取attention weights
            x_out, attn_weights = self.attn(q, k, v, require_attention_map=True)

            # 存储attention map
            if self.key_token_indices is not None:
                # 聚合所有头 [B, num_heads, seq_q, seq_k] -> [B, seq_q, seq_k]
                attn_mean = attn_weights.mean(dim=1)

                # 为每个关键物体提取attention
                attn_data = {}
                for obj, indices in self.key_token_indices.items():
                    # indices是列表，例如[6, 7]
                    if isinstance(indices, list):
                        # 提取多个token位置的attention并聚合
                        attn_map = attn_mean[:, :, indices].mean(dim=-1)  # [B, seq_q]
                    else:
                        attn_map = attn_mean[:, :, indices]  # [B, seq_q]

                    # 如果提供了spatial_shape，按帧分割
                    F, H, W = self.spatial_shape

                    ph, pw = 2, 2  # Patch height and width

                    try:
                        attn_map_reshaped = rearrange(
                            attn_map, 
                            'b (f h w) -> b f h w', 
                            f=F, h=H//ph, w=W//pw
                        )
                    except Exception as e:
                        print(f"维度转换出错，当前 attn_map shape: {attn_map.shape}, 目标 F: {F}")
                        raise e
                    

                    attn_data[obj] = {
                        'full': attn_map.detach().cpu(),  # [B, F*H*W]
                        'spatial_shape': (F, H, W)
                    }

                # 只保留最后一个 timestep 的 attention map，保持 attn_map_buffer 为 list
                self.attn_map_buffer = attn_data
        else:
            # 正常模式：高效的flash attention
            x_out = self.attn(q, k, v)

        # 处理图像输入（如果有）
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            img_attn = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x_out = x_out + img_attn

        # 输出投影
        return self.o(x_out)



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

        # ========memory attention========
        self.memory_attn = CrossAttention(dim, num_heads, eps, has_image_input=False)
        self.norm_mem = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        # 在 Block 中使用
        self.mem_scale = ScaleLayer()
        # ================================

    def forward(self, x, context, t_mod, freqs, memory_tokens=None):
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
            x = x + self.mem_scale(self.memory_attn(
                self.norm_mem(x), memory_tokens))


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



class PerceiverResampler(nn.Module):
    def __init__(self, dim, num_queries=64, num_heads=8, mult=4):
        super().__init__()
        # 1. 这里的 queries 是可学习的参数，它是模型的“提取器”
        self.queries = nn.Parameter(torch.randn(num_queries, dim))

        # basic sanity checks
        assert dim % num_heads == 0, "PerceiverResampler: `dim` must be divisible by `num_heads`"

        # 2. 注意力机制：用于从原始特征中提取信息
        # `batch_first=True` requires a recent PyTorch; using it keeps shapes (B, S, D)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        # 3. 前馈网络（FFN）用于进一步增强特征
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim)
        )

        # use two LayerNorms (one after each residual) rather than reusing a single one
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x 形状: [batch_size, total_patches, dim] 
        # total_patches 是所有参考图 patch 的总和
        
        b = x.shape[0]
        # 将可学习的 queries 复制到 batch 维度
        queries = self.queries.unsqueeze(0).repeat(b, 1, 1)
        
        # Cross-Attention: 
        # Query = 我们的学习参数
        # Key/Value = 原始图像的 Patch 特征
        # 这一步将成千上万个 patch 压缩到了 num_queries (如 64) 个特征中
        out, _ = self.attn(queries, x, x)

        # 残差连接与 FFN
        out = self.norm1(out + queries)
        out = self.norm2(out + self.ff(out))

        return out

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
        # self.q_former = PerceiverResampler(dim)

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        f, h, w = x.shape[-3:]
    
        # 注意：这里不再执行 flatten 和 transpose
        return x, (f, h, w)

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

                #========================================================
                memory_tokens: Optional[torch.Tensor] = None,
                #========================================================

                **kwargs,
                ):
        t_embed_dtype = self.time_embedding[0].weight.dtype
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(t_embed_dtype))
        t_proj_dtype = self.time_projection[1].weight.dtype
        t_mod = self.time_projection(t.to(t_proj_dtype)).unflatten(1, (6, self.dim)).to(x.dtype)
        context = self.text_embedding(context)
        # process memory tokens through the PerceiverResampler and overwrite
        # `memory_tokens` so downstream blocks receive the resampled tokens
        # memory_tokens = self.q_former(memory_tokens) if memory_tokens is not None else None
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)

        x = x.flatten(2).transpose(1, 2)

        
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
                            x, context, t_mod, freqs, memory_tokens,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs, memory_tokens,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs, memory_tokens=memory_tokens)

        x = self.head(x, t.to(x.dtype))
        x = self.unpatchify(x, (f, h, w))
        return x

        
