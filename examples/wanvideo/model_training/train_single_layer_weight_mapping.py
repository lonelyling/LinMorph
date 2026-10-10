import argparse
import math
import os
import random
import sys
from datetime import timedelta

import accelerate
import torch
import torch.nn.functional as F
from accelerate import InitProcessGroupKwargs
from accelerate.utils import set_seed

from diffsynth.core import load_state_dict
from diffsynth.core.data.uni_dataset import UnifiedDataset
from diffsynth.diffusion import DiffusionTrainingModule
from diffsynth.diffusion.logger import ModelLogger
from diffsynth.diffusion.runner import launch_training_task
from diffsynth.models.fla_compat import patch_fla_l2norm_for_metax
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
SHAPE_ADAPTER_NAMES = ("direct", "repeat")
SHAPE_ADAPTER_IDS = {name: index for index, name in enumerate(SHAPE_ADAPTER_NAMES)}
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
        full_rank_residual=False,
        mixing_init="manual_copy",
        low_rank_init_std=0.02,
        gdn_expand_v=1.0,
        gdn_head_dim=0,
        shape_adapter="direct",
        gdn_use_short_conv=True,
        identity_short_conv=True,
        map_auxiliary_parameters=False,
        capture_random_initialization=False,
        seed=42,
    ):
        super().__init__()
        patch_fla_l2norm_for_metax()
        from fla.layers.gated_deltanet import GatedDeltaNet

        dim = int(source_attention.dim)
        num_heads = int(source_attention.num_heads)
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        source_head_dim = dim // num_heads
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
        if shape_adapter == "direct":
            if not math.isclose(gdn_expand_v, 1.0):
                raise ValueError(
                    "The direct 4-way mixing experiment requires --gdn_expand_v 1.0 so "
                    "q/k/v/o/g all have the same matrix shape as the softmax projections."
                )
            if head_dim * num_heads != dim:
                raise ValueError(
                    f"gdn_head_dim * num_heads must equal dim for direct mixing, got "
                    f"{head_dim} * {num_heads} != {dim}"
                )
        else:
            for target_name, target_dim in (
                ("head_dim", head_dim),
                ("head_v_dim", head_v_dim),
            ):
                if target_dim < source_head_dim or target_dim % source_head_dim != 0:
                    raise ValueError(
                        "The repeat adapter requires integer per-head widening: "
                        f"{target_name}={target_dim}, source_head_dim={source_head_dim}"
                    )
        self.shape_adapter = shape_adapter
        self.dim = dim
        self.num_heads = num_heads
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
            len(SOURCE_NAMES), num_heads, source_head_dim, dim
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
            torch.tensor(5, dtype=torch.int64),
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
            "mapping_shape_adapter",
            torch.tensor(SHAPE_ADAPTER_IDS[shape_adapter], dtype=torch.int64),
            persistent=True,
        )

        cpu_rng_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        try:
            self.gdn = GatedDeltaNet(
                hidden_size=dim,
                expand_v=gdn_expand_v,
                head_dim=head_dim,
                num_heads=num_heads,
                num_v_heads=num_heads,
                mode="chunk",
                use_gate=True,
                use_short_conv=gdn_use_short_conv,
                layer_idx=block_index,
                norm_eps=source_attention.norm_q.eps,
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

        self.mixing = torch.nn.Parameter(
            _manual_copy_matrix(mixing_init).to(device=source_weights.device)
        )
        self.map_auxiliary_parameters = bool(map_auxiliary_parameters)
        self.register_buffer(
            "mapping_auxiliary_enabled",
            torch.tensor(int(self.map_auxiliary_parameters), dtype=torch.int64),
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
        self.auxiliary_mixing.requires_grad_(self.map_auxiliary_parameters)
        self.low_rank = int(low_rank)
        if self.low_rank < 0:
            raise ValueError(f"low_rank must be non-negative, got {self.low_rank}")
        self.full_rank_residual_enabled = bool(full_rank_residual)
        if self.low_rank > 0 and self.full_rank_residual_enabled:
            raise ValueError(
                "low_rank and full_rank_residual are mutually exclusive"
            )
        self.register_buffer(
            "mapping_low_rank",
            torch.tensor(self.low_rank, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "mapping_full_rank_residual",
            torch.tensor(
                int(self.full_rank_residual_enabled),
                dtype=torch.int64,
            ),
            persistent=True,
        )
        self.low_rank_scale = 1.0
        self.low_rank_u = torch.nn.ParameterDict()
        self.low_rank_v = torch.nn.ParameterDict()
        self.full_rank_residual = torch.nn.ParameterDict()
        if self.low_rank > 0 or self.full_rank_residual_enabled:
            gdn_parameters = dict(self.gdn.named_parameters())
            target_shapes = {
                name: tuple(gdn_parameters[parameter_name].shape)
                for name, parameter_name in TARGET_PARAMETER_NAMES.items()
            }
        if self.low_rank > 0:
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
                        u.to(device=source_weights.device)
                    )
                    self.low_rank_v[name] = torch.nn.Parameter(
                        v.to(device=source_weights.device)
                    )
            finally:
                torch.random.set_rng_state(cpu_rng_state)
        if self.full_rank_residual_enabled:
            for name, shape in target_shapes.items():
                self.full_rank_residual[name] = torch.nn.Parameter(
                    torch.zeros(
                        shape,
                        dtype=torch.float32,
                        device=source_weights.device,
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
        if self.full_rank_residual_enabled != other.full_rank_residual_enabled:
            raise ValueError(
                "Cannot share mapping parameters across different full-rank "
                "residual settings"
            )
        if self.map_auxiliary_parameters != other.map_auxiliary_parameters:
            raise ValueError("Auxiliary mapping settings must match before sharing")
        self.mixing = other.mixing
        for name in self.auxiliary_mixing:
            self.auxiliary_mixing[name] = other.auxiliary_mixing[name]
        for name in self.low_rank_u:
            self.low_rank_u[name] = other.low_rank_u[name]
            self.low_rank_v[name] = other.low_rank_v[name]
        for name in self.full_rank_residual:
            self.full_rank_residual[name] = other.full_rank_residual[name]

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
        if target_head_dim == self.source_head_dim:
            return weight
        repeat = target_head_dim // self.source_head_dim
        return (
            weight.reshape(
                self.num_heads,
                self.source_head_dim,
                weight.shape[-1],
            )
            .repeat(1, repeat, 1)
            .reshape(self.num_heads * target_head_dim, weight.shape[-1])
        )

    def _repeat_columns_per_head(self, weight, target_head_dim):
        if target_head_dim == self.source_head_dim:
            return weight
        repeat = target_head_dim // self.source_head_dim
        widened = (
            weight.reshape(
                weight.shape[0],
                self.num_heads,
                self.source_head_dim,
            )
            .repeat(1, 1, repeat)
            .reshape(weight.shape[0], self.num_heads * target_head_dim)
        )
        # Repeated value channels carry identical activations. Averaging their
        # output columns preserves the original projection at initialization.
        return widened / repeat

    def _repeat_channels_per_head(self, values, target_head_dim):
        if target_head_dim == self.source_head_dim:
            return values
        repeat = target_head_dim // self.source_head_dim
        return (
            values.reshape(self.num_heads, self.source_head_dim)
            .repeat(1, repeat)
            .reshape(self.num_heads * target_head_dim)
        )

    def _adapt_projection_weight(self, target_name, weight):
        if self.shape_adapter == "direct" or target_name == "beta":
            return weight
        if target_name in ("q", "k"):
            return self._repeat_rows_per_head(weight, self.target_head_dim)
        if target_name in ("v", "g"):
            return self._repeat_rows_per_head(weight, self.target_head_v_dim)
        if target_name == "o":
            return self._repeat_columns_per_head(weight, self.target_head_v_dim)
        raise KeyError(f"Unsupported mapping target: {target_name}")

    def _validate_target_shapes(self):
        parameters = dict(self.gdn.named_parameters())
        for target_name, parameter_name in TARGET_PARAMETER_NAMES.items():
            if parameter_name not in parameters:
                raise ValueError(
                    f"GDN does not expose required parameter {parameter_name!r}"
                )
            actual = tuple(parameters[parameter_name].shape)
            source = (
                self.source_weights_by_head[0]
                if target_name == "beta"
                else self.source_weights[0]
            )
            expected = tuple(
                self._adapt_projection_weight(target_name, source).shape
            )
            if actual != expected:
                raise ValueError(
                    f"Cannot map {target_name}: GDN {parameter_name} has shape {actual}, expected {expected}"
                )
        auxiliary_expected_shapes = {
            "a": self.source_weights_by_head.shape[1:],
            "A_log": self.source_head_statistics.shape[1:],
            "dt_bias": self.source_head_statistics.shape[1:],
            "q_conv1d": (
                self.num_heads * self.target_head_dim,
                1,
                self.auxiliary_mixing["q_conv1d"].shape[0],
            ),
            "k_conv1d": (
                self.num_heads * self.target_head_dim,
                1,
                self.auxiliary_mixing["k_conv1d"].shape[0],
            ),
            "v_conv1d": (
                self.num_heads * self.target_head_v_dim,
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
        # Keep the trainable factors and optimizer state in fp32, but form the
        # large residual matrix in the frozen source-weight dtype.
        u = self.low_rank_u[target_name].to(
            device=self.source_weights.device,
            dtype=self.source_weights.dtype,
        )
        v = self.low_rank_v[target_name].to(
            device=self.source_weights.device,
            dtype=self.source_weights.dtype,
        )
        return u @ v.transpose(0, 1)

    def _dense_residual(self, target_name):
        if not self.full_rank_residual_enabled:
            return None
        return self.full_rank_residual[target_name].to(
            device=self.source_weights.device,
            dtype=self.source_weights.dtype,
        )

    @staticmethod
    def _mixed_weight(coefficients, source, target):
        source = source.to(device=target.device, dtype=target.dtype)
        coefficients = coefficients.to(device=target.device, dtype=target.dtype)
        return torch.tensordot(coefficients, source, dims=([-1], [0]))

    def _materialized_auxiliary_parameters(self):
        parameters = dict(self.gdn.named_parameters())
        mapped = {}

        a_base = parameters[AUXILIARY_TARGET_PARAMETER_NAMES["a"]]
        mapped[AUXILIARY_TARGET_PARAMETER_NAMES["a"]] = a_base + self._mixed_weight(
            self.auxiliary_mixing["a"], self.source_weights_by_head, a_base
        )

        for target_name in ("A_log", "dt_bias"):
            parameter_name = AUXILIARY_TARGET_PARAMETER_NAMES[target_name]
            base = parameters[parameter_name]
            mapped[parameter_name] = base + self._mixed_weight(
                self.auxiliary_mixing[target_name],
                self.source_head_statistics,
                base,
            )

        for target_name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            parameter_name = AUXILIARY_TARGET_PARAMETER_NAMES[target_name]
            base = parameters[parameter_name]
            # [kernel, source] x [source, channel] -> [channel, 1, kernel]
            channel_values = self._mixed_weight(
                self.auxiliary_mixing[target_name],
                self.source_channel_statistics,
                base,
            )
            target_head_dim = (
                self.target_head_v_dim
                if target_name == "v_conv1d"
                else self.target_head_dim
            )
            channel_values = torch.stack(
                [
                    self._repeat_channels_per_head(values, target_head_dim)
                    for values in channel_values
                ],
                dim=0,
            )
            residual = channel_values.transpose(0, 1).unsqueeze(1)
            mapped[parameter_name] = base + residual

        norm_name = AUXILIARY_TARGET_PARAMETER_NAMES["o_norm"]
        norm_base = parameters[norm_name]
        norm_residual = self._mixed_weight(
            self.auxiliary_mixing["o_norm"],
            self.source_norm_statistics,
            norm_base,
        )
        if self.target_head_v_dim != self.source_head_dim:
            repeat = self.target_head_v_dim // self.source_head_dim
            norm_residual = norm_residual.repeat(repeat)
        mapped[norm_name] = norm_base + norm_residual
        return mapped

    def materialized_parameters(self):
        mapped = {}
        for target_index, target_name in enumerate(TARGET_NAMES):
            source = (
                self.source_weights_by_head
                if target_name == "beta"
                else self.source_weights
            )
            coefficients = self.mixing[target_index].to(
                dtype=source.dtype, device=source.device
            )
            weight = torch.tensordot(coefficients, source, dims=([0], [0]))
            weight = self._adapt_projection_weight(target_name, weight)
            residual = self._low_rank_residual(target_name)
            if residual is not None:
                weight = weight + residual.to(weight.dtype) * self.low_rank_scale
            dense_residual = self._dense_residual(target_name)
            if dense_residual is not None:
                weight = weight + dense_residual.to(weight.dtype)
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

        output = functional_call(
            self.gdn,
            mapped_parameters,
            (x,),
            strict=False,
        )
        if isinstance(output, tuple):
            output = output[0]
        self.last_output = output
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
        full_rank_residual=False,
        low_rank_init_std=0.02,
        gdn_expand_v=1.0,
        gdn_head_dim=0,
        shape_adapter="direct",
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
            full_rank_residual=full_rank_residual,
            mixing_init=mixing_init,
            low_rank_init_std=low_rank_init_std,
            gdn_expand_v=gdn_expand_v,
            gdn_head_dim=gdn_head_dim,
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
                full_rank_residual=full_rank_residual,
                mixing_init=mixing_init,
                low_rank_init_std=low_rank_init_std,
                gdn_expand_v=gdn_expand_v,
                gdn_head_dim=gdn_head_dim,
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
        self.train_base_mapping = bool(train_base_mapping)
        if (
            not self.train_base_mapping
            and low_rank <= 0
            and not full_rank_residual
        ):
            raise ValueError(
                "--train_base_mapping false requires --low_rank greater than "
                "zero or --full_rank_residual true"
            )
        self.mapped_attention.mixing.requires_grad_(self.train_base_mapping)
        self.mapped_attention.auxiliary_mixing.requires_grad_(
            bool(map_auxiliary_parameters) and self.train_base_mapping
        )
        for parameter in self.mapped_attention.low_rank_u.parameters():
            parameter.requires_grad_(True)
        for parameter in self.mapped_attention.low_rank_v.parameters():
            parameter.requires_grad_(True)
        for parameter in self.mapped_attention.full_rank_residual.parameters():
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
        self.attention_loss_weight = float(attention_loss_weight)
        self.block_loss_weight = float(block_loss_weight)
        self.mapping_regularization = float(mapping_regularization)
        self.normalized_mse = bool(normalized_mse)
        self.mapping_log_interval = max(1, int(mapping_log_interval))
        self.register_buffer(
            "mixing_initial_value", _manual_copy_matrix(mixing_init), persistent=False
        )
        self.latest_loss_metrics = {}
        self._printed_initial_metrics = False
        self._mapping_steps = 0

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
            f"low-rank residual rank: {self.mapped_attention.low_rank}; "
            "full-rank residual: "
            f"{self.mapped_attention.full_rank_residual_enabled}"
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
        if version > 5:
            raise ValueError(
                f"Unsupported mapping checkpoint format {version}"
            )
        checkpoint_expand_v = normalized.pop("mapping_gdn_expand_v", None)
        checkpoint_head_dim = normalized.pop("mapping_gdn_head_dim", None)
        checkpoint_adapter = normalized.pop("mapping_shape_adapter", None)
        if version >= 4:
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
            if actual != expected:
                raise ValueError(
                    "Mapping checkpoint architecture mismatch: "
                    f"checkpoint={actual}, current={expected}"
                )
        elif self.mapped_attention.shape_adapter != "direct":
            raise ValueError(
                "Legacy mapping checkpoints have no architecture metadata and can "
                "only initialize the direct square mapper."
            )
        checkpoint_low_rank_value = normalized.get("mapping_low_rank")
        checkpoint_low_rank = (
            int(checkpoint_low_rank_value.item())
            if checkpoint_low_rank_value is not None
            else (
                next(
                    (
                        int(value.shape[1])
                        for name, value in normalized.items()
                        if name.startswith("low_rank_u.")
                    ),
                    0,
                )
            )
        )
        checkpoint_full_rank_value = normalized.get("mapping_full_rank_residual")
        checkpoint_full_rank_keys = {
            name
            for name in normalized
            if name.startswith("full_rank_residual.")
        }
        checkpoint_has_full_rank_tensors = bool(checkpoint_full_rank_keys)
        checkpoint_full_rank = (
            bool(int(checkpoint_full_rank_value.item()))
            if checkpoint_full_rank_value is not None
            else checkpoint_has_full_rank_tensors
        )
        if (
            checkpoint_full_rank_value is not None
            and checkpoint_full_rank != checkpoint_has_full_rank_tensors
        ):
            raise ValueError(
                "Mapping checkpoint full-rank metadata and tensor contents disagree"
            )
        if checkpoint_low_rank > 0 and checkpoint_full_rank:
            raise ValueError(
                "Mapping checkpoint contains both low-rank and full-rank residuals"
            )
        if (
            checkpoint_low_rank > 0
            and checkpoint_low_rank != self.mapped_attention.low_rank
        ):
            raise ValueError(
                "Mapping checkpoint low-rank residual does not match this run: "
                f"checkpoint={checkpoint_low_rank}, "
                f"current={self.mapped_attention.low_rank}"
            )
        if (
            checkpoint_full_rank
            and not self.mapped_attention.full_rank_residual_enabled
        ):
            raise ValueError(
                "Mapping checkpoint contains a full-rank residual, but this run "
                "did not enable --full_rank_residual"
            )
        if checkpoint_full_rank:
            expected_full_rank_keys = {
                f"full_rank_residual.{name}"
                for name in self.mapped_attention.full_rank_residual
            }
            if checkpoint_full_rank_keys != expected_full_rank_keys:
                missing = sorted(expected_full_rank_keys - checkpoint_full_rank_keys)
                unexpected = sorted(
                    checkpoint_full_rank_keys - expected_full_rank_keys
                )
                raise ValueError(
                    "Mapping checkpoint has incomplete full-rank residual tensors: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        for metadata_name in (
            "mapping_auxiliary_enabled",
            "mapping_low_rank",
            "mapping_full_rank_residual",
        ):
            normalized.pop(metadata_name, None)
        if "mixing" not in normalized:
            raise ValueError(
                f"Mapping checkpoint {checkpoint_path} does not contain mixing coefficients"
            )
        materialized_gdn_keys = [
            name for name in normalized if name.startswith("gdn.")
        ]
        normalized = {
            name: value
            for name, value in normalized.items()
            if not name.startswith("gdn.")
        }
        if materialized_gdn_keys:
            print(
                "Ignoring materialized GDN tensors while initializing mapping "
                f"coefficients: {len(materialized_gdn_keys)} tensors."
            )
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

    def _capture_teacher_block_features(self, model_kwargs):
        captured = {index: {} for index in self.block_indices}
        attention_handles = []
        block_handles = []
        last_block_index = self.block_indices[-1]

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

        for layer_index in self.block_indices:
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
                self.pipe.model_fn(
                    dit=self.teacher_dit,
                    use_gradient_checkpointing=False,
                    use_gradient_checkpointing_offload=False,
                    memory_tokens=None,
                    **model_kwargs,
                )
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
        teacher_by_block = self._capture_teacher_block_features(model_kwargs)
        attention_losses = []
        block_losses = []
        per_block_metrics = {}
        for layer_index in self.block_indices:
            teacher = teacher_by_block[layer_index]
            mapped_attention, mapped_block = self._forward_mapped_block(
                teacher["block_inputs"], block_index=layer_index
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
            regularization_deltas.extend(
                parameter.reshape(-1)
                for parameter in self.mapped_attention.auxiliary_mixing.values()
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
            if self.mapped_attention.map_auxiliary_parameters:
                auxiliary = {
                    name: parameter.detach().float().cpu().tolist()
                    for name, parameter in self.mapped_attention.auxiliary_mixing.items()
                }
                print(f"Auxiliary mixing at local step {self._mapping_steps}: {auxiliary}")
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
            if self.mapped_attention.full_rank_residual_enabled:
                dense_norms = {
                    name: parameter.detach().float().norm().item()
                    for name, parameter in (
                        self.mapped_attention.full_rank_residual.items()
                    )
                }
                print(
                    f"Full-rank residual norms at local step {self._mapping_steps}: "
                    f"{dense_norms}"
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
        inputs_shared["memory_tokens"] = None
        return self.weight_mapping_loss(inputs_shared, inputs_posi)


class WeightMappingLogger(ModelLogger):
    """Save only T/GDN state, never gather the frozen 5B teacher checkpoint."""

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
                    or name.startswith("auxiliary_mixing.")
                    or name.startswith("low_rank_u.")
                    or name.startswith("low_rank_v.")
                    or name.startswith("full_rank_residual.")
                }
                for metadata_name in (
                    "selected_block_index",
                    "selected_block_indices",
                    "mapping_format_version",
                    "mapping_gdn_expand_v",
                    "mapping_gdn_head_dim",
                    "mapping_shape_adapter",
                    "mapping_auxiliary_enabled",
                    "mapping_low_rank",
                    "mapping_full_rank_residual",
                ):
                    mapper_state[metadata_name] = full_mapper_state[metadata_name]
            # DeepSpeed may expose parameters as views of flattened storage. Clone
            # the compact mapper state so safetensors preserves every coefficient.
            mapper_state = {
                name: value.detach().clone() for name, value in mapper_state.items()
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
        "--full_rank_residual",
        type=_parse_bool,
        default=_env_bool("MAPPING_FULL_RANK_RESIDUAL", False),
        help=(
            "Add one trainable dense residual matrix to every mapped q/k/v/o/g/b "
            "weight. This is mutually exclusive with --low_rank."
        ),
    )
    parser.add_argument(
        "--low_rank_init_std",
        type=float,
        default=float(os.environ.get("LOW_RANK_INIT_STD", 0.02)),
    )
    parser.add_argument(
        "--gdn_expand_v",
        type=float,
        default=float(os.environ.get("MAPPING_GDN_EXPAND_V", 1.0)),
    )
    parser.add_argument(
        "--gdn_head_dim",
        type=int,
        default=int(os.environ.get("MAPPING_GDN_HEAD_DIM", 0)),
    )
    parser.add_argument(
        "--mapping_shape_adapter",
        choices=SHAPE_ADAPTER_NAMES,
        default=os.environ.get("MAPPING_SHAPE_ADAPTER", "direct"),
        help=(
            "How square Softmax projections are adapted to GDN shapes. "
            "'direct' requires square GDN projections; 'repeat' performs "
            "per-head function-preserving widening before an optional residual."
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
            "controlled residual-only experiment."
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
        full_rank_residual=args.full_rank_residual,
        low_rank_init_std=args.low_rank_init_std,
        gdn_expand_v=args.gdn_expand_v,
        gdn_head_dim=args.gdn_head_dim,
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
    )
    model_logger = WeightMappingLogger(args.output_path)
    launch_training_task(accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    main()
