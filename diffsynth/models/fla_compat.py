import importlib
import os

import torch


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_PATCH_MARKER = "_diffsynth_metax_l2norm_fallback"


def _is_metax_runtime():
    vendor = os.environ.get("ACCELERATOR_VENDOR", "").strip().lower()
    return vendor == "metax" or bool(getattr(torch.version, "maca", None))


def _torch_l2norm_fwd(x, eps=1e-6, output_dtype=None):
    x_float = x.float()
    denominator = x_float.square().sum(dim=-1, keepdim=True) + eps
    output = x_float * torch.rsqrt(denominator)
    return output.to(dtype=output_dtype or x.dtype)


def _torch_l2norm_bwd(x, dy, eps=1e-5):
    x_float = x.float()
    dy_float = dy.float()
    denominator = x_float.square().sum(dim=-1, keepdim=True) + eps
    reciprocal_norm = torch.rsqrt(denominator)
    projection = (dy_float * x_float).sum(dim=-1, keepdim=True)
    dx = (
        dy_float * reciprocal_norm
        - projection * reciprocal_norm / denominator * x_float
    )
    return dx.to(dtype=x.dtype)


def patch_fla_l2norm_for_metax():
    """Avoid FLA l2norm Triton kernels that exceed MetaX private-memory limits."""
    setting = os.environ.get(
        "DIFFSYNTH_METAX_L2NORM_FALLBACK", "auto"
    ).strip().lower()
    if setting in _FALSE_VALUES:
        return False
    if setting != "auto" and setting not in _TRUE_VALUES:
        raise ValueError(
            "DIFFSYNTH_METAX_L2NORM_FALLBACK must be auto, 0, or 1; "
            f"got {setting!r}"
        )
    if setting == "auto" and not _is_metax_runtime():
        return False

    l2norm_module = importlib.import_module("fla.modules.l2norm")
    if getattr(l2norm_module, _PATCH_MARKER, False):
        return True

    minimum_dim = int(
        os.environ.get("DIFFSYNTH_METAX_L2NORM_FALLBACK_MIN_DIM", "256")
    )
    if minimum_dim < 1:
        raise ValueError(
            "DIFFSYNTH_METAX_L2NORM_FALLBACK_MIN_DIM must be positive, "
            f"got {minimum_dim}"
        )

    original_fwd = l2norm_module.l2norm_fwd
    original_bwd = l2norm_module.l2norm_bwd

    def compatible_fwd(x, eps=1e-6, output_dtype=None):
        if x.shape[-1] >= minimum_dim:
            return _torch_l2norm_fwd(x, eps=eps, output_dtype=output_dtype)
        return original_fwd(x, eps=eps, output_dtype=output_dtype)

    def compatible_bwd(x, dy, eps=1e-5):
        if x.shape[-1] >= minimum_dim:
            return _torch_l2norm_bwd(x, dy, eps=eps)
        return original_bwd(x, dy, eps=eps)

    l2norm_module.l2norm_fwd = compatible_fwd
    l2norm_module.l2norm_bwd = compatible_bwd
    setattr(l2norm_module, _PATCH_MARKER, True)

    # These modules import the functions by value, so update their bound globals
    # in addition to the source module.
    for module_name in (
        "fla.ops.gated_delta_rule.chunk",
        "fla.ops.delta_rule.chunk",
        "fla.ops.delta_rule.fused_recurrent",
    ):
        module = importlib.import_module(module_name)
        module.l2norm_fwd = compatible_fwd
        module.l2norm_bwd = compatible_bwd

    print(
        "[FLA] Enabled MetaX torch l2norm fallback for feature dimensions "
        f">= {minimum_dim}.",
        flush=True,
    )
    return True
