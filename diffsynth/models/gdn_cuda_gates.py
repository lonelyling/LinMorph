"""Experimental native CUDA gates; validate against original FLA before use."""
from functools import lru_cache
import hashlib
from pathlib import Path
import torch


@lru_cache(maxsize=1)
def load_extension():
    from torch.utils.cpp_extension import load
    source = Path(__file__).parent/'csrc/gdn_gates.cu'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    return load(
        name='diffsynth_gdn_gates_'+digest,
        sources=[str(source)],
        extra_cflags=['-O3'],
        extra_cuda_cflags=['-O3', '--fmad=false', '--ftz=false',
                          '-gencode=arch=compute_80,code=sm_80'],
        verbose=False,
    )


def native_gates(a, b, a_log, dt_bias):
    if torch.is_grad_enabled() and any(t.requires_grad for t in (a,b,a_log,dt_bias)):
        raise RuntimeError('Native GDN gates are inference-only; autograd is unsupported')
    return load_extension().forward(a,b,a_log,dt_bias)
