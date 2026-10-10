"""Wan sampling defaults and the official UniPC adapter."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_UNIPC_SOURCE = (
    PROJECT_ROOT / ".cache/vbench/official_protocol/fm_solvers_unipc.py"
)


DEFAULT_UNIPC_DEPS = PROJECT_ROOT / ".deps/wan_unipc"


DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def load_unipc_class(source_path: Path, deps_path: Path) -> type:
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Official Wan UniPC source is missing: {source_path}\n"
            "Run: bash scripts/evaluation/vbench/setup_vbench_standard.sh"
        )
    if not deps_path.is_dir():
        raise FileNotFoundError(
            f"Pinned diffusers dependency is missing: {deps_path}\n"
            "Run: bash scripts/evaluation/vbench/setup_vbench_standard.sh"
        )
    sys.path.insert(0, str(deps_path))
    try:
        import diffusers
    except Exception as exc:
        raise RuntimeError(f"Cannot import pinned diffusers from {deps_path}: {exc}") from exc
    if diffusers.__version__ != "0.31.0":
        raise RuntimeError(f"Official Wan sampler requires diffusers 0.31.0, got {diffusers.__version__}")
    spec = importlib.util.spec_from_file_location("wan_official_flow_unipc", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load official Wan UniPC source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FlowUniPCMultistepScheduler


class DiffSynthUniPCAdapter:
    """Adapt Wan's official flow-UniPC API to DiffSynth's scheduler contract."""

    training = False

    def __init__(self, scheduler_class: type):
        self.scheduler_class = scheduler_class
        self.inner = None
        self.timesteps = None

    def set_timesteps(
        self,
        num_inference_steps: int,
        denoising_strength: float = 1.0,
        shift: float | None = None,
        **_: Any,
    ) -> None:
        if denoising_strength != 1.0:
            raise ValueError("Official text-to-video UniPC adapter only supports denoising_strength=1")
        self.inner = self.scheduler_class(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False,
        )
        self.inner.set_timesteps(num_inference_steps, device="cpu", shift=shift)
        self.timesteps = self.inner.timesteps

    def step(self, model_output: Any, timestep: Any, sample: Any, **_: Any) -> Any:
        if self.inner is None:
            raise RuntimeError("set_timesteps must be called before step")
        return self.inner.step(model_output, timestep, sample, return_dict=False)[0]

