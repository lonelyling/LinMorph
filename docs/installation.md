# Installation

Use Linux, Python 3.10, an NVIDIA CUDA GPU, and a fresh environment. The source environment records PyTorch 2.6.0 + CUDA 12.6, torchvision 0.21.0, Accelerate 1.12.0 and FLA 0.3.2. This is an observed research environment, not a claim that a clean installation has been verified here. Some installed package metadata contains multiple versions; requirements are bounded rather than presented as a complete lockfile.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e .
python -m pip check
```

The distribution is named `linmorph`; it deliberately provides the `diffsynth` import namespace. Do not install a second DiffSynth-Studio checkout in the same environment. Run the repository scripts from the checkout root. The editable package includes `diffsynth` and the shared `linmorph` modules; scripts and training examples are checkout entry points, not installed console commands.

FLA supplies GatedDeltaNet and its Triton kernels. FlashAttention is optional for the core pipeline (PyTorch SDPA fallback exists), but some external evaluation stacks require it. The recorded research FlashAttention version is 2.6.3; compile it only against a compatible CUDA/PyTorch toolchain. Multi-GPU optimizer sharding can use the optional `requirements-train.txt`; the public default uses Accelerate DDP and requires sufficient GPU memory. The original experiments used large-memory accelerators; no minimum VRAM claim is made for the public recipes.

Obtain the original model from [Wan-AI/Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B), preserving its directory layout:

```text
model_weight/Wan2.1-T2V-1.3B/
  config.json
  diffusion_pytorch_model.safetensors
  models_t5_umt5-xxl-enc-bf16.pth
  Wan2.1_VAE.pth
  google/umt5-xxl/...
```

Wan2.2 uses its own original model assets, including `Wan2.2_VAE.pth` and sharded DiT weights. The two backbone asset directories are not interchangeable.

Paper-style inference also needs the official Wan UniPC implementation and `diffusers==0.31.0`. Core dependencies provide the sampler dependencies; install `requirements-eval.txt` if preparing a separate evaluation environment. Run `bash scripts/evaluation/vbench/setup_vbench_standard.sh`; it installs the sampler dependency into `.deps/wan_unipc` and records downloaded resource hashes. It does not install the full VBench evaluation environment.
