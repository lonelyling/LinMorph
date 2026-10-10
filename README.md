# LinMorph

**Learning to Morph Weights for Efficient Video DiT Linearization**

[中文说明](README_zh.md) · [Installation](docs/installation.md) · [Training](docs/training.md) · [Evaluation](docs/evaluation.md) · [Models](docs/models.md)

LinMorph learns a shared transformation from pretrained softmax-attention projections to Gated DeltaNet (GDN) parameters. The generated parameters initialize independently trainable replacement blocks, followed by progressive distillation and full hybrid-DiT recovery.

This release directory consolidates the local research implementation on top of DiffSynth-Studio. Core import names and checkpoint tensor names are retained for compatibility. Weights and datasets are not bundled. Public weight URLs and paper metadata have not yet been assigned. See [release status](docs/release_checklist.md) for the remaining publication steps and validation limits.

## Models

| Model | Backbone | Replaced blocks | Paper checkpoint | VBench | VBench-2.0 |
|---|---|---:|---:|---:|---:|
| LinMorph-14GDN | Wan2.1-T2V-1.3B | 14/30 | microstep 8,000 | 82.16 | 55.11 |
| LinMorph-21GDN | Wan2.1-T2V-1.3B | 21/30 | microstep 40,000 | 81.85* | 52.13 |

These are existing research results, not measurements rerun during repository preparation. `*` denotes the generation variant documented in the manuscript; retain the qualification when reporting this score. Prompt enhancement, CFG and sample selection matter; see [evaluation](docs/evaluation.md). Wan2.2-TI2V-5B extension utilities and a 24-GDN inference configuration are also included.

## Quick start

Use a fresh Python 3.10 environment and install the dependencies as described in [installation](docs/installation.md). Download the original Wan2.1-T2V-1.3B assets into `model_weight/Wan2.1-T2V-1.3B/`, and place the chosen LinMorph checkpoint under `checkpoints/`.

```bash
python -m pip install -e .
bash scripts/evaluation/vbench/setup_vbench_standard.sh
python scripts/inference.py \
  --config configs/inference/wan21_14gdn.json \
  --model-dir model_weight/Wan2.1-T2V-1.3B \
  --checkpoint checkpoints/linmorph-wan21-14gdn.safetensors \
  --prompt "A cinematic tracking shot of a motorcycle on a coastal road." \
  --output output/demo.mp4
```

The checkpoint filename above is a suggested release name, not an existing download. To inspect the configuration without loading models, add `--dry-run`. To generate with the original softmax backbone, replace `--checkpoint ...` with `--base-model`. `--scheduler euler` is available for a demo, but changes the paper's UniPC sampling protocol.

## Repository layout

```text
LinMorph/
├── configs/             # Inference, mapping, progressive KD, recovery
├── diffsynth/           # Preserved framework, GDN modules and pipelines
├── linmorph/            # Shared inference, sampling and evaluation utilities
├── examples/
│   ├── data/            # Public CSV schema example (no videos)
│   └── wanvideo/model_training/  # Original training and mapper implementations
├── scripts/             # train.py and inference.py; specialized tools in subdirectories
├── tools/               # Release checks
├── docs/                # Reproduction, model release, provenance and validation
├── requirements*.txt
├── pyproject.toml
├── LICENSE              # Preserved DiffSynth-Studio Apache-2.0 text
└── THIRD_PARTY_NOTICES.md
```

See the [script guide](scripts/README.md) for every entry point and its purpose.

## Training and evaluation

The portable training launcher accepts a local `video,caption` CSV and explicit model/data/output paths. Follow [training](docs/training.md) for mapper fitting, materialization, 7 → 14 → 21 progressive recovery, and teacher-free fine-tuning. These public recipes adapt the research settings to local data; the private QWorld corpus is not distributed.

For evaluation use [the documented official VBench workflow](docs/evaluation.md). Single-video demos and diagnostic subsets are not substitutes for complete benchmark runs.

## Acknowledgments and provenance

Built on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [Wan](https://github.com/Wan-Video/Wan2.1), and [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention). Evaluation uses [VBench](https://github.com/Vchitect/VBench). Release organization was informed by [LinFusion](https://github.com/Huage001/LinFusion) and [Wan2.2](https://github.com/Wan-Video/Wan2.2); no code was copied from these examples for repository organization.

See [third-party notices](THIRD_PARTY_NOTICES.md) and [source manifest](docs/source_manifest.json). A final citation will be added when authors and a public paper identifier are confirmed.
