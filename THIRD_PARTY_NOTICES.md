# Third-party notices

The `LICENSE` file is copied unchanged from the local DiffSynth-Studio checkout and contains the Apache License 2.0. Existing notices and source headers in copied framework code are preserved. The `diffsynth` framework and original training utilities derive from ModelScope's DiffSynth-Studio, with local research modifications; they must not be represented as entirely new LinMorph code.

- DiffSynth-Studio: https://github.com/modelscope/DiffSynth-Studio
- Wan backbones and sampler: https://github.com/Wan-Video/Wan2.1 and https://github.com/Wan-Video/Wan2.2
- Flash Linear Attention: https://github.com/fla-org/flash-linear-attention (installed dependency)
- VBench: https://github.com/Vchitect/VBench (separately installed dependency)

The local `from_congyue/train_single_layer_linear_adapter.py` is included under `examples/wanvideo/model_training/` because it is required by the actual FullMix path. Its source directory had no separate license file. Its contributor attribution and redistribution permission must be confirmed before publication; the copied Apache license is not evidence of permission for that separate contribution. No new copyright owner or author list has been invented.

Pretrained weights, datasets and separately downloaded evaluation resources retain their own terms. They are not included in this code tree. This file records provenance rather than asserting a new license grant for unspecified contributions.
