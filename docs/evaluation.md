# Evaluation

Paper checkpoints: 14-GDN microstep 8000 and 21-GDN microstep 40000. Sampling: 81 frames, 480×832, 50 UniPC steps, CFG 6, flow shift 8, FP32 timestep MLP. Enhanced prompts drive generation; original prompts name files. The 21-GDN VBench result 81.85 is a qualified generation variant in the manuscript: preserve its asterisk and do not imply every external baseline uses the same protocol. No benchmark was rerun for this release preparation.

## VBench

Install the official stack and weights from https://github.com/Vchitect/VBench and place its checkout at `.deps/VBench`. Run `bash scripts/evaluation/vbench/setup_vbench_standard.sh` for the Wan sampler and enhanced prompts. Record the actual VBench revision and sampler resource hashes with results; the setup verifies sampler and enhanced-prompt SHA-256 values against the original research setup. URLs follow upstream branches, so a changed upstream file fails verification instead of silently changing the protocol.

```bash
export DIFFSYNTH_STANDARD_WAN_TEACHER=1
export DIFFSYNTH_FORCE_FP32_TIME_MLP=1
python scripts/evaluation/vbench/generate_vbench_standard_wan21_13b.py --model-dir model_weight/Wan2.1-T2V-1.3B --checkpoint checkpoints/linmorph-wan21-14gdn.safetensors --gdn-block-indices 0,1,4,5,8,9,12,13,16,17,20,21,24,25 --output output/vbench14/raw_videos --dry-run
```

Remove `--dry-run` to generate. For 21-GDN select its checkpoint and layer list. `--num-shards N --shard-index I` partitions generation. Temporal flickering requires the official static filtering procedure; run it on the generated candidates before evaluation:

```bash
python scripts/evaluation/vbench/evaluate_vbench_standard.py output/vbench14/raw_videos --temporal-filtered-dir /path/to/static_filter/filtered_videos --output output/vbench14/evaluation --reference none
```

Diagnostic subsets, incomplete dimensions and single-video demos do not establish official aggregate scores.

## VBench-2.0

The retained scripts are research utilities: `scripts/evaluation/vbench2/generate_vbench2_wan21.py`, `scripts/evaluation/vbench2/evaluate_vbench2_wan21.py`, `scripts/evaluation/vbench2/report_vbench2_wan21.py`. Install the official detector/LLM stack and weights separately. `VBENCH2_SOURCE` defaults to `.deps/VBench/VBench-2.0`; `LINMORPH_MODEL_DIR` selects base model assets. Use a distinct `RUN_ROOT` per model, and set `CHECKPOINT` plus `GDN_BLOCK_INDICES` consistently.

The evaluator retains a local readiness convention: `download_status.json` and `cache_status.json` under `VBENCH2_WEIGHTS_RECORD` (default `output/vbench2/weights`). A clean external-asset setup has not been verified here. Until it is configured, evaluate generated videos with the official VBench-2.0 commands. No turnkey VBench-2.0 environment is claimed.

## Wan2.2

Use `configs/inference/wan22_24gdn.json`, matching Wan2.2-TI2V-5B assets and 24-GDN checkpoint, and `--scheduler euler` for the recorded extension settings. The Wan2.2 generation/benchmark scripts are retained research diagnostics; inspect their CLI/configuration before use.

For Wan2.2 diagnostic commands see `scripts/ablations/prepare_wan22_5b_quick_eval.py`; latency measurements use `scripts/benchmarks/benchmark_wan22_5b.py`. Install `nvidia-ml-py` for the existing GPU monitor (also listed in `requirements-eval.txt`). The original A800 hardware restrictions and evaluation protocols are preserved.
