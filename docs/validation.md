# Release validation

These checks concern source preparation. No GPU training, video generation, full benchmark, model upload or clean dependency installation was run.

Completed:

- All selected original source files were checked against their pre-copy SHA-256; no original changes were found. See [source_verification.json](source_verification.json).
- All seven training recipes passed the actual preserved argument parsers and portable launcher dry runs.
- All three inference recipes passed the new inference entry-point dry run.
- The FullMix mapper, KD training, inference helper, materialization and staged-merge modules imported in the existing Python 3.10 environment.
- The inference import check verified that 52 loaded `diffsynth` modules resolve inside LinMorph rather than the original checkout.
- The two local main-table checkpoint headers match the 14/21-GDN layer configurations and contain no LoRA tensors. Only headers were read; complete file hashes and actual generation remain unverified.
- Python syntax, JSON configuration, documented local links, shell syntax and common credential/private-path patterns were checked. The checks report locations, not secret values.

Recheck the portable source checks with:

```bash
python tools/check_release.py
bash -n scripts/evaluation/vbench/setup_vbench_standard.sh
```

Existing-environment imports do not prove that all dependency versions work in a fresh environment. Native FLA/attention kernels, DDP/DeepSpeed behavior, complete weight loading and numerical output must still be validated on the publication hardware. A source-pattern scan is not a comprehensive security audit.

## Script reorganization checks

All 18 Python command-line entry points passed `--help` from a temporary directory outside the checkout with isolated Python and no inherited project search path. Seven training and three inference recipes passed dry runs; the comparison, Wan2.2 diagnostic/benchmark and two VBench planning paths also passed (15 dry runs in total). VBench planner tests used synthetic temporary fixtures and do not constitute benchmark results.

All 23 shared inference function bodies are AST-identical to their pre-reorganization versions. Shared module root paths and runtime imports resolve within this checkout. Release checks also enforce the two top-level script entries, internal module existence and all provenance destinations. Details are recorded in [layout_validation.json](layout_validation.json). The source manifest now includes the previously omitted benchmark runtime helper; all original research sources remain unchanged.

No GPU model execution or external upload was performed during this reorganization.
