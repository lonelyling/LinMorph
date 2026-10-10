# Organization and provenance

Research-release structure references: https://github.com/Huage001/LinFusion and https://github.com/Wan-Video/Wan2.2 . Their organization informed the separation of overview, installation, training, inference, evaluation and hosted weights; no source was imported from them for repository organization.

The `diffsynth` namespace and checkpoint tensor names remain compatible. The external exact-large FullMix mapper is now next to the preserved training implementations. Public entry points and JSON configurations replace dependence on machine-specific launch scripts.

`source_manifest.json` records selected source paths relative to the original longlingyi directory and their pre-copy SHA-256. These are provenance labels, not runtime dependencies. Modified copies are enumerated in the validation report.

Excluded: original Git history, environments, private datasets, weights, optimizer states, caches, logs, videos, cloud transfer helpers and machine-specific debugging/recovery shell scripts. All remain in their original locations; this curated release is not a duplicate backup of every experimental artifact.

## Script layout

Only `scripts/train.py` and `scripts/inference.py` remain as top-level command-line entry points. Initialization, VBench/VBench-2.0 evaluation, analysis, ablations and performance measurements have separate subdirectories. `tools/check_release.py` handles release validation. See [the script guide](../scripts/README.md).

Shared code lives in the installable `linmorph` package: model loading and media helpers in `inference.py`, UniPC in `sampling.py`, prompt/task planning in `vbench.py` and `vbench2.py`, Wan2.2 configuration and loading in `wan22.py`, projection/materialization helpers in `materialization.py`, and the original GPU timing/monitoring utility in `benchmark.py`. The latter fixes a missing dependency in the initial release copy. Entry points import these modules rather than borrowing code from other executable scripts.

`script_migration.json` maps old release-entry paths to new ones. `source_manifest.json` retains original research paths and records `additional_destinations` when one original script was split into multiple files. Run documentation uses only the new paths; no forwarding scripts were left at the old top-level locations. The original research directories were not modified.
