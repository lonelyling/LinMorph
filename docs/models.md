# Model release

Primary candidates: Wan2.1 14-GDN `model-step-8000.safetensors` and 21-GDN `model-step-40000.safetensors`. Suggested public filenames are `linmorph-wan21-14gdn.safetensors` and `linmorph-wan21-21gdn.safetensors`. See [model_manifest.json](model_manifest.json) for local source paths, sizes and sampling settings. No weights were moved, copied into this tree or uploaded.

Host code/configuration/docs on GitHub; host these inference weights and matching configurations on Hugging Face. Link to the original Wan base assets instead of duplicating the text encoder, VAE and tokenizer. Keep intermediate checkpoints, optimizer/ZeRO states and caches in the research workspace.

Optionally publish the final shared FullMix/PostFullResidual mapper separately for method reproduction. Mapper, materialized GDN initialization bank, recovered inference DiT and full-state resume checkpoints are different artifacts and must be labeled distinctly.

Before upload, verify tensor coverage, inference over the original backbone, full-file SHA-256, license, base model identity, exact block configuration and generation protocol. Microstep numbers are not diffusion sampling steps. Download URLs remain unassigned; none have been invented.

Use [model_card_template.md](model_card_template.md) when preparing the hosted release.
