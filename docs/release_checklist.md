# Publication status

Prepared: independent source copy, internalized mapper, private dataset test-demo removal, portable launchers, model/training configurations, CSV schema example, documentation, dependency metadata and Git ignore rules.

Before publication:

- Confirm authors, project ownership, paper URL and final citation.
- Confirm attribution and redistribution permission for the externally supplied mapper (see THIRD_PARTY_NOTICES.md).
- Confirm the two model candidates against the final manuscript, then publish inference weights to the chosen Hugging Face account. Fill in real URLs and full-file SHA-256 hashes.
- Verify a fresh environment, generation with both exact checkpoints, and a small local-data training run on CUDA hardware.
- Record exact VBench/sampler revisions, full coverage and the 21-GDN generation variant.
- Run `python tools/check_release.py` and inspect staged files before the first commit.

No remote repository was created or pushed. This is a prepared release candidate, not a claim that all reproduction runs or weight uploads are complete. The original research files remain untouched. A credential-bearing dataset demo was removed from this copy; if its original credentials remain valid or were previously shared, the owner should rotate them. No credential values are documented here.
