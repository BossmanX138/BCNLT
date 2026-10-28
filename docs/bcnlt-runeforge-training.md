# BCNLT Runeforge Mk2 training

This workflow trains a derived Runeforge Mk2 adapter on the native BCNLT hybrid tokenizer. The preserved Runeforge Mk2 checkpoint is read-only. Sparse BCNLT embedding and output rows plus LoRA weights are saved as a standalone adapter; any merged checkpoint is written to a new staging directory.

## Current live gate

The code path and deterministic tests are complete through corpus admission, packing, sparse extension training, and promotion evaluation. Live materialization remains blocked until WSL is authenticated to Hugging Face and the access terms for `bigcode/the-stack-dedup` are accepted. The preflight stops before writing evidence or tokenizer artifacts when that access is missing.

## Build order

1. Run `scripts/preflight_bcnlt_runeforge.ps1`. It verifies the pinned Runeforge files, WSL, CUDA, RTX memory, disk, packages, Hugging Face authentication, and access to each pinned dataset revision.
2. Build the tokenizer artifact with `scripts/build_runeforge_bcnlt_tokenizer.py` using the successful preflight evidence. Keep the base checkpoint unchanged.
3. Run `scripts/build_bcnlt_training_corpus.ps1`. It executes ingestion, curriculum generation, admission, and packing as separate phases. Admission requires at least 100,000 accepted records, 50,000,000 encoded tokens, the configured diversity floors, zero holdout collisions, and all provenance checks.
4. Run `scripts/run_bcnlt_tokenizer_smoke.ps1` for the one-step compatibility check and bounded smoke adapter.
5. Evaluate the preserved base, untrained extension, and smoke adapter with the frozen `configs/bcnlt_tokenizer_eval_v1.json` suite. Write a promotion file only from those stored results.
6. Create the full-run handoff manifest with every required artifact path and SHA-256 digest.
7. Run `scripts/launch_bcnlt_full_run.ps1 -Handoff <path>`. This validates and prints the exact command without starting training.
8. Add `-StartTraining` only after reviewing that proposal. The launcher still refuses any missing or changed source, base, tokenizer, corpus, adapter, smoke, evaluation, GPU, disk, or resume identity.

## Artifact boundaries

- `Runeforge-Mk2-7B`: immutable base checkpoint.
- tokenizer artifact: added-token manifest and frozen base-tokenizer identity.
- corpus database: admitted source documents and rejection audit.
- accepted curriculum: rows that survived deduplication, split isolation, caps, holdout checks, and diversity gates.
- packed dataset: uint32 token streams, uint64 indexes, provenance JSONL, and a hashed dataset manifest.
- smoke adapter: sparse extension rows, LoRA weights, tokenizer copy, lineage, losses, and adapter hash.
- merged checkpoint: optional derived output; never replaces the base or standalone adapter.

Generated weights and corpus shards stay outside Git. Commit only code, configs, manifests that contain no private data, and documentation.
