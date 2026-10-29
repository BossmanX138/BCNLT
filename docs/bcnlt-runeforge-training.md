# BCNLT Runeforge Mk2 training

This workflow trains a derived Runeforge Mk2 adapter on the native BCNLT hybrid tokenizer. The preserved Runeforge Mk2 checkpoint is read-only. Sparse BCNLT embedding and output rows plus LoRA weights are saved as a standalone adapter; any merged checkpoint is written to a new staging directory.

## Current live gate

Hugging Face authentication and source access are working. The corrected v4
smoke lineage materialized 979 admitted records and 858,505 packed tokens, then
trained all 932 train records for 117 optimizer steps on the RTX 3070. The
standalone adapter is
`D:\BCNLT-runs\runeforge-mk2-bcnlt-v4-context-smoke\adapter-epoch-1` with
SHA-256
`3574f08bb9386b7c006278acb98633056c1c201316989f1c1030486bca4605bc`.
The preserved Runeforge Mk2 checkpoint was not modified.

Held-out loss fell from 4.364707 for the untrained extension to 0.095036 for
the v4 adapter. Structural resolution, semantic-to-code, raw-code control, and
ordinary-instruction probes passed exactly. The Unicode exact-code probe
returned the requested line plus an unwanted comment, so the corrected
promotion gate records `tokenizer_accuracy_below_threshold` and does not
promote the adapter. An aligned held-out code-to-semantic probe now emits a
complete 99-token canonical stream instead of failing to terminate, but one
span endpoint was 10 instead of 13; one mixed reconstruction emitted the
semantically equivalent `import pytest as pytest` instead of the original
surface form.

Full training remains locked. The smoke corpus does not satisfy the full
100,000-record and 50,000,000-token admission floors, and the v4 adapter has not
passed the corrected promotion gate.

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
