# BCNLT Runeforge Mk2 training

This workflow trains a derived Runeforge Mk2 adapter on the native BCNLT hybrid tokenizer. The preserved Runeforge Mk2 checkpoint is read-only. Sparse BCNLT embedding and output rows plus LoRA weights are saved as a standalone adapter; any merged checkpoint is written to a new staging directory.

## Multilingual v1 recipe

The current multilingual path trains one shared Runeforge adapter across Python, JavaScript, TypeScript, C, C++, C#, Java, Go, Rust, PHP, Ruby, Kotlin, Swift, and Bash. It does not create language-specific LoRA directories. The sampler rotates through every language and all six curriculum families while one sparse BCNLT embedding/output extension and one LoRA remain trainable.

The authoritative configuration is split into bounded smoke and full-run files:

- `configs/runeforge_mk2_bcnlt_multilingual_smoke_v1.json`: seed 1729, microbatch 1, gradient accumulation 8, and exactly 128 optimizer steps.
- `configs/runeforge_mk2_bcnlt_multilingual_full_v1.json`: one complete epoch with validation and checkpoints every 500 optimizer steps.

Both use 4,096-token sequences, AdamW, learning rate `2e-4`, 3% linear warmup followed by cosine decay, and LoRA rank 16, alpha 32, dropout 0 over `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.

Build a content-addressed smoke corpus first. The command samples 100 live rows from every pinned language partition, ingests the v2 source registry, generates and admits the six task families, packs the native tokenizer streams, and refuses a corpus without all 84 language/family cells or 256 held-out validation rows per language.

```powershell
.\scripts\build_bcnlt_multilingual_corpus.ps1 `
  -TokenizerArtifact /mnt/d/BCNLT-runs/runeforge-mk2-bcnlt-v1/tokenizer `
  -OutputRoot /mnt/d/BCNLT-runs/runeforge-mk2-bcnlt-multilingual-v1 `
  -PolicyProfile smoke
```

After creating exact compatibility and smoke lineage files for the two training configs, run the one-step VRAM check and then the 128-step smoke. The second phase measures the same 256 held-out examples per language before and after training. Promotion requires finite loss, at least 2% improvement between the first and last 32 optimizer-step means, and no language final loss above 125% of its initial loss.

```powershell
.\scripts\run_bcnlt_multilingual_smoke.ps1 `
  -TokenizerArtifact <wsl-tokenizer-path> `
  -PackedDataset <wsl-packed-path> `
  -SourceSample <source-sample.json> `
  -PackedAudit <per-language-validation.json> `
  -CompatibilityLineage <one-step-lineage.json> `
  -SmokeLineage <multilingual-smoke-lineage.json> `
  -Output <new-smoke-adapter-path>
```

Full training uses handoff schema `bcnlt-full-run-handoff-v2`. It hashes the parser bundle, all 14 semantic specs, source registry, tokenizer, corpus, per-language validation, preserved base identity, standalone adapter, smoke evidence, frozen evaluations, promotion decision, and full training config. Any missing or changed file blocks validation and resume. The launcher remains validation-only unless `-StartTraining` is present:

```powershell
.\scripts\launch_bcnlt_multilingual_full_run.ps1 -Handoff <handoff-v2.json>
```

Task 16 prepares and tests this guarded path. It does not authorize or start the 140,000-record full run; that begins only after the source sample, packed-corpus audit, 128-step smoke promotion, frozen evaluation, and whole-branch review all pass.

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

## Adaptive vocabulary v5 experiment

The v5 work is a controlled copy-on-write experiment over the preserved v1
tokenizer and v4 adapter. Its [design](superpowers/specs/2026-10-29-bcnlt-adaptive-vocabulary-v5-experiment-design.md)
and [implementation plan](superpowers/plans/2026-10-29-bcnlt-adaptive-vocabulary-v5.md)
define the gates. The runner writes only beneath
`D:\BCNLT-runs\runeforge-mk2-bcnlt-v5-adaptive-vocab` and rejects any output
that overlaps the repository, base checkpoint, v1 tokenizer, or v4 artifacts.

The current pinned corpus is Python-focused and uses the legacy database
schema. Run this preliminary, non-GPU audit from the repository root in
PowerShell. It copies the legacy database into a new v2 database, regenerates
the curriculum from that copy, captures lineage, and then audits the required
14-language diversity. None of these commands writes to the v4 source.

```powershell
$Py = "/home/bossm/.venvs/wizard-phi-eval/bin/python"
$Repo = (wsl.exe -d Ubuntu -- wslpath -a (Resolve-Path .)).Trim()
$Out = "/mnt/d/BCNLT-runs/runeforge-mk2-bcnlt-v5-adaptive-vocab/preliminary-audit"

wsl.exe -d Ubuntu --cd $Repo -- $Py -m scripts.rebuild_bcnlt_corpus_v2 `
  --source /mnt/d/BCNLT-runs/runeforge-mk2-bcnlt-v4-context-smoke/corpus.sqlite `
  --destination "$Out/corpus-v2.sqlite" `
  --source-registry "$Repo/configs/bcnlt_corpus_sources_v1.json" `
  --report "$Out/corpus-rebuild.json"

wsl.exe -d Ubuntu --cd $Repo -- $Py -m scripts.generate_bcnlt_curriculum `
  --database "$Out/corpus-v2.sqlite" `
  --output "$Out/curriculum.jsonl" `
  --split train

wsl.exe -d Ubuntu --cd $Repo -- $Py -c "from pathlib import Path; import sys; from bcnlt.adaptive_vocab.lineage import ExperimentLineage; x=ExperimentLineage.capture(config_path=sys.argv[1],source_config_path=sys.argv[2],corpus_manifest_path=sys.argv[3],v1_manifest_path=sys.argv[4],v4_gate_path=sys.argv[5],base_model_path=sys.argv[6],parser_bundle_path=sys.argv[7],code_revision=sys.argv[8]); Path(sys.argv[9]).write_bytes(x.canonical_bytes())" `
  "$Repo/configs/bcnlt_adaptive_vocab_v5.json" `
  "$Repo/configs/bcnlt_corpus_sources_v1.json" `
  "$Out/corpus-rebuild.json" `
  /mnt/d/BCNLT-runs/runeforge-mk2-bcnlt-v1/tokenizer/bcnlt_tokenizer_manifest.json `
  /mnt/d/BCNLT-runs/runeforge-mk2-bcnlt-v4-context-smoke/smoke-promotion-epoch-1-gate-v2.json `
  /home/bossm/models/Runeforge-Mk2-7B `
  "$Repo/configs/bcnlt_parser_bundle_v1.json" `
  (git rev-parse HEAD) `
  "$Out/lineage.json"

wsl.exe -d Ubuntu --cd $Repo -- $Py -m scripts.audit_bcnlt_adaptive_vocab `
  --database "$Out/corpus-v2.sqlite" `
  --accepted-records "$Out/curriculum.jsonl" `
  --tokenizer-artifact /mnt/d/BCNLT-runs/runeforge-mk2-bcnlt-v1/tokenizer `
  --base-tokenizer /home/bossm/models/Runeforge-Mk2-7B `
  --config "$Repo/configs/bcnlt_adaptive_vocab_v5.json" `
  --source-config "$Repo/configs/bcnlt_corpus_sources_v1.json" `
  --corpus-manifest "$Out/corpus-rebuild.json" `
  --lineage "$Out/lineage.json" `
  --output "$Out/corpus-audit.json"
```

Exit code `2` is the expected valid research hard stop for this preliminary
corpus. The original report incorrectly classified curriculum rows with no
semantic side as `invalid_semantic_payload`; the corrected audit ignores those
rows for macro counting while still validating their source and language
provenance. The verified corrected report contains only
`insufficient_shared_candidate_diversity`. Do not run candidate discovery from
this report. A qualifying run needs at least 100 documents, three repositories,
and 100,000 eligible payload tokens for every configured language before the
shared-candidate gates can be evaluated.

For a run specification whose phase commands and immutable inputs have been
filled in, the tokenizer-only runner is:

```powershell
.\scripts\run_bcnlt_adaptive_vocab_v5.ps1 -Spec D:\BCNLT-runs\v5-run-spec.json
```

The runner returns `0` for a completed pass, `2` for a valid research hard
stop, and `1` for corruption or a runtime failure. `-Resume` accepts only exact
code, config, input, command, and output hashes. Training phases are absent by
default. GPU continuation requires both a passing `selected-inventory.json`
and the explicit `-StartTraining` switch:

```powershell
.\scripts\run_bcnlt_adaptive_vocab_v5.ps1 -Spec D:\BCNLT-runs\v5-run-spec.json -Resume -StartTraining
```

The `paired_training` phase must invoke
`python -m scripts.train_bcnlt_adaptive_pair`. That command loads two separate
models with the v1 extension, verifies and restores the same custom v4 adapter
state into each, grows only the adaptive arm, checks the ordered packed
`record_id` population, and runs both arms with one optimizer-step and
validation schedule. Separate v1 and adaptive validation packs must contain
the same ordered validation IDs; their measured losses are stored at each
shared validation step. It writes the two standalone adapters plus
`matched-training.json` beneath a new output directory.

Expected artifacts are lineage, corpus audit, candidates, inventories,
tokenizer evaluations, boundary entropy, paired datasets, two standalone
adapter continuations, the four-state model evaluation, and a promotion
proposal. A gate failure stops downstream phases. The experiment never merges
weights, replaces v1, launches an unbounded full run, or publishes an artifact.
