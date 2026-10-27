# BCNLT Corpus and Runeforge Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an admitted 100,000-record, 50-million-token BCNLT curriculum and a bounded Runeforge Mk2 smoke adapter that proves the new tokenizer can be trained safely.

**Architecture:** Register and pin approved local and Hugging Face sources, parse them into canonical semantic records, generate five deterministic curriculum families, then deduplicate and split by source before packing IDs. Train only appended semantic embeddings, a BCNLT output extension, and LoRA weights against the preserved 4-bit Runeforge core; gate any full run behind tokenizer, corpus, and smoke evidence.

**Tech Stack:** Python 3.10+, SQLite, pytest, Hugging Face Datasets/Hub, Transformers, PyTorch, PEFT, bitsandbytes/Unsloth where compatible, PowerShell, WSL2, CUDA

**Spec:** `docs/superpowers/specs/2026-10-27-bcnlt-hybrid-tokenizer-design.md`

## Global Constraints

- This plan starts only after the tokenizer foundation plan passes in full.
- Accepted corpus size is at least 100,000 records and 50,000,000 encoded tokens.
- Existing named Wizard-versus-Runeforge holdouts never enter train, validation, or curriculum test.
- Splits are assigned by repository identity and source-document hash before record expansion.
- Source licenses must be explicit and belong to the approved allow-list; `no_license`, unknown, copyleft, and mixed-license rows are rejected.
- The BigCode source is `bigcode/the-stack-dedup`, Python only, at a pinned currently usable revision after its access terms are accepted.
- NVIDIA instruction data is `nvidia/OpenCodeInstruct`, pinned by commit, CC BY 4.0, Python-parseable, and test-passing only.
- BossCrafts-owned sources remain highest priority and carry an explicit `owner_provided` designation.
- The preserved Runeforge Mk2 base, existing adapter, and merged model are never overwritten.
- A full training launcher may be built, but the full run does not start as part of this plan.

## Review Focus

- Renamed or copied source files must not cross splits; Task 4 tests content and semantic duplicate clustering before split assignment.
- Dataset rows with ambiguous, absent, or conflicting licenses must be rejected with provenance; Task 2 adds license-matrix tests.
- Code containing credentials or private keys must be quarantined before persistence or export; Task 2 adds secret fixtures.
- Packed IDs must never exceed the model extension vocabulary or shift base IDs; Tasks 5-6 add range and identity tests.
- Interrupted ingestion, export, or smoke training must resume only when every pinned hash matches; Tasks 2, 5, and 7 add mismatch tests.

---

### Task 1: Pin the real Runeforge tokenizer and runtime lineage

**Files:**
- Create: `configs/runeforge_mk2_bcnlt_v1.json`
- Create: `scripts/preflight_bcnlt_runeforge.ps1`
- Create: `scripts/build_runeforge_bcnlt_tokenizer.py`
- Test: `tests/test_runeforge_tokenizer_build.py`

**Interfaces:**
- Consumes: foundation `BCNLTTokenizer`, Hugging Face adapter, and manifest builder.
- Produces: a pinned tokenizer artifact plus `tokenizer_lineage.json`.
- Produces: preflight JSON containing WSL distribution, CUDA/GPU, disk, Python package versions, base checkpoint hashes, and Hugging Face access status.

- [ ] **Step 1: Write failing lineage tests**

Assert refusal when the base config/tokenizer hashes differ, output exists, WSL is unavailable, GPU memory is below the configured smoke requirement, disk is insufficient, or the HF dataset gate has not been accepted.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_runeforge_tokenizer_build.py -q`

Expected: FAIL because the build/preflight scripts do not exist.

- [ ] **Step 3: Implement configuration and fail-closed preflight**

Resolve the actual WSL distribution and model path at runtime, then write the resolved values and hashes into evidence. Never assume `/home/bossm/models/Runeforge-Mk2-7B` is reachable merely because it existed previously.

- [ ] **Step 4: Build and verify the real tokenizer artifact**

Run the builder only after preflight passes. Verify twice and record the artifact manifest digest.

- [ ] **Step 5: Run focused tests and preflight**

Run: `python -m pytest tests/test_runeforge_tokenizer_build.py -q`

Expected: PASS; the live preflight either passes with evidence or stops with a specific blocker and no partial artifact.

- [ ] **Step 6: Commit**

```powershell
git add configs/runeforge_mk2_bcnlt_v1.json scripts/preflight_bcnlt_runeforge.ps1 scripts/build_runeforge_bcnlt_tokenizer.py tests/test_runeforge_tokenizer_build.py
git commit -m "feat: pin Runeforge BCNLT tokenizer lineage"
```

### Task 2: Build the provenance-first source registry and ingestion database

**Files:**
- Create: `configs/bcnlt_corpus_sources_v1.json`
- Create: `bcnlt/corpus/__init__.py`
- Create: `bcnlt/corpus/database.py`
- Create: `bcnlt/corpus/sources.py`
- Create: `bcnlt/corpus/filters.py`
- Create: `scripts/ingest_bcnlt_corpus.py`
- Test: `tests/test_corpus_ingestion.py`

**Interfaces:**
- Produces: `SourceSpec`, `SourceDocumentRecord`, `CorpusDatabase`, `ingest_local_source`, `ingest_stack_stream`, and `ingest_opencodeinstruct_stream`.
- Produces immutable document IDs from source provider, pinned revision, repository identity, relative path, and content SHA-256.
- Approved SPDX allow-list: `MIT`, `Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`, `ISC`, `0BSD`, `Unlicense`, and `CC0-1.0`.

- [ ] **Step 1: Write failing schema and idempotency tests**

Assert content-addressed documents, rerun idempotency, source-revision pins, stored ownership/license evidence, rejection rows, and transactional restart after interruption.

- [ ] **Step 2: Write failing license and secret tests**

Cover every approved license plus unknown, absent, mixed, GPL-family, MPL, generated/vendor, private-key blocks, known credential prefixes, and high-entropy assignments.

- [ ] **Step 3: Run tests and verify failure**

Run: `python -m pytest tests/test_corpus_ingestion.py -q`

Expected: FAIL because corpus ingestion does not exist.

- [ ] **Step 4: Implement local and Hugging Face streaming ingestion**

For The Stack, admit only Python rows with explicit allowed license metadata and repository identity. For OpenCodeInstruct, retain only CC BY 4.0 rows whose extracted Python parses, `average_test_score == 1.0`, and test-execution status contains no failure. Store attribution and upstream dataset URLs.

- [ ] **Step 5: Add source limits and quality filters**

Reject empty files, generated/vendor paths, files outside 512 bytes to 200 KiB, pathological line lengths, binary/control characters, parse failures, secrets, and evaluation-holdout hashes.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tests/test_corpus_ingestion.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add configs/bcnlt_corpus_sources_v1.json bcnlt/corpus scripts/ingest_bcnlt_corpus.py tests/test_corpus_ingestion.py
git commit -m "feat: ingest licensed BCNLT corpus sources"
```

### Task 3: Generate the five curriculum families

**Files:**
- Create: `bcnlt/corpus/curriculum.py`
- Create: `bcnlt/corpus/tasks/semantic_continuation.py`
- Create: `bcnlt/corpus/tasks/semantic_to_code.py`
- Create: `bcnlt/corpus/tasks/code_to_semantic.py`
- Create: `bcnlt/corpus/tasks/structural_reasoning.py`
- Create: `bcnlt/corpus/tasks/mixed_instruction.py`
- Create: `scripts/generate_bcnlt_curriculum.py`
- Test: `tests/test_curriculum_generation.py`

**Interfaces:**
- Consumes: admitted source documents, tokenizer records, and source split identity.
- Produces: `CurriculumRecord` with stable ID, family, template ID, source IDs, prompt mode, target mode, canonical inputs/targets, provenance, and inherited split.
- Produces: `generate_document_curriculum(document, units, records) -> Iterable[CurriculumRecord]`.

- [ ] **Step 1: Write one deterministic golden test per family**

Assert exact prompts/targets for continuation, semantic-to-code, code-to-semantic, definition/reference/scope reasoning, and mixed natural-language plus semantic-context use.

- [ ] **Step 2: Add semantic correctness tests**

Assert reconstructed code parses, emitted semantic targets decode to the original canonical records, reasoning answers derive from symbol links rather than surface-name guesses, and transformations inherit the source split.

- [ ] **Step 3: Run tests and verify failure**

Run: `python -m pytest tests/test_curriculum_generation.py -q`

Expected: FAIL because generators do not exist.

- [ ] **Step 4: Implement deterministic generators**

Generate bounded function/block windows; never create multiple rows by paraphrasing unchanged facts. OpenCodeInstruct wording may seed mixed tasks only after its code and tests pass the Task 2 filters.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_curriculum_generation.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add bcnlt/corpus/curriculum.py bcnlt/corpus/tasks scripts/generate_bcnlt_curriculum.py tests/test_curriculum_generation.py
git commit -m "feat: generate BCNLT tokenizer curriculum"
```

### Task 4: Deduplicate, split, and enforce corpus admission

**Files:**
- Create: `bcnlt/corpus/admission.py`
- Create: `scripts/validate_bcnlt_corpus.py`
- Test: `tests/test_corpus_admission.py`

**Interfaces:**
- Consumes: candidate `CurriculumRecord` rows and canonical source records.
- Produces: `AdmissionDecision`, rejection reason codes, split manifests, and `CorpusValidationReport`.
- Produces: `validate_corpus(database, tokenizer, policy) -> CorpusValidationReport`.

- [ ] **Step 1: Write failing split-isolation tests**

Assert 90/5/5 deterministic assignment by repository and content identity, copied/renamed files cluster together, transformations inherit splits, and named holdout hashes remain outside all curriculum splits.

- [ ] **Step 2: Write failing deduplication and cap tests**

Cover exact record hashes, normalized-source hashes, canonical semantic duplicates, token 5-gram near duplicates, 0.5% per-document cap, and 10% per-template/family cap.

- [ ] **Step 3: Write admission-threshold tests**

Assert failure below 100,000 accepted records or 50,000,000 encoded tokens, missing provenance, insufficient event/control-flow diversity, or a non-empty holdout-collision count.

- [ ] **Step 4: Run tests and verify failure**

Run: `python -m pytest tests/test_corpus_admission.py -q`

Expected: FAIL because admission logic does not exist.

- [ ] **Step 5: Implement deterministic admission and reporting**

Write every rejection with a stable reason code. Reports include family, event, scope-depth, control-flow, identifier, literal, uncertainty, source, license, split, duplicate, and token-count distributions.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tests/test_corpus_admission.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add bcnlt/corpus/admission.py scripts/validate_bcnlt_corpus.py tests/test_corpus_admission.py
git commit -m "feat: gate BCNLT corpus admission"
```

### Task 5: Materialize reproducible packed training shards

**Files:**
- Create: `bcnlt/corpus/packing.py`
- Create: `scripts/export_bcnlt_training_shards.py`
- Test: `tests/test_corpus_packing.py`

**Interfaces:**
- Consumes: admitted records and the exact tokenizer manifest.
- Produces: per-split `.bin` uint32 token streams, `.idx` uint64 offset/length tables, `records.jsonl` provenance indexes, and `dataset_manifest.json`.
- Produces: `PackedBCNLTDataset` with `__len__` and `__getitem__` returning `input_ids`, `labels`, and `attention_mask`.

- [ ] **Step 1: Write failing packing tests**

Assert byte-identical reruns, exact offsets, token-range validation, prompt-label masking, no truncation, shard rollover, tokenizer-digest checks, and corruption detection.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_corpus_packing.py -q`

Expected: FAIL because packing does not exist.

- [ ] **Step 3: Implement content-addressed packing and loader**

Rows longer than the configured maximum are rejected with evidence; no silent truncation is allowed. Resume skips only a shard whose input-range and output hashes match.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_corpus_packing.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add bcnlt/corpus/packing.py scripts/export_bcnlt_training_shards.py tests/test_corpus_packing.py
git commit -m "feat: pack BCNLT tokenizer training shards"
```

### Task 6: Add sparse BCNLT embeddings and training integration

**Files:**
- Create: `integrations/runeforge_mk2_tokenizer/embedding_extension.py`
- Create: `integrations/runeforge_mk2_tokenizer/train_bcnlt_adapter.py`
- Create: `integrations/runeforge_mk2_tokenizer/merge_bcnlt_adapter.py`
- Test: `tests/test_embedding_extension.py`
- Test: `tests/test_bcnlt_training_preflight.py`

**Interfaces:**
- Consumes: packed datasets, Runeforge base, tokenizer manifest, and LoRA target modules.
- Produces: `ExtendedEmbedding`, `ExtendedLMHead`, `attach_bcnlt_extension(model, manifest, definitions, seed)`, and a standalone adapter artifact.
- Preserves base embedding/output rows as frozen and byte-identical.

- [ ] **Step 1: Write failing embedding tests on a tiny causal model**

Assert base IDs return unchanged embeddings/logits, added IDs select only extension rows, initialization is deterministic from token definitions, gradients touch only extension rows and LoRA parameters, and out-of-range IDs fail.

- [ ] **Step 2: Write failing training-preflight tests**

Assert exact base, tokenizer, corpus, config, CUDA, and resume hashes; require nonzero trainable extension and LoRA parameters; reject whole-table trainability.

- [ ] **Step 3: Run tests and verify failure**

Run: `python -m pytest tests/test_embedding_extension.py tests/test_bcnlt_training_preflight.py -q`

Expected: FAIL because extension/training integration does not exist.

- [ ] **Step 4: Implement sparse extensions and QLoRA integration**

Reuse the proven Runeforge target modules. Prefer Unsloth for the 4-bit base and checkpointing only if the custom extension passes the tiny-model and one-step compatibility test; otherwise use Transformers plus PEFT/bitsandbytes and record the engine choice.

- [ ] **Step 5: Implement separate save and merge paths**

The adapter save contains extension weights, LoRA weights, tokenizer artifact, and lineage. Merge writes to a new staging directory, verifies every source hash before and after, and never deletes the adapter.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tests/test_embedding_extension.py tests/test_bcnlt_training_preflight.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add integrations/runeforge_mk2_tokenizer tests/test_embedding_extension.py tests/test_bcnlt_training_preflight.py
git commit -m "feat: train sparse BCNLT embeddings on Runeforge"
```

### Task 7: Freeze evaluation and run the bounded smoke training

**Files:**
- Create: `bcnlt/tokenizer_eval.py`
- Create: `configs/bcnlt_tokenizer_eval_v1.json`
- Create: `scripts/run_bcnlt_tokenizer_eval.py`
- Create: `scripts/run_bcnlt_tokenizer_smoke.ps1`
- Test: `tests/test_tokenizer_eval.py`

**Interfaces:**
- Produces deterministic tokenizer, structural, semantic-to-code, raw-code, and ordinary-instruction metrics.
- Produces `smoke_manifest.json`, training logs, adapter hash, and `smoke_promotion.json`.

- [ ] **Step 1: Write failing evaluator tests**

Pin scoring for round trips, symbol/scope facts, parseable reconstruction, exact raw-code controls, and instruction controls. Assert identical prompts and decoding parameters across base and adapter.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_tokenizer_eval.py -q`

Expected: FAIL because the evaluator does not exist.

- [ ] **Step 3: Implement and baseline the frozen v1 suite**

Run the preserved Mk2 checkpoint and untrained extension first. Store all prompts, outputs, seeds, decoding settings, runtime quantization, and hashes.

- [ ] **Step 4: Run one-step VRAM compatibility smoke**

Use the longest admitted smoke record. Require a completed forward/backward/update, finite loss, saved checkpoint, and unchanged source hashes.

- [ ] **Step 5: Run the bounded smoke adapter**

Train only the configured smoke subset. Resume only with exact manifest identity. Do not launch the full corpus run.

- [ ] **Step 6: Evaluate promotion gates**

Require lower held-out BCNLT loss, at least 90% BCNLT structural accuracy, no tokenizer/runtime failures, and no raw-code or ordinary-instruction regression greater than two absolute percentage points.

- [ ] **Step 7: Commit code and configuration, excluding generated weights**

```powershell
git add bcnlt/tokenizer_eval.py configs/bcnlt_tokenizer_eval_v1.json scripts/run_bcnlt_tokenizer_eval.py scripts/run_bcnlt_tokenizer_smoke.ps1 tests/test_tokenizer_eval.py
git commit -m "feat: gate BCNLT tokenizer smoke training"
```

### Task 8: Generate the full admitted corpus and guarded run handoff

**Files:**
- Create: `scripts/build_bcnlt_training_corpus.ps1`
- Create: `scripts/launch_bcnlt_full_run.ps1`
- Create: `docs/bcnlt-runeforge-training.md`
- Modify: `README.md`
- Test: `tests/test_full_run_gate.py`

**Interfaces:**
- Consumes: every manifest and promotion decision from Tasks 1-7.
- Produces: final corpus database, packed shards, validation report, source-attribution report, and a full-run launcher that defaults to validation-only.

- [ ] **Step 1: Write failing full-run gate tests**

Assert refusal for any missing/changed source, tokenizer, corpus, base, adapter, smoke, evaluation, disk, GPU, or resume hash. Assert the launcher does nothing without explicit `-StartTraining`.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_full_run_gate.py -q`

Expected: FAIL because guarded orchestration does not exist.

- [ ] **Step 3: Implement restartable end-to-end corpus orchestration**

Run ingestion, generation, admission, and packing in separate content-addressed phases. Write one top-level manifest linking every phase and rejection report.

- [ ] **Step 4: Generate until quality gates, not merely counts, pass**

Stop ingestion after at least 100,000 accepted records and 50,000,000 encoded tokens survive admission with required diversity. Do not fill shortfalls with duplicate templates.

- [ ] **Step 5: Implement the guarded full-run launcher and documentation**

Default behavior verifies and prints the exact proposed command, estimated steps, disk use, and manifest hashes. Starting training requires the explicit switch and a passing smoke promotion file.

- [ ] **Step 6: Run all tests and corpus validation**

Run: `python -m pytest -q`

Expected: all tests pass and corpus validation reports every admission gate as passed.

- [ ] **Step 7: Commit code and documentation, excluding generated corpus shards**

```powershell
git add scripts/build_bcnlt_training_corpus.ps1 scripts/launch_bcnlt_full_run.ps1 docs/bcnlt-runeforge-training.md README.md tests/test_full_run_gate.py
git commit -m "feat: prepare guarded BCNLT full training run"
```
