# BCNLT Adaptive Vocabulary v5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated, append-only BCNLT v5 experiment that discovers cross-language semantic-payload macros, verifies lossless compression, measures boundary entropy, and compares matched v1/v5 continuation adapters without changing Runeforge Mk2 or BCNLT v1 artifacts.

**Architecture:** Add canonical language provenance to the existing admitted-corpus path, then implement adaptive-vocabulary work in a focused `bcnlt/adaptive_vocab` package. The v5 tokenizer wraps the verified v1 tokenizer, substitutes only contiguous base-token runs inside semantic fields, and expands every macro before v1 decoding. Tokenizer gates precede any GPU work; paired training starts from copied v4 adapter state and differs only by tokenization and new macro rows.

**Tech Stack:** Python 3.11+, pytest, PyTorch, Transformers, PEFT, bitsandbytes/Unsloth, SQLite, Hugging Face fast tokenizers, PowerShell, WSL Ubuntu, canonical JSON/JSONL artifacts.

**Spec:** `docs/superpowers/specs/2026-10-29-bcnlt-adaptive-vocabulary-v5-experiment-design.md`

## Global Constraints

- Preserve `/home/bossm/models/Runeforge-Mk2-7B`, every Runeforge base token ID, every BCNLT v1 structural ID, and all v4 artifacts byte-for-byte.
- Write run artifacts only beneath `D:\BCNLT-runs\runeforge-mk2-bcnlt-v5-adaptive-vocab`; reject the base model, repository, v1 artifact, and every v4 path as output roots.
- Do not discover or download new corpus sources automatically. A qualifying source registry must explicitly pin approved names, revisions, licenses, repository rules, and hashes.
- Require all fourteen IDs in `bcnlt/languages.py`; each must contribute at least 100 admitted documents, three repositories, and 100,000 semantic-payload base tokens.
- Discover only 2-16-token sequences inside one semantic field; cap a candidate at eight occurrences per deduplicated document.
- A macro requires at least three languages, three repositories, and five documents; one repository may supply at most 50 percent and one document at most 10 percent of capped occurrences.
- Use `score(c) = (L - 1) * max_p(third_highest_l(rate(c,l,p)))`, `rate = 1,000,000 * capped_occurrences / eligible_base_tokens`, and require `score >= 25`.
- Attempt inventories of exactly 256, 512, and 1,024 eligible macros; never pad an inventory with a below-threshold candidate.
- Require zero round-trip failures, unchanged text/raw-code streams, at least 10 percent held-out semantic-stream compression, no per-language expansion, no more than 10 percent p95 encode/decode slowdown, and byte-identical Windows/WSL artifacts.
- Keep predictive boundary entropy separate from knowledge uncertainty, UYR, wildcard memory, admission, and candidate ranking.
- Train matched arms on at most 10,000 identical examples for one epoch or 512 optimizer steps, validate every 64 steps, and require at least eight available training occurrences per selected macro.
- Use held-out negative log likelihood in bits per identical canonical UTF-8 byte for cross-tokenizer loss gates; per-token loss is informational only.
- Every stage is content-addressed, writes atomically, and resumes only from exact code/configuration/corpus/tokenizer/model/prior-stage hashes.

## Review Focus

- A legacy corpus database without canonical-language provenance must fail with a rebuild instruction and must never be migrated in place; Task 1 pins this.
- Overlapping macro prefixes must always choose the longest expansion and then the lower macro ID, on Windows and WSL; Task 5 pins this.
- A corrupt adaptive manifest, structural-ID expansion, out-of-range ID, or macro cycle must be rejected before decoding or model loading; Tasks 5 and 6 pin this.
- Empty fields, Unicode normalization variants, CRLF/LF, and text resembling `<|bcnlt:...|>` must reconstruct exactly; Task 6 pins this.
- Entropy alignment must ignore padding and `-100` labels and must map the prediction at position `i` to target token `i+1`; Task 8 pins this.

---

### Task 1: Canonical Language Provenance in the Admitted Corpus

**Files:**
- Modify: `bcnlt/corpus/database.py`
- Modify: `bcnlt/corpus/sources.py`
- Modify: `bcnlt/corpus/curriculum.py`
- Modify: `bcnlt/corpus/admission.py`
- Modify: `scripts/ingest_bcnlt_corpus.py`
- Create: `scripts/rebuild_bcnlt_corpus_v2.py`
- Modify: `configs/bcnlt_corpus_sources_v1.json`
- Modify: `tests/test_corpus_ingestion.py`
- Modify: `tests/test_corpus_admission.py`
- Modify: `tests/test_curriculum_generation.py`

**Interfaces:**
- Produces: `SourceSpec.canonical_language: str` and `SourceDocumentRecord.canonical_language: str`.
- Produces: required curriculum provenance key `canonical_language` containing an exact value from `CANONICAL_LANGUAGE_IDS`.
- Produces: corpus schema marker `bcnlt-corpus-database-v2`; opening a populated v1 database raises `ValueError("legacy corpus database requires a new v2 output path")`.
- Produces: `rebuild_corpus_v2(source: Path, destination: Path, source_registry: Path) -> RebuildReport`, which copies a legacy database into a new v2 path only when each source name/revision maps to exactly one canonical language in the pinned registry.
- Consumes later: Tasks 3, 4, and 9 use canonical language from admitted provenance and never infer it from filenames.

- [ ] **Step 1: Write failing provenance and legacy-database tests**

Add tests asserting:

```python
assert SourceSpec(..., canonical_language="python").canonical_language == "python"
with pytest.raises(ValueError, match="canonical language"):
    SourceSpec(..., canonical_language="c++")
assert dict(curriculum.provenance)["canonical_language"] == "python"
with pytest.raises(ValueError, match="new v2 output path"):
    CorpusDatabase(populated_v1_database)
```

Update existing fixtures to supply `canonical_language="python"`; add an admission test proving missing language provenance returns `missing_provenance`. Add rebuild tests proving the source database is unchanged, the destination must not exist, and an unmapped or ambiguously mapped source name/revision fails before copying any document.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `python -m pytest tests/test_corpus_ingestion.py tests/test_corpus_admission.py tests/test_curriculum_generation.py -q`

Expected: FAIL because the dataclasses and schema lack canonical language.

- [ ] **Step 3: Add exact language fields and v2 schema validation**

Validate `SourceSpec.canonical_language` with `resolve_language_id` but require the supplied value itself to be canonical. Store it in the documents table, all `SourceDocumentRecord` instances, and curriculum provenance. Create a metadata table containing the exact database schema version; accept a new/empty database and refuse a populated unversioned database.

- [ ] **Step 4: Wire pinned source configuration**

Require each enabled registry source to declare `"canonical_language"`; set both existing sources to `"python"`. Pass the value through every ingestion function and rejection record without changing the v1 source revisions.

- [ ] **Step 5: Implement an explicit new-path legacy rebuild**

Read the legacy database read-only, validate every source name/revision against the supplied pinned registry, and write a fresh v2 database through a temporary sibling path. Record source/destination hashes and language counts in `RebuildReport`. Never add a column to, replace, or write metadata into the source database. Regenerate curriculum/admission from the new database so canonical language enters every new accepted row.

- [ ] **Step 6: Run focused and full tests**

Run: `python -m pytest tests/test_corpus_ingestion.py tests/test_corpus_admission.py tests/test_curriculum_generation.py -q`

Expected: PASS.

Run: `wsl.exe -d Ubuntu -- /home/bossm/.venvs/wizard-phi-eval/bin/python -m pytest -q`

Expected: all repository tests pass, except any already-documented platform skip.

- [ ] **Step 7: Commit**

```bash
git add bcnlt/corpus/database.py bcnlt/corpus/sources.py bcnlt/corpus/curriculum.py bcnlt/corpus/admission.py scripts/ingest_bcnlt_corpus.py scripts/rebuild_bcnlt_corpus_v2.py configs/bcnlt_corpus_sources_v1.json tests/test_corpus_ingestion.py tests/test_corpus_admission.py tests/test_curriculum_generation.py
git commit -m "feat: record canonical language in BCNLT corpus provenance"
```

### Task 2: Immutable Adaptive Experiment Configuration and Lineage

**Files:**
- Create: `bcnlt/adaptive_vocab/__init__.py`
- Create: `bcnlt/adaptive_vocab/config.py`
- Create: `bcnlt/adaptive_vocab/lineage.py`
- Create: `configs/bcnlt_adaptive_vocab_v5.json`
- Create: `tests/test_adaptive_vocab_config.py`
- Create: `tests/test_adaptive_vocab_lineage.py`

**Interfaces:**
- Produces: `AdaptiveVocabularyConfig.load(path: str | Path) -> AdaptiveVocabularyConfig`.
- Produces: `ExperimentLineage.capture(*, config_path, source_config_path, corpus_manifest_path, v1_manifest_path, v4_gate_path, base_model_path, parser_bundle_path, code_revision) -> ExperimentLineage`.
- Produces: `prepare_output_root(output: Path, protected: Sequence[Path]) -> Path`, which resolves Windows/WSL aliases before overlap checks and creates a new empty root atomically.
- Consumes later: every adaptive CLI accepts this configuration and lineage record.

- [ ] **Step 1: Write failing configuration tests**

Assert the production JSON loads with the spec's exact values: languages from `CANONICAL_LANGUAGE_IDS`, floors `100/3/100000`, candidate lengths `2/16`, document cap `8`, support `3/3/5`, concentration `0.50/0.10`, score floor `25`, budgets `[256,512,1024]`, compression `0.10`, latency `0.10`, macro coverage `8`, example cap `10000`, optimizer cap `512`, and validation interval `64`.

Also assert malformed regexes, duplicate budgets, noncanonical languages, and relaxed production thresholds are rejected.

- [ ] **Step 2: Write failing lineage and protected-path tests**

Cover exact SHA-256 capture, a changed input hash, output equal to a protected path, output nested under a protected path, a protected path nested under output, symlink/junction aliases, existing nonempty output, and interrupted temporary output cleanup.

- [ ] **Step 3: Run focused tests and verify failure**

Run: `python -m pytest tests/test_adaptive_vocab_config.py tests/test_adaptive_vocab_lineage.py -q`

Expected: FAIL because the package does not exist.

- [ ] **Step 4: Implement strict dataclasses and canonical serialization**

Use frozen dataclasses, exact schema version `bcnlt-adaptive-vocab-v5`, canonical UTF-8 JSON with LF endings, compiled secret-pattern validation, and lowercase SHA-256 digests. Put the explicit private-key, `AKIA`, GitHub token, OpenAI-style key, JWT, URI-userinfo, and 24-character hex/base64 rules in the JSON so they travel with the experiment identity.

- [ ] **Step 5: Implement lineage capture and safe-root creation**

Hash files by bytes and directories by sorted relative path plus file hash. Record paths for human audit but compare hashes for identity. Write through a sibling temporary path and rename only after validation.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tests/test_adaptive_vocab_config.py tests/test_adaptive_vocab_lineage.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add bcnlt/adaptive_vocab configs/bcnlt_adaptive_vocab_v5.json tests/test_adaptive_vocab_config.py tests/test_adaptive_vocab_lineage.py
git commit -m "feat: define immutable BCNLT adaptive experiment lineage"
```

### Task 3: Multilingual Corpus Audit and Hard Diversity Stop

**Files:**
- Create: `bcnlt/adaptive_vocab/corpus_audit.py`
- Create: `scripts/audit_bcnlt_adaptive_vocab.py`
- Create: `tests/test_adaptive_vocab_corpus_audit.py`

**Interfaces:**
- Consumes: `CorpusDatabase.documents()`, admitted `CurriculumRecord` JSONL, a verified v1 `BCNLTTokenizer`, `AdaptiveVocabularyConfig`, and `ExperimentLineage`.
- Produces: `audit_corpus(documents: Sequence[SourceDocumentRecord], records: Sequence[CurriculumRecord], tokenizer: BCNLTTokenizer, config: AdaptiveVocabularyConfig) -> CorpusAuditReport`.
- Produces: `CorpusAuditReport.passed`, `failure_codes`, exact per-language document/repository/payload-token counts, training-only source document IDs, and canonical `to_dict()`.
- Produces: `corpus-audit.json`; later tasks refuse a report with `passed == false`.

- [ ] **Step 1: Write failing audit tests**

Use reduced test-only floors while asserting production floors in Task 2. Cover: Python-only input returns `insufficient_shared_candidate_diversity`; fourteen qualifying languages pass; two documents with the same repository count once; validation/test records do not contribute; a curriculum row whose language disagrees with its source document fails; missing source IDs fail closed.

- [ ] **Step 2: Add the Review Focus legacy-input test**

Assert the CLI refuses an unversioned/legacy corpus database and prints the v2 rebuild instruction without writing a partial passing report.

- [ ] **Step 3: Run the focused test and verify failure**

Run: `python -m pytest tests/test_adaptive_vocab_corpus_audit.py -q`

Expected: FAIL because `audit_corpus` is undefined.

- [ ] **Step 4: Implement deterministic audit counting**

Count only admitted training records and their referenced source documents. Use v1 semantic-field payload base-token counts, not raw source bytes or full curriculum prompt lengths. Sort all report mappings by canonical language and stable identity.

- [ ] **Step 5: Implement the audit CLI**

Require `--database`, `--accepted-records`, `--tokenizer-artifact`, `--base-tokenizer`, `--config`, `--source-config`, `--corpus-manifest`, `--lineage`, and `--output`. Write atomically; return `0` on pass and `2` on a completed hard-stop report.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tests/test_adaptive_vocab_corpus_audit.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add bcnlt/adaptive_vocab/corpus_audit.py scripts/audit_bcnlt_adaptive_vocab.py tests/test_adaptive_vocab_corpus_audit.py
git commit -m "feat: audit multilingual BCNLT adaptive corpus diversity"
```

### Task 4: Deterministic Payload Candidate Discovery

**Files:**
- Create: `bcnlt/adaptive_vocab/candidates.py`
- Create: `bcnlt/adaptive_vocab/secrets.py`
- Create: `scripts/discover_bcnlt_macro_candidates.py`
- Create: `tests/test_adaptive_vocab_candidates.py`
- Create: `tests/test_adaptive_vocab_secrets.py`

**Interfaces:**
- Produces: `PayloadObservation(language, repository, document_id, event, field, base_ids, decoded_text)`.
- Produces: `iter_payload_observations(records: Iterable[CurriculumRecord], tokenizer: BCNLTTokenizer) -> Iterator[PayloadObservation]` for training-split canonical semantic streams only.
- Produces: `discover_candidates(observations: Iterable[PayloadObservation], config: AdaptiveVocabularyConfig) -> CandidateDiscoveryReport`.
- Produces: `MacroCandidate.expansion`, `length`, `score`, support identities, raw/capped counts, denominators, payload-class scores, eligibility, and reason codes.

- [ ] **Step 1: Write failing field-boundary and split tests**

Assert candidate windows cover lengths 2 through 16 inside one field, never join two fields/records, never include structural IDs, and ignore validation/test observations. Prove CRLF/LF and Unicode variants remain separate exact byte/base-ID sequences unless the verified base tokenizer itself maps them identically.

- [ ] **Step 2: Write failing scoring tests**

Construct known counts and assert:

```python
assert candidate.score == (length - 1) * expected_third_language_rate
assert candidate.eligible is (candidate.score >= 25)
```

Cover eight-per-document caps, three-language/repository/five-document floors, 50/10 concentration limits, descending score/support ordering, and lexical base-ID tie-breaking.

- [ ] **Step 3: Write failing secret and memorization tests**

Cover each configured credential shape, PEM headers, JWT, URI user info, 24-character hex/base64 runs, a whole 16-character literal above 3.5 bits/character, a low-entropy literal, a path/repository string, and a unique project symbol. Assert every rejection has a stable reason code and evidence without copying the secret into the report.

- [ ] **Step 4: Run tests and verify failure**

Run: `python -m pytest tests/test_adaptive_vocab_candidates.py tests/test_adaptive_vocab_secrets.py -q`

Expected: FAIL because discovery is undefined.

- [ ] **Step 5: Implement observation extraction, exclusions, and scoring**

Parse canonical semantic streams with `record_from_canonical_bytes`; obtain field payload base IDs with the verified v1 tokenizer. Hash rejected sensitive text for provenance, keep only redacted metadata, and serialize every numerator/denominator used by the exact formula.

- [ ] **Step 6: Implement candidate CLI and atomic reports**

Require a passed audit and matching lineage. Write `candidates.jsonl`, `candidate-rejections.jsonl`, and `candidate-summary.json`; use canonical order and LF endings.

- [ ] **Step 7: Run focused tests**

Run: `python -m pytest tests/test_adaptive_vocab_candidates.py tests/test_adaptive_vocab_secrets.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add bcnlt/adaptive_vocab/candidates.py bcnlt/adaptive_vocab/secrets.py scripts/discover_bcnlt_macro_candidates.py tests/test_adaptive_vocab_candidates.py tests/test_adaptive_vocab_secrets.py
git commit -m "feat: discover balanced BCNLT payload macro candidates"
```

### Task 5: Append-Only Macro Manifests and Lossless Codec

**Files:**
- Create: `bcnlt/adaptive_vocab/manifest.py`
- Create: `bcnlt/adaptive_vocab/codec.py`
- Create: `scripts/build_bcnlt_adaptive_inventories.py`
- Create: `tests/test_adaptive_vocab_manifest.py`
- Create: `tests/test_adaptive_vocab_codec.py`

**Interfaces:**
- Produces: `MacroToken(name: str, id: int, expansion: tuple[int, ...], allowed_fields: tuple[str, ...])`.
- Produces: `AdaptiveVocabularyManifest` with `schema_version`, verified `v1_manifest_sha256`, complete candidate/config/lineage hashes, `base_vocab_size`, `v1_vocab_size`, `vocab_size`, `macros`, and `manifest_sha256`.
- Produces: `build_inventories(report: CandidateDiscoveryReport, v1_manifest: TokenizerManifest, config: AdaptiveVocabularyConfig) -> tuple[AdaptiveVocabularyManifest, ...]`.
- Produces: `MacroCodec.substitute(base_ids: Sequence[int], field: str) -> list[int]` and `MacroCodec.expand(ids: Sequence[int]) -> list[int]`.

- [ ] **Step 1: Write failing budget and stable-ID tests**

Assert fewer than 256 candidates yields `insufficient_eligible_candidates`; 300 yields only 256; 700 yields 256/512; 1,024 yields all three. Assert every v1 ID stays unchanged and macro IDs are contiguous from `v1_manifest.vocab_size` in candidate rank order.

- [ ] **Step 2: Write failing codec precedence tests**

Use overlapping expansions `(1,2)`, `(1,2,3)`, and two equal expansions with different IDs. Assert longest match wins, equal length chooses lower ID, fields outside `allowed_fields` remain unchanged, and `expand(substitute(ids)) == ids`.

- [ ] **Step 3: Add corrupt-manifest Review Focus tests**

Reject changed digests, duplicate IDs/names, IDs inside v1 range, noncontiguous IDs, one-token expansions, structural/out-of-range expansion IDs, macro-to-macro expansion, cycles, unknown fields, and noncanonical JSON.

- [ ] **Step 4: Run focused tests and verify failure**

Run: `python -m pytest tests/test_adaptive_vocab_manifest.py tests/test_adaptive_vocab_codec.py -q`

Expected: FAIL because the manifest and codec are undefined.

- [ ] **Step 5: Implement canonical manifests and trie-based substitution**

Build the trie from base-token expansions only. Keep `expand` a direct macro-ID lookup. Name macros `<|bcnlt:macro:0000|>` through the selected budget width; names are display/embedding definitions and expansions are authoritative.

- [ ] **Step 6: Implement inventory CLI**

Require candidate/config/lineage/v1 hashes. Write each budget to an immutable sibling directory through a temporary directory; write an explicit no-pass JSON and return `2` when fewer than 256 candidates qualify.

- [ ] **Step 7: Run focused tests**

Run: `python -m pytest tests/test_adaptive_vocab_manifest.py tests/test_adaptive_vocab_codec.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add bcnlt/adaptive_vocab/manifest.py bcnlt/adaptive_vocab/codec.py scripts/build_bcnlt_adaptive_inventories.py tests/test_adaptive_vocab_manifest.py tests/test_adaptive_vocab_codec.py
git commit -m "feat: build lossless append-only BCNLT macro inventories"
```

### Task 6: Experimental Adaptive BCNLT Tokenizer Artifact

**Files:**
- Create: `bcnlt/adaptive_vocab/tokenizer.py`
- Create: `bcnlt/adaptive_vocab/hf_artifact.py`
- Create: `tests/test_adaptive_vocab_tokenizer.py`
- Modify: `bcnlt/corpus/packing.py`
- Modify: `tests/test_corpus_packing.py`

**Interfaces:**
- Produces: `AdaptiveBCNLTTokenizer(v1: BCNLTTokenizer, manifest: AdaptiveVocabularyManifest)`.
- Produces: v1-compatible `encode_text`, `encode_raw_code`, `encode_records`, `encode_source`, `decode`, and `decode_records`; exposes adaptive `manifest.vocab_size` and `manifest.manifest_sha256` to packing/training.
- Produces: `save_adaptive_tokenizer_artifact(path, tokenizer)` and `load_adaptive_tokenizer_artifact(path, base_path)`.
- Consumes: Task 5 `MacroCodec`; downstream tasks can use the existing `pack_training_shards` protocol without a second packer.

- [ ] **Step 1: Write failing delegation and semantic substitution tests**

Assert text/raw-code ID sequences exactly equal v1, semantic payload runs use macros, no macro crosses a structural token, and decoding expands before v1 validation.

- [ ] **Step 2: Add exact-surface Review Focus tests**

Cover empty optional fields, composed/decomposed Unicode, emoji, CRLF/LF, final newlines, quotes/backslashes, and literal text `<|bcnlt:event:call|>`. Assert canonical bytes and decoded records are exact.

- [ ] **Step 3: Write failing artifact and packing tests**

Assert save/load preserves manifest and v1 hashes, refuses the wrong base/v1 artifact, rejects unknown/out-of-range macro IDs, and packs/reloads adaptive shards with the adaptive digest and vocabulary ceiling.

- [ ] **Step 4: Run focused tests and verify failure**

Run: `python -m pytest tests/test_adaptive_vocab_tokenizer.py tests/test_corpus_packing.py -q`

Expected: FAIL because the wrapper does not exist.

- [ ] **Step 5: Implement wrapper by transforming v1 semantic streams**

Call v1 `encode_records`, split only contiguous IDs below `base_vocab_size` between structural IDs, and apply the field-allowed codec. Expand every macro to v1 IDs before calling any v1 decoder. Delegate text/raw code unchanged.

- [ ] **Step 6: Implement self-contained artifact save/load**

Copy the verified v1 artifact into a nested immutable `v1/` directory, write the adaptive manifest beside it, and verify both hashes on reload. Do not alter `bcnlt/vocab/bcnlt-tokenizer-v1.tokens.json`.

- [ ] **Step 7: Generalize packing only to the existing tokenizer protocol**

Replace concrete v1 manifest assumptions with the already-used `manifest_sha256`/`vocab_size` protocol. Keep packed schema compatibility and prove v1 packed bytes remain unchanged.

- [ ] **Step 8: Run focused and regression tests**

Run: `python -m pytest tests/test_adaptive_vocab_tokenizer.py tests/test_tokenizer.py tests/test_hf_tokenizer.py tests/test_corpus_packing.py -q`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add bcnlt/adaptive_vocab/tokenizer.py bcnlt/adaptive_vocab/hf_artifact.py bcnlt/corpus/packing.py tests/test_adaptive_vocab_tokenizer.py tests/test_corpus_packing.py
git commit -m "feat: add experimental adaptive BCNLT tokenizer artifact"
```

### Task 7: Tokenizer-Only Benchmark and Inventory Selection

**Files:**
- Create: `bcnlt/adaptive_vocab/tokenizer_eval.py`
- Create: `scripts/run_bcnlt_adaptive_tokenizer_eval.py`
- Create: `tests/test_adaptive_vocab_tokenizer_eval.py`

**Interfaces:**
- Produces: `evaluate_tokenizer(v1, adaptive, train_records, heldout_records, config, *, clock, memory_probe) -> AdaptiveTokenizerEvaluation`.
- Produces: `select_inventory(evaluations: Sequence[AdaptiveTokenizerEvaluation], config) -> InventorySelection`.
- Produces: per-language counts, bytes/token, macro coverage, round-trip results, mode identity, p50/p95 latency, peak memory, platform artifact digests, gate failures, and canonical reports.

- [ ] **Step 1: Write failing metric and gate tests**

Use injected clocks/memory probes. Assert exact aggregate compression math, per-language no-expansion, p95 calculation, 10 percent compression/latency boundaries, zero-failure round-trip gate, unchanged mode gate, and stable failure codes.

- [ ] **Step 2: Write failing smallest-near-best selection tests**

Assert the smallest passing inventory within one percentage point of the best compression wins; a smaller inventory farther away loses; no passing inventory yields `no_inventory_passed` and no training eligibility.

- [ ] **Step 3: Write failing cross-platform identity tests**

Compare supplied Windows/WSL artifact digests. Equal canonical artifacts pass; any byte difference yields `cross_platform_artifact_mismatch`. Keep performance measurements platform-specific and never compare clocks across hosts.

- [ ] **Step 4: Run focused tests and verify failure**

Run: `python -m pytest tests/test_adaptive_vocab_tokenizer_eval.py -q`

Expected: FAIL because evaluation is undefined.

- [ ] **Step 5: Implement evaluation and CLI**

Warm each codec before timing, use the same ordered records for all inventories, report training metrics separately from untouched heldout metrics, and write one report per budget plus `selected-inventory.json` atomically.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tests/test_adaptive_vocab_tokenizer_eval.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add bcnlt/adaptive_vocab/tokenizer_eval.py scripts/run_bcnlt_adaptive_tokenizer_eval.py tests/test_adaptive_vocab_tokenizer_eval.py
git commit -m "feat: gate BCNLT adaptive tokenizer inventories"
```

### Task 8: Predictive Boundary Entropy Diagnostic

**Files:**
- Create: `bcnlt/adaptive_vocab/boundary_entropy.py`
- Create: `scripts/run_bcnlt_boundary_entropy.py`
- Create: `tests/test_boundary_entropy.py`
- Modify: `bcnlt/hf_tokenizer.py`
- Modify: `tests/test_hf_tokenizer.py`

**Interfaces:**
- Produces: `HuggingFaceBaseTokenizer.encode_with_offsets(text: str) -> tuple[list[int], list[tuple[int, int]]]` using the verified fast tokenizer and its Unicode character offsets.
- Produces: `token_entropies(logits: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> tuple[float | None, ...]` where index `i` predicts target `i+1`.
- Produces: `collect_boundary_report(*, v4_model, base_model, v1_tokenizer, raw_tokenizer, heldout_documents, output) -> BoundaryEntropyReport`.
- Produces: separate primary `v4_adapter_semantic_v1` and control `runeforge_base_raw_code` records; no mutating dependency on wildcard memory or admission.

- [ ] **Step 1: Write failing numerical entropy tests**

Assert known uniform logits equal `log2(vocab_size)`, a one-hot distribution approaches zero, and results use bits. Cover batched shapes and nonfinite logits.

- [ ] **Step 2: Add alignment Review Focus tests**

Assert prediction position `i` attaches to target `i+1`; final input has no next-target value; padding, zero attention, and label `-100` yield `None`; AST/field boundary flags attach to the target token, not the predictor token.

- [ ] **Step 3: Write failing offset and nonmutation tests**

Assert Unicode character offsets select the original substring exactly. Snapshot learned-definition, corpus, tokenizer, and candidate files before a dummy diagnostic and assert their hashes remain unchanged afterward.

- [ ] **Step 4: Run focused tests and verify failure**

Run: `python -m pytest tests/test_boundary_entropy.py tests/test_hf_tokenizer.py -q`

Expected: FAIL because entropy and offsets are undefined.

- [ ] **Step 5: Implement pure entropy/alignment functions and offset support**

Use `log_softmax`/probabilities in float32 under inference mode. Require a fast tokenizer for offsets and fail clearly if unavailable. Keep correlation/precision/recall calculations deterministic and model-specific.

- [ ] **Step 6: Implement the two-model diagnostic CLI**

Load the frozen v4 adapter for semantic v1 streams and the preserved Runeforge base for raw-code controls. Require exact lineage and heldout hashes; write only `boundary-entropy.jsonl` and its summary beneath the new run root.

- [ ] **Step 7: Run focused tests**

Run: `python -m pytest tests/test_boundary_entropy.py tests/test_hf_tokenizer.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add bcnlt/adaptive_vocab/boundary_entropy.py scripts/run_bcnlt_boundary_entropy.py tests/test_boundary_entropy.py bcnlt/hf_tokenizer.py tests/test_hf_tokenizer.py
git commit -m "feat: measure BCNLT predictive boundary entropy"
```

### Task 9: Matched Continuation Dataset and Extension Growth

**Files:**
- Create: `bcnlt/adaptive_vocab/training.py`
- Create: `scripts/build_bcnlt_adaptive_training_pair.py`
- Create: `tests/test_adaptive_vocab_training.py`
- Modify: `integrations/runeforge_mk2_tokenizer/embedding_extension.py`
- Modify: `integrations/runeforge_mk2_tokenizer/train_bcnlt_adapter.py`
- Modify: `tests/test_embedding_extension.py`
- Modify: `tests/test_bcnlt_training_preflight.py`

**Interfaces:**
- Produces: `select_paired_training_records(records, manifest, *, minimum_macro_occurrences=8, maximum_examples=10000, seed=1729) -> PairedTrainingSelection`.
- Produces: `grow_bcnlt_extension(model, old_manifest, new_manifest, definitions, *, seed) -> None`, preserving every old embedding/head row exactly and initializing only appended macro rows.
- Produces: `run_bounded_training_trace(..., maximum_optimizer_steps=512, maximum_epochs=1, validation_interval=64, validation_callback=...) -> TrainingTrace`.
- Preserves: existing `run_training_steps(...) -> list[float]` behavior by delegating to the bounded loop and returning its per-sample losses, so current smoke callers remain compatible.
- Produces: paired v1/v5 packed datasets with identical ordered record IDs and target-byte sidecars.

- [ ] **Step 1: Write failing deterministic selection tests**

Assert every retained macro has eight available occurrences, unreachable macros are removed and force inventory reevaluation, selected examples are identical for v1/v5, language/family balancing is deterministic, the cap is 10,000, and holdouts never appear.

- [ ] **Step 2: Write failing extension-growth tests**

Create a tiny model and assert all base and old v1 input/output rows remain bit-identical after growth, only new rows depend on macro definitions and seed, gradients touch permitted extension/LoRA weights only, and shrinking/reordering manifests fails.

- [ ] **Step 3: Write failing paired-loop tests**

Use a tiny CPU test harness to assert one epoch/512-step termination, validation at steps 64 and final, identical record order and optimizer settings across arms, finite-loss enforcement, and exact resume identity. Keep the production CUDA requirement in the CLI preflight.

- [ ] **Step 4: Run focused tests and verify failure**

Run: `python -m pytest tests/test_adaptive_vocab_training.py tests/test_embedding_extension.py tests/test_bcnlt_training_preflight.py -q`

Expected: FAIL because selection/growth/traces are undefined.

- [ ] **Step 5: Implement paired selection and sidecar manifests**

Write selected record IDs, language/family counts, macro counts, canonical target UTF-8 byte lengths, input hashes, and selection configuration. Pack the exact same record order once with v1 and once with v5.

- [ ] **Step 6: Implement safe extension growth and bounded traces**

Load v4 into a v1 model first, clone its state for both arms, then grow only the v5 copy. Extract the common optimizer loop behind `run_bounded_training_trace`; keep `run_training_steps` as a compatibility wrapper with its existing signature and return type.

- [ ] **Step 7: Implement paired-dataset CLI**

Require the selected tokenizer report and exact v4/corpus lineage. Produce two packed dataset directories and a shared paired-selection manifest; do not start training.

- [ ] **Step 8: Run focused tests**

Run: `python -m pytest tests/test_adaptive_vocab_training.py tests/test_embedding_extension.py tests/test_bcnlt_training_preflight.py -q`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add bcnlt/adaptive_vocab/training.py scripts/build_bcnlt_adaptive_training_pair.py tests/test_adaptive_vocab_training.py integrations/runeforge_mk2_tokenizer/embedding_extension.py integrations/runeforge_mk2_tokenizer/train_bcnlt_adapter.py tests/test_embedding_extension.py tests/test_bcnlt_training_preflight.py
git commit -m "feat: prepare matched BCNLT adaptive continuation training"
```

### Task 10: Paired Model Evaluation and Promotion Proposal

**Files:**
- Create: `bcnlt/adaptive_vocab/model_eval.py`
- Create: `scripts/run_bcnlt_adaptive_model_eval.py`
- Create: `tests/test_adaptive_vocab_model_eval.py`
- Modify: `bcnlt/tokenizer_eval.py`
- Modify: `tests/test_tokenizer_eval.py`

**Interfaces:**
- Produces: `evaluate_nll_bits_per_byte(model, dataset, target_byte_lengths, *, maximum_examples=None) -> NLLReport` with total NLL bits, total canonical bytes, and bits/byte.
- Produces: `evaluate_adaptive_matrix(frozen_v4, untrained_v5, v1_control, trained_v5, suite, paired_heldout, config) -> AdaptiveModelEvaluation`.
- Produces: `evaluate_adaptive_promotion(report, tokenizer_selection) -> PromotionDecision`, defaulting to `promoted=False` and never merging/publishing.

- [ ] **Step 1: Write failing bits-per-byte tests**

Use fixed logits and byte lengths; assert total NLL is summed before division, identical target hashes are required, empty/zero byte targets fail, `-100` labels do not contribute NLL, and per-token loss is reported but cannot drive promotion.

- [ ] **Step 2: Write failing matrix and gate tests**

Require all four named model states, identical prompts/settings/heldout order, current category thresholds, exact Unicode/final newline, no AST/parseability regression, v5 bits/byte no worse than frozen v4 and matched v1 control, 10 percent compression, finite traces, and a reloadable adapter. Assert every failure has a stable code.

- [ ] **Step 3: Run focused tests and verify failure**

Run: `python -m pytest tests/test_adaptive_vocab_model_eval.py tests/test_tokenizer_eval.py -q`

Expected: FAIL because adaptive evaluation is undefined.

- [ ] **Step 4: Implement shared-unit NLL and matrix evaluation**

Reuse current frozen generation cases and `decode_generated_stream`. Hash identical canonical target bytes before model evaluation; reject mismatched datasets rather than normalizing incomparable losses.

- [ ] **Step 5: Implement evaluation CLI**

Load the frozen v4 adapter, attach untrained v5 macros, load both continued adapters, verify all lineage, run the matrix, and write `model-evaluation.json` plus `promotion-proposal.json`. The proposal records passing evidence but remains false until a later explicit promotion action.

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tests/test_adaptive_vocab_model_eval.py tests/test_tokenizer_eval.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add bcnlt/adaptive_vocab/model_eval.py scripts/run_bcnlt_adaptive_model_eval.py tests/test_adaptive_vocab_model_eval.py bcnlt/tokenizer_eval.py tests/test_tokenizer_eval.py
git commit -m "feat: evaluate matched BCNLT adaptive model controls"
```

### Task 11: Resumable WSL Experiment Runner and Operator Documentation

**Files:**
- Create: `bcnlt/adaptive_vocab/runner.py`
- Create: `scripts/run_bcnlt_adaptive_vocab_v5.py`
- Create: `scripts/run_bcnlt_adaptive_vocab_v5.ps1`
- Create: `tests/test_adaptive_vocab_runner.py`
- Modify: `docs/bcnlt-runeforge-training.md`
- Modify: `README.md`

**Interfaces:**
- Produces: `plan_experiment(inputs: ExperimentInputs) -> tuple[ExperimentPhase, ...]` with ordered phases `lineage`, optional `preliminary_corpus_rebuild`, `audit`, `candidates`, `inventories`, `tokenizer_eval`, `entropy`, `paired_data`, `paired_training`, and `model_eval`.
- Produces: `run_phase(phase, *, resume: bool) -> PhaseResult` with exact input/output hashes and atomic status.
- Produces: PowerShell wrapper that resolves the repository into WSL and invokes the Python runner without embedding dynamic shell fragments.

- [ ] **Step 1: Write failing runner-state tests**

Cover fresh run, read-only legacy-to-v2 preliminary rebuild, preliminary-audit hard stop, no-inventory hard stop, tokenizer-gate hard stop, successful phase ordering, interruption before rename, resume with exact hashes, refusal after changed config/code/input, and refusal to use any protected output path.

- [ ] **Step 2: Write failing command-construction tests**

Assert paths containing spaces are passed as argument-list elements, no token or source text enters command logs, `--start-training` is absent by default, and training phases require both a passing tokenizer selection and the explicit flag.

- [ ] **Step 3: Run focused tests and verify failure**

Run: `python -m pytest tests/test_adaptive_vocab_runner.py -q`

Expected: FAIL because the runner is undefined.

- [ ] **Step 4: Implement phase planner, hash-checked resume, and CLI**

Keep orchestration in Python. Let the PowerShell file perform only WSL path resolution, executable checks, and argument forwarding. Return `0` for completed pass, `2` for a valid research hard stop, and `1` for corruption/runtime failure.

- [ ] **Step 5: Document the experiment boundary**

Add the exact preliminary audit command, qualifying-corpus prerequisite, tokenizer-only command, GPU opt-in, expected artifacts, failure codes, and the statement that no merge/full run/publish occurs. Link the design and this plan.

- [ ] **Step 6: Run focused and full verification**

Run: `python -m pytest tests/test_adaptive_vocab_runner.py -q`

Expected: PASS.

Run: `wsl.exe -d Ubuntu -- /home/bossm/.venvs/wizard-phi-eval/bin/python -m pytest -q`

Expected: all repository tests pass, except any already-documented platform skip.

Run: `git diff --check`

Expected: no whitespace errors.

- [ ] **Step 7: Run the non-GPU preliminary audit only**

Run the documented command against a read-only current pinned Python-focused corpus database. First rebuild it into a new v2 database beneath the v5 output root and regenerate the audit curriculum there; do not mutate or reuse the legacy database as writable state.

Expected: canonical `corpus-audit.json` with `insufficient_shared_candidate_diversity`; no candidate inventory, adapter, or model files are created.

- [ ] **Step 8: Commit**

```bash
git add bcnlt/adaptive_vocab/runner.py scripts/run_bcnlt_adaptive_vocab_v5.py scripts/run_bcnlt_adaptive_vocab_v5.ps1 tests/test_adaptive_vocab_runner.py docs/bcnlt-runeforge-training.md README.md
git commit -m "feat: orchestrate the BCNLT adaptive vocabulary experiment"
```

## Final Verification and Handoff

- [ ] Confirm the preserved Runeforge checkpoint, v1 tokenizer artifact, v4 corpus, v4 adapter, and v4 evidence hashes are unchanged from the captured lineage.
- [ ] Confirm the branch contains only reviewed source, tests, configuration, plans/docs, and the intended preliminary audit metadata; keep generated run artifacts beneath the external v5 run root.
- [ ] Run `git status --short`, `git log --oneline --decorate -15`, and the full WSL test suite; record exact results in the implementation handoff.
- [ ] Request one whole-branch review focused on append-only IDs, lossless decoding, heldout isolation, protected paths, matched-control fairness, and the entropy/knowledge-uncertainty boundary.
- [ ] Do not start qualifying multilingual acquisition, GPU continuation training, merging, promotion, or publication without their own satisfied prerequisites and explicit task scope.
