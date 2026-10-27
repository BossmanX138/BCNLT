# BCNLT Tokenizer Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, versioned BCNLT tokenizer that assigns dedicated IDs to semantic structure while preserving every Runeforge base-token ID.

**Architecture:** Expand Python semantic coverage first, project open-ended contextual units into a closed canonical record, then encode those records with appended structural IDs and base-token payloads. Keep the core tokenizer independent of Transformers through a small protocol, with a separate Hugging Face adapter for real checkpoints.

**Tech Stack:** Python 3.10+, standard library, pytest, Hugging Face Transformers adapter

**Spec:** `docs/superpowers/specs/2026-10-27-bcnlt-hybrid-tokenizer-design.md`

## Global Constraints

- The schema identifier is exactly `bcnlt-tokenizer-v1`.
- Existing base-token IDs and base-token encode/decode behavior must remain unchanged.
- Added BCNLT IDs begin at the exact base vocabulary size and are append-only.
- Canonical JSON is UTF-8, sorted by key, compact, NFC-normalized, and LF-terminated.
- Python source is never claimed to round-trip through the AST-derived semantic representation.
- The preserved Runeforge Mk2 checkpoint is read-only.
- No artifact with a base-tokenizer hash mismatch may load.
- Python is the only language admitted in vocabulary version 1.

## Review Focus

- Nested functions, classes, lambdas, and comprehensions must enter and leave scopes without leaking definitions; Task 1 adds explicit scope tests.
- Payload text containing strings identical to BCNLT token spellings must remain payload data; Task 4 adds a collision test.
- Unicode identifiers and literals must NFC-normalize deterministically without corrupting ordinary text mode; Tasks 2 and 4 add Unicode tests.
- Corrupt manifests, duplicate IDs, unknown added IDs, and a mismatched base tokenizer must fail closed; Tasks 3 and 5 add negative tests.
- Empty programs and unsupported syntax must produce a valid bounded stream rather than silent false semantics; Tasks 1 and 4 add empty/unknown tests.

---

### Task 1: Freeze explicit Python semantic events and coverage

**Files:**
- Create: `bcnlt/events.py`
- Modify: `bcnlt/core.py`
- Modify: `bcnlt/adapters/python.py`
- Test: `tests/test_python_adapter.py`
- Test: `tests/test_python_semantic_coverage.py`

**Interfaces:**
- Produces: `SemanticEvent(str, Enum)` and `CURRENT_EVENT_VALUES: tuple[str, ...]`.
- Preserves: `PythonAdapter.parse(source: str, path: str) -> SourceDocument` and `PythonAdapter.to_units(document) -> Iterable[ContextualUnit]`.
- Produces explicit event values for existing events plus `CLASS`, `ASYNC`, `CONTEXT_ENTER`, `CONTEXT_EXIT`, `TRY`, `EXCEPT`, `FINALLY`, `RAISE`, `COMPREHENSION`, `YIELD`, `AWAIT`, and `MATCH`.

- [ ] **Step 1: Write failing event and coverage tests**

Add tests asserting stable ordered event values and explicit semantic units/scopes for classes, async functions, lambdas, context managers, exception paths, comprehensions, generators, awaits, and pattern matching. Assert an unsupported AST node emits `UNKNOWN_SEMANTIC` with `unknown_node` metadata.

- [ ] **Step 2: Run the focused tests and verify failure**

Run: `python -m pytest tests/test_python_adapter.py tests/test_python_semantic_coverage.py -q`

Expected: FAIL because the event enum and explicit visitors do not exist.

- [ ] **Step 3: Implement the stable event enum and explicit visitors**

Use `SemanticEvent` values at emission boundaries while retaining string compatibility. Each scope-producing visitor must use `try/finally` around scope-stack updates so malformed traversal cannot leak scope state.

- [ ] **Step 4: Run the focused tests**

Run: `python -m pytest tests/test_python_adapter.py tests/test_python_semantic_coverage.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add bcnlt/events.py bcnlt/core.py bcnlt/adapters/python.py tests/test_python_adapter.py tests/test_python_semantic_coverage.py
git commit -m "feat: freeze BCNLT Python semantic events"
```

### Task 2: Add the closed canonical semantic record

**Files:**
- Create: `bcnlt/semantic_record.py`
- Create: `bcnlt/canonical.py`
- Test: `tests/test_semantic_record.py`

**Interfaces:**
- Consumes: `ContextualUnit`, `ProjectIndex`, `HeuristicUncertaintyScorer`, and `SemanticEvent` from Task 1.
- Produces: `ScoreBand(str, Enum)`, `SemanticTokenRecord`, `records_from_units(units, index, scorer=None) -> tuple[SemanticTokenRecord, ...]`.
- Produces: `canonical_record_bytes(record: SemanticTokenRecord) -> bytes` and `record_from_canonical_bytes(payload: bytes) -> SemanticTokenRecord`.

- [ ] **Step 1: Write failing record projection tests**

Assert that all required fields survive canonical byte round-trip, absent optional fields become null, dictionaries sort, Unicode becomes NFC, recent history caps at eight, non-allow-listed contextual metadata is removed, and recursive IR children round-trip.

- [ ] **Step 2: Write failing score-band boundary tests**

Pin confidence bands to `low < 0.40`, `medium < 0.72`, `high >= 0.72`; pin uncertainty bands to `low <= 0.25`, `medium <= 0.60`, `high > 0.60`.

- [ ] **Step 3: Run tests and verify failure**

Run: `python -m pytest tests/test_semantic_record.py -q`

Expected: FAIL because the projection and canonical codec do not exist.

- [ ] **Step 4: Implement immutable record projection and canonical codec**

Reject missing score bands, non-finite scores, non-JSON IR values, and metadata outside the allow-list rather than silently stringifying them.

- [ ] **Step 5: Run the focused tests**

Run: `python -m pytest tests/test_semantic_record.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add bcnlt/semantic_record.py bcnlt/canonical.py tests/test_semantic_record.py
git commit -m "feat: add canonical BCNLT semantic records"
```

### Task 3: Define and validate vocabulary manifests

**Files:**
- Create: `bcnlt/vocab/bcnlt-tokenizer-v1.tokens.json`
- Create: `bcnlt/tokenizer_base.py`
- Create: `bcnlt/tokenizer_manifest.py`
- Test: `tests/test_tokenizer_manifest.py`

**Interfaces:**
- Consumes: `CURRENT_EVENT_VALUES` from Task 1.
- Produces: `BaseTokenizer` protocol with `vocab_size`, `encode(text, add_special_tokens=False)`, `decode(ids)`, and `get_vocab()`.
- Produces: `BaseTokenizerIdentity`, `TokenizerManifest`, `build_manifest(base, identity)`, `load_manifest(path, base)`, and `write_manifest(path, manifest)`.

- [ ] **Step 1: Write the ordered structural-token inventory**

Include stream/mode boundaries, record boundaries, every Task 1 event, typed field delimiters, null/boolean/list separators, confidence/uncertainty bands, and the five curriculum task boundaries. Token spellings use the reserved `<|bcnlt:...|>` namespace.

- [ ] **Step 2: Write failing manifest tests**

Assert the first new ID equals `base_vocab_size`, IDs are contiguous and unique, event coverage is exact, the manifest digest is stable, and base-vocabulary collisions are rejected.

- [ ] **Step 3: Add negative tests**

Assert load failure for a wrong base hash, duplicate ID, changed digest, missing required token, unknown schema, and non-contiguous IDs.

- [ ] **Step 4: Run tests and verify failure**

Run: `python -m pytest tests/test_tokenizer_manifest.py -q`

Expected: FAIL because manifest support does not exist.

- [ ] **Step 5: Implement manifest construction and validation**

Hash the source tokenizer files and canonical manifest separately. Never infer identity from a directory name.

- [ ] **Step 6: Run the focused tests**

Run: `python -m pytest tests/test_tokenizer_manifest.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add bcnlt/vocab/bcnlt-tokenizer-v1.tokens.json bcnlt/tokenizer_base.py bcnlt/tokenizer_manifest.py tests/test_tokenizer_manifest.py
git commit -m "feat: define BCNLT tokenizer vocabulary manifest"
```

### Task 4: Implement deterministic hybrid encoding and decoding

**Files:**
- Create: `bcnlt/tokenizer.py`
- Test: `tests/test_tokenizer.py`

**Interfaces:**
- Consumes: `BaseTokenizer`, `TokenizerManifest`, `SemanticTokenRecord`, `records_from_units`, and the canonical codec.
- Produces: `BCNLTTokenizer(base: BaseTokenizer, manifest: TokenizerManifest)`.
- Produces: `encode_text(text)`, `encode_raw_code(source)`, `encode_records(records)`, `encode_source(source, path)`, `decode(ids)`, and `decode_records(ids)`.

- [ ] **Step 1: Write failing mode and round-trip tests**

Use a deterministic fake base tokenizer. Assert unchanged text/raw-code IDs, exact record round-trip, multiple-record boundaries, empty semantic streams, Unicode payloads, and payloads containing literal `<|bcnlt:...|>` spellings.

- [ ] **Step 2: Write malformed-stream tests**

Assert failure for truncated fields, an unexpected mode marker, an added ID absent from the manifest, duplicated fields, and base IDs outside the declared base vocabulary.

- [ ] **Step 3: Run tests and verify failure**

Run: `python -m pytest tests/test_tokenizer.py -q`

Expected: FAIL because `BCNLTTokenizer` does not exist.

- [ ] **Step 4: Implement explicit-mode encoding and state-machine decoding**

Call the base tokenizer with `add_special_tokens=False` for every payload. Detect appended structural IDs numerically; never parse payload text for token spellings.

- [ ] **Step 5: Run the focused tests**

Run: `python -m pytest tests/test_tokenizer.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add bcnlt/tokenizer.py tests/test_tokenizer.py
git commit -m "feat: implement BCNLT hybrid tokenizer"
```

### Task 5: Add Hugging Face persistence and inspection CLI

**Files:**
- Create: `bcnlt/hf_tokenizer.py`
- Create: `bcnlt/tokenizer_cli.py`
- Modify: `pyproject.toml`
- Test: `tests/test_hf_tokenizer.py`
- Test: `tests/test_tokenizer_cli.py`

**Interfaces:**
- Consumes: `BCNLTTokenizer` and manifest APIs from Tasks 3-4.
- Produces: `HuggingFaceBaseTokenizer`, `BCNLTTokenizer.save_pretrained(path)`, and `BCNLTTokenizer.from_pretrained(path, base_path)`.
- Produces CLI commands `build`, `encode`, `inspect`, and `verify` under the `bcnlt-tokenizer` entry point.

- [ ] **Step 1: Write failing persistence and CLI tests**

Use a temporary local tokenizer fixture. Assert saved artifact contents, reload equality, manifest verification, JSON inspection output, and nonzero exits for corrupted artifacts and base mismatches.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_hf_tokenizer.py tests/test_tokenizer_cli.py -q`

Expected: FAIL because the adapter and CLI do not exist.

- [ ] **Step 3: Implement the Hugging Face adapter and artifact layout**

Store `bcnlt_tokenizer_manifest.json`, `bcnlt-tokenizer-v1.tokens.json`, base-tokenizer file hashes, and no model weights. Load with `trust_remote_code=False` and `local_files_only=True` unless the CLI build command is explicitly given a Hub revision.

- [ ] **Step 4: Implement the CLI without changing the existing `bcnlt` CLI contract**

The `verify` command returns success only after digest, required-token, base-hash, and round-trip checks all pass.

- [ ] **Step 5: Run the focused tests**

Run: `python -m pytest tests/test_hf_tokenizer.py tests/test_tokenizer_cli.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add bcnlt/hf_tokenizer.py bcnlt/tokenizer_cli.py pyproject.toml tests/test_hf_tokenizer.py tests/test_tokenizer_cli.py
git commit -m "feat: persist and inspect BCNLT tokenizer artifacts"
```

### Task 6: Verify foundation behavior and document the contract

**Files:**
- Modify: `README.md`
- Create: `docs/bcnlt-tokenizer-v1.md`
- Test: `tests/test_readme_architecture.py`

**Interfaces:**
- Consumes: every tokenizer foundation API.
- Produces: a documented v1 artifact contract and reproducible verification commands.

- [ ] **Step 1: Add documentation assertions**

Assert that README names the schema, explicit modes, base-ID preservation, semantic round-trip boundary, and the artifact verification command.

- [ ] **Step 2: Write the contract documentation**

Document token classes, canonical field order, score bands, artifact files, mode behavior, failure modes, and Python-source non-reversibility.

- [ ] **Step 3: Run the complete test suite**

Run: `python -m pytest -q`

Expected: all tests pass.

- [ ] **Step 4: Build and verify a local fixture artifact twice**

Run the `bcnlt-tokenizer build` command twice against the same fixture base tokenizer and compare manifest SHA-256 values.

Expected: byte-identical manifests and identical encoded IDs.

- [ ] **Step 5: Commit**

```powershell
git add README.md docs/bcnlt-tokenizer-v1.md tests/test_readme_architecture.py
git commit -m "docs: publish BCNLT tokenizer v1 contract"
```
