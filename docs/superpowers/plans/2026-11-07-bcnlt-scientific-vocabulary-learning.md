# BCNLT Scientific Vocabulary Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded, reproducible scientific-learning loop whose verified interpretations enter BCNLT's Extended Core without changing Common Core, existing token IDs, or model weights.

**Architecture:** A JSON-only verifier evaluates externally authored fixtures with the candidate projection and a required wrong-projection control. Wildcard memory persists versioned, integrity-checked experiment evidence and derives current validation from three independent passing contexts by default. Validated meanings receive stable Extended Core IDs; Common Core interpretations remain optional decomposition metadata.

**Tech Stack:** Python dataclasses, canonical JSON/SHA-256, existing wildcard JSONL/SQLite stores, pytest.

**Spec:** `docs/superpowers/specs/2026-11-07-bcnlt-scientific-vocabulary-learning-design.md`

## Global Constraints

- Do not execute arbitrary or generated code.
- Do not train, merge, promote, or modify model weights/adapters.
- Do not change existing tokenizer IDs or their meanings.
- Keep caller-supplied coding outcomes provisional.
- Mirror behavior in main and `.worktrees/bcnlt-multilingual-corpus` while preserving feature SQLite behavior and user-owned changes.
- Do not commit, push, or merge.

## Review Focus

- Replayed or renamed evidence must not count as independent reproduction.
- Serialized validation flags and experiment evidence must not be forgeable without detection.
- Hypothesis revision and later contradiction must remove tokenizer eligibility.
- Wrong controls that also satisfy the fixture must invalidate the experiment.
- Unsafe, oversized, unsupported, or unavailable-verifier cases must fail closed.

---

### Task 1: Bounded projection verifier

**Files:**
- Create: `bcnlt/scientific_learning.py`
- Create: `tests/test_scientific_learning.py`

**Interfaces:**
- Produces: immutable projection/fixture/result types and `BoundedProjectionVerifier.verify(...)`.

- [x] Write failing tests for safe successful operations, wrong projections, nondiscriminating controls, unsupported/unsafe inputs, verifier unavailability, and exact digests.
- [x] Run the focused tests and confirm feature-missing failures.
- [x] Implement the minimal JSON-only verifier and canonical identity helpers.
- [x] Run the focused tests green in both worktrees.

### Task 2: Versioned scientific evidence and decision policy

**Files:**
- Modify: `bcnlt/wildcard.py`
- Modify: `tests/test_wildcard_verified_outcomes.py`

**Interfaces:**
- Produces: `WildcardMemory.run_scientific_experiment(...)`, versioned hypothesis state, append-only experiment evidence, configurable reproduction policy, and load-time decision recomputation.

- [x] Write failing tests for one success, three independent successes, replay, mismatched/stale evidence, revision freshness, later contradiction, failed controls, persistence, and forged state.
- [x] Run the focused tests and confirm feature-missing failures.
- [x] Implement minimal persistence and state derivation while preserving legacy provisional outcomes.
- [x] Run wildcard/learning tests green in both worktrees, including feature SQLite reload.

### Task 3: Tokenizer-visible Extended Core

**Files:**
- Modify: `bcnlt/tokenizer.py`
- Modify: `tests/test_semantic_preprocessing_pipeline.py`
- Modify: `tests/test_hf_tokenizer.py`

**Interfaces:**
- Produces: an auditable Extended Core library and end-to-end encoding that preserves `UNKNOWN_SEMANTIC` plus `IRKind.EXTENSION` without changing Common Core IDs.

- [x] Write failing tests showing provisional unknown encoding, validated projection encoding, stable IDs across reload, and contradiction-driven reversion.
- [x] Run the focused tests and confirm expected failures.
- [x] Expose validated learned entries without adding aliases or IDs; rely on existing UYR projection for encoding.
- [x] Run tokenizer/preprocessing tests green in both worktrees.

### Task 4: Documentation and final verification

**Files:**
- Modify: `README.md`
- Modify: `docs/repairs/BCNLT_CORRECTNESS_REPAIR_2026-11-06.md`

- [x] Document repetition versus experiments, vocabulary versus weight learning, supported operations, trusted oracle boundary, and future work.
- [x] Run unfiltered test suites, compile checks, PowerShell parsing, and `git diff --check` in both worktrees.
- [x] Inspect final diffs and report exact revisions, dirty-state preservation, commands, outputs, and remaining limitations.

