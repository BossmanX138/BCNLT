# BCNLT correctness repair — 2026-11-06

## Scope and source state

No model, adapter, training output, evaluation artifact, branch, or remote was merged, trained, published, deleted, or overwritten.

- Current main checkout started and remains at `ceec7e2402f0da456be0b8686ebac1f0a44bd84c` (48 commits ahead of `origin/main` at `623a5f49d8637ebdc6253af48b0b8093c7a09351`).
- Multilingual worktree started and remains at `4857b8ec0d765c4ecf50da352aa610b1571c0ffb`.
- The multilingual worktree already had user-owned changes in `bcnlt/corpus/admission.py`, `tests/test_corpus_admission.py`, `scripts/recover_bcnlt_validation.py`, and `unsloth_compiled_cache/`. They were preserved and not overwritten.
- No commits, pushes, or branch merges were made.

## Findings and disposition

1. Ordinary definition/reference links: confirmed and fixed in both source states.
2. Wildcard cached guesses self-validating: confirmed. Repetition and caller-supplied pass metadata can no longer promote a guess. Direct metadata-only validation is disabled; legacy autonomous validation records load as provisional.
3. Mutation operators collapsing: confirmed and fixed; update nodes retain their operator surface.
4. JavaScript block scope and `var`: confirmed and fixed; lexical declarations remain block-scoped and `var` binds in the enclosing function/module.
5. Saved tokenizer omitted learned semantic state: confirmed and fixed. The actual tokenizer JSONL dependency is copied and verified by version, storage kind, SHA-256, and byte count.
6. Truncated wildcard JSONL tail: confirmed and fixed with byte-accurate truncation before the next append.
7. Adaptive wrapper dropped language/dialect: confirmed and fixed. Main still fails clearly for non-Python because it is Python-only; the feature tokenizer forwards multilingual language/dialect.
8. Python semantic adapter defects: confirmed and fixed for while-else, local/global/nonlocal binding, and first-iterable comprehension scope.
9. Ruby ordinary calls emitted imports: confirmed and fixed; only `require`, `require_relative`, and `load` project as imports.
10. Duplicate Unicode spans overcounted boundaries: confirmed and fixed.
11. Full-run gate trusted hashes without evaluation/command identity: confirmed and fixed with evaluation schema/content checks, artifact identity, and canonical command SHA-256.
12. Candidate selector accepted nonfinite values and loose booleans: not present on current main; confirmed and fixed on the multilingual feature branch.
13. Native-Windows path test assumed WSL: fixed with platform-specific alias assertions.
14. Packing rejection could be masked by an open SQLite temporary database on Windows: fixed with guaranteed connection cleanup.

## Wildcard scientific-method boundary
The original repair deliberately stopped at provisional bookkeeping because no
independent verifier existed. That safeguard remains: repeated sightings,
document renames, and caller-supplied `correct=True` outcomes can never validate
a guess, and metadata-only `validate()` remains disabled.

The bounded verifier implemented on 2026-11-07 is now the sole promotion path.
It requires independently authored expected results, projection-dependent
execution in the JSON-only interpreter, a rejected wrong-projection control,
three distinct successful contexts by default, and no contradiction. Failures
and hypothesis revisions revoke or reset eligibility. Validation is limited to
the tested construct/domain; it is not universal semantic proof.


## Meaning-evaluation evidence boundary

The meaning-retrieval and model-context evaluators were audited after the semantic repairs. Their probes, target line/event/symbol identity, and BCNLT retrieval candidates are all derived from the same extractor output. They are therefore useful self-consistency and context-format diagnostics, but they are not independent semantic-correctness measurements.

A regression now constructs a fabricated `DEFINE` unit for source that contains no such definition and demonstrates that the internally consistent BCNLT path still records a hit. Report contracts now disclose:

- `evaluation_kind`;
- `target_origin: same_extractor_output`;
- `independent_semantic_oracle: false`;
- `promotion_eligible: false`;
- canonical retrieval/extractor-target hit-rate names, while retaining legacy `*_accuracy` fields for compatibility.

The head-to-head script no longer declares a semantic winner. It reports only an extractor-derived diagnostic leader and explicitly states that the result is not promotion evidence. Runeforge and Wizard-Phi integration smoke evidence carries the same non-promotion boundary. No current tokenizer, adaptive-vocabulary, or full-run promotion gate consumes these diagnostics.

That boundary is now implemented below with independently authored fixtures and deliberately wrong projection controls. Extractor-derived meaning/model scores remain ineligible.

## Scientific vocabulary-learning implementation — 2026-11-07

Implemented in both repaired source states:

- a versioned `HypothesisProjection` and externally authored
  `CodingExperimentFixture`;
- a bounded JSON-only verifier whose candidate projection fills the operation slot
  and whose required wrong-projection control must not reproduce the expected result;
- exact source, context, fixture, oracle, hypothesis, semantic-spec, verifier, result,
  and evidence digests;
- a configurable default of three independent reproductions, with replay
  deduplication and distinct-context/document/fixture requirements;
- fresh evidence after hypothesis revision and immediate revocation after any later
  verified failure, invalid control, unavailable verifier, or unsupported experiment;
- integrity-checked JSONL persistence and matching feature-worktree SQLite behavior;
- load-time recomputation that ignores forged serialized validation flags and keeps
  legacy caller-supplied `correct=True` records provisional;
- a derived `ExtendedCoreLibrary` with stable versioned definition IDs, explicit
  live reload, frozen-artifact persistence, and contradiction-driven revocation;
- immutable Common Core semantics: validated outliers remain `UNKNOWN_SEMANTIC` with
  `IRKind.EXTENSION`, while tested Common Core interpretations are stored only as
  optional decomposition metadata.

The verifier supports bounded branch selection, loop summation, literal return,
literal assignment, allowlisted numeric/comparison operations, and literal error
signaling. It never executes source/generated code and has no subprocess, import,
filesystem, or network capability.

This updates BCNLT Extended Core state, not Common Core or language-model weights.
It does not allocate or renumber token IDs. A genuinely new model-facing ID still
requires a separately reviewed adaptive vocabulary plus compatible embedding/output
rows and is outside this repair.


## Verification

Red/green regression development was used for each repair group.

- Main, unfiltered: `.venv\Scripts\python.exe -m pytest -q` → **468 passed**.
- Multilingual feature, unfiltered: main venv running feature tests → **585 passed**.
- `git diff --check` passed in both worktrees. The feature check reports only Git's existing LF-to-CRLF conversion warning for the selector script and its test.
- `python -m compileall -q bcnlt scripts` passed in both worktrees.

The checkouts intentionally remain dirty with the repairs plus the pre-existing multilingual corpus-recovery work.

