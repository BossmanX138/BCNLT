# BossCrafts V10 exhaustive code audit

> Historical checkpoint: this document records the 2026-11-12 audit at the
> commit named below. It is superseded for current status, schemas, counts, and
> remaining work by `docs/pre-model-code-audit.md`. It must not be used as
> authorization to resume training.

Date: 2026-11-12

Audited repair commit: `628f74071e7031ec9bb4ade6b4a6b2579fe6d28c`

## Scope

The audit covered the complete `bosscrafts_model` package (56 source files),
the scientific sandbox, production scripts, evidence gates, semantic audit
sidecar and client, curriculum builders, model/configuration code, and the
repository test suite. Generated corpora, model weights, databases, caches,
and derived audit artifacts remain excluded from Git.

## Executed evidence

- `python -m compileall -q bosscrafts_model sandbox scripts tests`: passed.
- `python -m mypy bosscrafts_model --check-untyped-defs`: 56 source files,
  zero errors.
- Ruff correctness rules `E9,F63,F7,F82`: passed.
- Ruff bug-pattern rules `F,B,PLE,PLW`: passed.
- Bandit medium/high scan over `bosscrafts_model`, `sandbox`, and `scripts`:
  zero findings.
- Bandit all-severity review: 74 low-confidence reports were examined. They
  consist of schema values misclassified as passwords, deliberate cleanup
  suppression, `subprocess` imports, and subprocess calls that do not use a
  shell. No additional vulnerability was identified.
- Final complete regression: 329 tests and 133 subtests passed in 170.93
  seconds.
- `git diff --check`: passed before the repair commit.

## Material repairs

- Added a global disk-backed prompt-claim index to prevent conflicting or
  cross-split prompt targets from escaping bounded curriculum sampling.
- Prevented local semantic-review requests from following redirects.
- Replaced production assertions with explicit fail-closed checks.
- Bounded raw-AST sidecar response transport and rejected backward spans.
- Added strict schema validation for semantic replay, challenge, calibration,
  qualification, task, and model-acceptance evidence.
- Made CodeX workers use the project-owned, hash-verified tokenizer runtime.
- Preserved floating-point curriculum weights and made zip truncation policy
  explicit.
- Re-attested the scientific sandbox images after runner changes.
- Removed ambiguous import-time model configuration construction and narrowed
  PyTorch module initialization/cache paths.

## Regenerated semantic evidence

The raw-AST challenge artifact was regenerated after the sidecar repair. The
combined qualification still passed with all 140 negative controls detected
and zero false flags across 140 positive controls. The regenerated
qualification report SHA-256 is
`87ce37f0c4ba2ca4974ceb9a579c2b6164bda04988a91abe80590dd13167deca`.

## Mandatory lineage consequence

The repaired raw-AST sidecar, `resumable_current_corpus.py`, and
`training_corpus_freeze.py` changed code identities used by frozen training
evidence. The previously completed admission database must not be treated as
current-lineage training authorization. Rebuild the qualification/freeze and
full admission evidence from their preserved source inputs before task packing
or any model training.

Training and model insertion remain held until that fresh evidence passes its
own verification gates.

## Independent review

The repository is ready for CodeRabbit review. The CLI is installed and sees
the Git repository, but the independent review is pending interactive
CodeRabbit authentication. This report does not claim that CodeRabbit has run.
