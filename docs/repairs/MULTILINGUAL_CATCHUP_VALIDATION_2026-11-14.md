# Multilingual catch-up validation (2026-11-14)

This source-only preservation branch is based on multilingual commit 4857b8e.
It does not merge main, audit branches, or the separate PATISBORG catch-up branch.
36 of the 54 recovered source/config/test/document files differ from the separate
main-worktree snapshot; shared repairs are retained to keep this branch coherent.

Included: semantic/provenance repairs, scientific verification, SQLite learned-state
snapshots, corpus admission/packing fixes, strict adapter selection, source registry
hash updates, and validation-recovery tooling. No corpus records, model checkpoints,
generated outputs, compiled Unsloth caches, or credentials are included.

Validation using the existing Python environment in an isolated worktree:
- Full suite: 583 passed, 2 failed (both C# parser tests).
- Both failures reproduce on untouched baseline 4857b8e because the installed
  parser download manifest has no `c_sharp` entry. These are not newly introduced.
- All staged Python files parse. Secret scan matched only mocked `test-key`.
- No production training, corpus validation/recovery, or weight merge was launched.
- The recovery CLI is preserved but has not been exercised on real artifacts;
  it can truncate partial output to checkpoint offsets and must only be used with
  verified backups and matching artifacts. Publication is not operational approval.

Original dirty working trees remain untouched. This is a review/preservation
branch, not a merged release or a claim of fully green validation.
