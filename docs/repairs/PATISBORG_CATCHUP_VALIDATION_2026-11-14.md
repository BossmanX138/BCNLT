# PATISBORG catch-up validation (2026-11-14)

This branch preserves the source-only pending PATISBORG work based on ceec7e2.
It does not merge the newer audit branches or claim production readiness.

Included: correctness and provenance repairs, scientific vocabulary verification,
Extended Core artifacts/token IDs, native-foundation experiment tooling, and tests.
Excluded: recovered staging files, corpora, checkpoints, generated outputs and caches.
Original working trees were not changed.

Validation in an isolated worktree using the existing Python 3.13 environment:
- 487 tests passed; one C# grammar projection test failed because the installed
  tree-sitter-language-pack download manifest has no `c_sharp` grammar entry.
- The identical C# failure reproduces on untouched baseline ceec7e2.
- The native-foundation trainer test module was intentionally not executed.
- All staged Python files parse successfully. No model training was launched.
- Pattern-based secret review found only a dummy `test-key` in a mocked test.
- Existing extra blank lines at EOF remain in several newly recovered source/docs files.

This is a preservation/review branch, not a merged or fully green release.
