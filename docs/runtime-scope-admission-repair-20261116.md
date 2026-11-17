# Runtime scope-name and zero-record admission repair

Date: 2026-11-16

## Evidence that triggered the repair

The complete frozen evaluation audits under parser runtime
`5b8f0b043d8d749449b468fd479564aba6193785081336f7d011f3c5079f3511`
processed every committed validation and test document.

- Test: 3,207 documents, 1,673,365 semantic records, 23 findings across
  19 records, and four source-level preprocessing rejects.
- Validation: 3,219 documents, 1,762,805 semantic records, eight findings
  across seven records, and two source-level preprocessing rejects.
- Every raw-AST finding was `raw_surface_mismatch` on a scope exit whose
  exact scope name contained a dot. The affected observed forms were
  JavaScript/TypeScript computed method names and Swift qualified extension
  names.
- The five coverage findings were files containing comments only. They were
  admitted even though semantic extraction emitted zero records.
- Direct diagnostic replay established that the six source-level rejects
  exceeded the deliberate 16,384 semantic-record transport bound. They were
  bounded fail-closed rejections, not unexplained parser crashes.

The partial train admission under the old runtime was stopped and preserved
after the runtime defect was confirmed. It must not be resumed or mixed with
the repaired lineage.

## Root causes and corrections

The Tree-sitter engine reconstructed a scope-exit surface with
`symbols.scope.rsplit(".", 1)[-1]`. This assumed that a scope name could not
contain a dot. The engine now retains the exact name used for `SCOPE_ENTER`
and reuses that name for `SCOPE_EXIT`.

Current-corpus admission now rejects a successful parse that produces zero
semantic records with the explicit reason `no_semantic_records`. The policy
is identical in pilot, threaded resumable, and process-worker resumable
paths. The resumable database schema is therefore V5; the read-only progress
reader retains support for historical V1 and V4 snapshots.

No Common Coding Core concept, semantic specification, tokenizer vocabulary,
or token ID changed.

## Repaired contained-runtime identity

- parser runtime identity:
  `4cc9aac31f89be5c4e3f77ef61eabeea229dfa6db4c1dc80bd578a8f2d9a7917`
- contained source aggregate:
  `341613f6b7d732f9f6e653a3cf1ae378087aabff48755c3b39d7189893cd054e`
- contained source inventory: 79 files, 705,539 bytes
- runtime manifest SHA-256:
  `f11a9408ac6fb465a0eec62f3b0c300a7244395fbd0822407090873baba43c67`

## Verification

- Regression tests were written first and observed failing for both exact
  dotted scope-exit surfaces and zero-record admission.
- Focused runtime/admission set: 86 tests and 38 subtests passed.
- Complete warning-as-error suite: 692 tests and 202 subtests passed in
  754.75 seconds.
- Runtime identity/lineage/raw-AST subset: 44 tests passed.
- Ruff: all project checks passed; contained runtime E9/F checks passed.
- Strict mypy: 238 source files passed.

Fresh admission and semantic-audit artifacts must use a new root and the
repaired parser runtime identity. Historical roots remain immutable evidence.
