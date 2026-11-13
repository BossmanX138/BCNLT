# BCNLM multilingual repair handoff integration

Date: 2026-11-12

## Input identity

- Remote source: `D:\BossCrafts Dev\BossmanX138\BCNLM\artifacts\wizard-phi-handoff-20261112\BCNLM-multilingual-repair-83-tests.zip`
- Local preserved copy: `imports/bcnlm-repair-handoff-20261112/BCNLM-multilingual-repair-83-tests.zip`
- ZIP SHA-256: `75f3956665d3f1382aa3bb9b6a2dadfa997b8aaa25e705ec5ab93425d4a0dfdd`
- Base commit: `27521f4bce5426712edc5ed92adca961dfa9ae28`
- Snapshot inventory: 176 files, all hashes and byte counts verified
- Repair overlay: 15 files, all hashes verified

The archive was read from BossCrafts MSI over SSH after PATISBORg confirmed the
path and remote hash. It was extracted into `imports/`; no archive member escaped
the selected extraction directory.

## Selective-port decision

No production source was copied from the older BCNLM runtime into V10.

- `adapters/symbols.py` is already byte-identical to the verified handoff:
  `ffa1d7824252a474df3f70c7f969fd1ca85faec16bbbdfee7a7f9c877b6173cf`.
- V10 `adapters/tree_sitter.py` is a later superset with SHA-256
  `45d6d4e415ef479cc8a8b70ad14ca6b49eb1be17b6873a885f851c1918ff8e57`.
- V10 `common_core_registry.py` is a later superset with SHA-256
  `3aa5a816c82c23ffe3d201f4df9e7e87f1e97817f3dcb81187d21be6c1f9a9af`.
- The handoff's repaired JavaScript/TypeScript arrow ownership and concise
  return, Rust tail return, Go binding/scope, Python class ownership, C/C++
  block shadowing, and JavaScript/Ruby parameter behaviors all pass against the
  V10 runtime.
- The handoff `vendor/runtime-files.json` was intentionally not copied because
  it identifies the older V2 bundle and would invalidate V10's contained
  runtime identity and later learning/safety changes.

The immutable handoff tests are connected to V10 through
`tests/bcnlm_handoff_runtime_plugin.py`. The plugin changes only the parser
transport boundary; it does not change the handoff fixtures, selection logic,
or assertions.

## Fresh verification evidence

1. The extracted handoff's own complete suite: **83 passed in 24.89 seconds**.
2. The six repaired-behavior handoff test modules against V10:
   **29 passed, 57 subtests passed in 17.19 seconds**.
3. V10 native suite at handoff-integration time:
   **232 passed, 120 subtests passed in 83.53 seconds**.

The V10 suite now uses `pytest.ini` to keep immutable evidence below `imports/`
out of native project discovery. Two stale lineage assertions were corrected:
semantic preparation now pins V10 parser identity
`ed65443f8468baafa5d9d087489c60eed83e9d386cfce97cba6e06c86c24d19b`,
and the copied V9 post-training controller is required to fail closed when run
against the modified V10 tree.

The identity and test count above are historical handoff evidence. Subsequent
isolated V10 architecture work added sandbox-attested execution, immutable
Extended Core version control, and held-out uncertainty calibration. The
current identity and regression result are recorded in
`docs/bosscrafts-model-architecture-status-20261112.md`; this handoff record is
not rewritten to imply those later bytes existed during the comparison.

## Remaining limits

This result verifies parser/semantic compatibility only. It does not establish
complete language semantics, model-quality improvement, compiler correctness,
or permission to resume model training. The handoff's documented gaps remain:
complex parameters, additional implicit/expression returns, richer JavaScript
ownership/aliases, remaining Go declarations/scopes, advanced Python lexical
rules, and cross-file/type semantics.

