# Admission runtime delta gate (2026-11-23)

## Purpose

The current parser repair changes the runtime identity and some language semantic
specification hashes. An old task database therefore cannot be relabeled as current,
even when many source documents produce the same semantic payload.

`bosscrafts_model.admission_runtime_delta` provides a deterministic comparison between
two completed admission databases. It is a reuse proof, not a training-readiness proof.

## Reuse rule

A document is classified as `reusable` only when every admission field except these two
runtime-provenance fields is exactly equal:

- `semantic_spec_sha256`
- `parser_runtime_identity_sha256`

The comparison includes status, rejection evidence, canonical JSONL, semantic token
payloads, all counts, source identity, language, and path. Changed, baseline-only, and
candidate-only documents are explicitly classified and cannot be reused by this proof.

## Fail-closed checks

The gate requires:

- complete, integrity-clean admission databases with no active write-ahead log;
- identical source catalog, freeze, review, tokenizer, vocabulary, and document-count
  lineage;
- distinct parser runtimes, with the candidate bound to the checked-in current runtime;
- the candidate semantic-spec map to match the checked-in current map;
- valid per-row runtime provenance and bounded decompression of admitted payloads;
- canonical payload hashes and semantic-token byte counts to validate;
- stable physical database hashes throughout the comparison.

The published JSON receipt is immutable, canonical, self-hashed, and contains the full
per-document classification inventory. Its `training_ready` field is always `false`.

## Command

```powershell
python scripts/compare_admission_runtime_delta.py `
  --baseline-database <completed-old-admission.sqlite> `
  --candidate-database <completed-current-admission.sqlite> `
  --output <new-immutable-delta-receipt.json>
```

Task reuse remains a separate gate. It must additionally prove the old task database's
full audit receipt, intersect reuse with the new eligibility inventory, bootstrap a new
task database under current metadata, and let the normal packer regenerate every
non-reusable document.
