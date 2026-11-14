# BCNLT Scientific Vocabulary Learning

## Intent

BCNLT should accumulate encounters with an unknown construct, propose a versioned
interpretation, test that interpretation against independently authored coding
outcomes, refine it when contradicted, and reuse it only after reproducible
success. Repetition starts and informs learning; repetition alone is never proof.

This loop updates BCNLT's Extended Core library. It does not train language-model
weights and it never mutates the universal Common Core. A validated meaning receives
a stable, versioned Extended Core ID. Its tested Common Core event/IR interpretation
is retained only as an optional decomposition, not used as the meaning's identity.
New model-facing IDs remain a separate adaptive-vocabulary and embedding/output-head
change.

## Architecture

1. Wildcard memory accumulates sightings across genuinely distinct contexts and
   keeps a versioned interpretation and Common Core decomposition hypothesis.
2. An externally authored fixture supplies a literal expected coding result and
   trusted oracle provenance. It is separate from extractor output and from the
   hypothesis under test.
3. A bounded JSON-only verifier executes a small declarative projection program.
   The hypothesis fills the wildcard operation slot and therefore causally affects
   the result. No Python source, generated code, subprocess, import, filesystem, or
   network operation is executed.
4. Every experiment runs the candidate plus a deliberately wrong projection. The
   candidate must produce the expected result and the wrong projection must be
   rejected or produce a different result.
5. The implementation default requires three non-replayed passing experiments in
   distinct observed contexts with distinct fixtures and no active contradiction.
   This is configurable and is not a user-mandated scientific constant.
6. Acceptance is scoped to the exact signature, hypothesis revision, language,
   semantic-spec hash, verifier version, and tested fixture domain. A later verified
   failure immediately revokes acceptance. A revised projection starts with fresh
   evidence.
7. Persistence is append-only. Load recomputes eligibility from integrity-checked
   experiment evidence instead of trusting a serialized `validated` flag.

## Supported bounded operations

The first verifier supports literal branch selection, numeric loop summation,
returning a literal value, assigning a literal JSON value, bounded binary operators,
and raising a literal error category. Inputs are finite JSON with strict size,
depth, collection, and step limits. Unsupported operations and invalid payloads
fail closed.

## Extended Core behavior

Provisional entries remain evidence only and encode as `UNKNOWN_SEMANTIC`.
Scientifically validated entries enter `ExtendedCoreLibrary`. UYR preserves the
original `UNKNOWN_SEMANTIC` event, emits `IRKind.EXTENSION`, and attaches the stable
Extended Core ID, revision, and optional Common Core decomposition as metadata.
Reload preserves the evidence and decision. Revocation removes that Extended Core
metadata from future encodings; token IDs remain valid because no ID allocation
changed.

The append-only evidence ledger remains the persisted source of truth. The
`ExtendedCoreLibrary` is derived from its currently validated records at load time,
so there is no second mutable registry that can drift from the scientific evidence.

## Trust and safety boundaries

- Fixture authorship/oracle provenance is trusted input and cannot be proven solely
  by software. The interface rejects extractor- and hypothesis-derived origins and
  records exact fixture/oracle digests for audit.
- The verifier establishes only the tested declarative behavior, not universal
  semantic correctness.
- Arbitrary code execution is not supported. A real language sandbox and broader
  conformance corpus remain future work.
- Extractor-derived meaning/model evaluations remain diagnostics and are never
  admitted as scientific-learning evidence.
- Caller-supplied `correct=True` outcome records remain provisional forever.

