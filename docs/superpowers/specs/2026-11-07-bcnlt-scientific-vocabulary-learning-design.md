# BCNLT Scientific Vocabulary Learning

This worktree mirrors `docs/superpowers/specs/2026-11-07-bcnlt-scientific-vocabulary-learning-design.md` from the repaired main checkout. The design requires a bounded JSON-only verifier, versioned independently reproduced evidence, required wrong-projection controls, revocation on contradiction, and tokenizer-visible Extended Core identities without Common Core or model-weight changes.

The conservative implementation default is three distinct passing experiments. Repetition alone remains insufficient, caller-supplied outcomes remain provisional, and validation is scoped to the tested construct/domain.

Validated meanings enter the versioned Extended Core library. They never mutate or
become members of the universal Common Core. The tested Common Core event/IR pair is
stored only as optional decomposition metadata, while tokenizer records preserve
`UNKNOWN_SEMANTIC` with `IRKind.EXTENSION` and a stable Extended Core ID.

The append-only evidence ledger remains the persisted source of truth. The
Extended Core library is derived from currently validated records at load time,
avoiding a second mutable registry that could drift from the evidence.

