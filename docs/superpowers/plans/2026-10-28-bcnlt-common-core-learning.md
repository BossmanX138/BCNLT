# BCNLT Common Core Learning Implementation Plan

> Supersedes Tasks 6-11 of `2026-10-28-bcnlt-multilingual-adapter-carousel.md`.

1. Freeze the Common Core concept catalog in one versioned registry and test
   uniqueness, stable identity, allowed events/IR kinds, and all 14 grammar
   projections.
2. Replace per-language builders with one `build_semantic_spec(language)`
   loader backed by concept-centered declarative mapping data.
3. Keep one `TreeSitterSemanticEngine`; route the public adapter through the
   registry and retain `PythonAdapter` only as a compatibility facade.
4. Extend wildcard memory into an append-only learned-definition store with
   provisional and validated states, target event/IR kind, evidence, and
   grammar/spec identity. Add autonomous promotion from multiple independent
   evidence channels and contexts; identical repetition must not promote.
   Promote irreducible language oddities into stable named extensions while
   shared meanings project into the Common Core.
5. Make UYR retrieve learned definitions. Provisional matches provide context;
   validated matches project the unknown unit into its recorded Common Core
   concept with an auditable evidence trail.
6. Consolidate parse, uncertainty ratings, UYR, wildcard learning, and final
   canonical scores in one semantic preprocessing pipeline; route the CLI,
   database, tokenizer, and corpus builder through it.
7. Add shared grammar fixtures and coverage tests without creating language
   implementation modules.
8. Resume provenance, curriculum, evaluation, smoke, and full-run work only
   after the core registry and learned-definition loop pass the full regression
   suite.
