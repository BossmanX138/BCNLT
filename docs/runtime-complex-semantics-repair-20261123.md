# Runtime complex-semantics repair — 2026-11-23

## Scope

This repair closes concrete semantic-extraction defects found by executing the
contained V10 runtime against nontrivial parameter and implicit-return forms.
It does not claim complete semantics for every construct in all 14 grammars.

## Repaired bindings

- Python default, typed, typed-default, splat, typed-splat, and lambda parameters.
- TypeScript optional and rest parameters.
- C and C++ pointer, array, and reference parameter declarators.
- C# `params` arrays and implicit lambda parameters.
- Java varargs and inferred lambda parameters.
- Rust tuple, struct, tuple-struct, reference, mutable, and slice patterns.
- PHP variadic parameters.
- Ruby lambda parameters.
- Swift closure parameters.

Every emitted binding uses the narrow identifier span. Default expressions,
types, field labels, and constructor names are not promoted to definitions.
References in the repaired fixtures resolve to the emitted symbol identity.

## Repaired implicit returns

- Python lambdas.
- C# expression-bodied methods and expression lambdas.
- Java expression lambdas.
- Rust expression closures and valid block-tail expressions.
- PHP arrow functions.
- Ruby methods and lambdas.
- Kotlin expression-bodied functions and expression-valued lambda tails.
- Swift single-expression functions and expression-valued closure tails.

Declaration, jump, explicit-return, and semicolon-terminated tails are excluded
where the language does not treat them as implicit returned values.

## Independent witness

`raw_ast_witness_sidecar.py` independently reconstructs every new synthetic
binding and return relation from the pinned Tree-sitter grammar. It includes
negative controls proving that Python default expressions are not parameter
bindings and rejects unsupported Rust pattern shapes rather than guessing.

## Sealed runtime identity

- Source aggregate SHA-256:
  `d7468d5e308a7aa1d4f4bbb304ae2d93f237d67c067fca2d5976a2269f1938e5`
- Parser runtime identity SHA-256:
  `18992265d9cc6337aac12cab6c88307cbe5a1b855c39844e7681c6f7df2885b8`
- Runtime manifest SHA-256:
  `c976fa02440d2b78bfcf0275a795ad01386d64c372ce4c75242304adaeb7ce01`
- PHP semantic-spec SHA-256:
  `04fb81c46fce64995672f19bd48b7b57ff0f74f845794fefdcab740dd6728f11`

The PHP spec changed because `arrow_function` is now a declared function scope.
Other semantic-spec hashes are unchanged; their behavior change is bound by the
new complete parser-runtime identity.

## Verification

- Focused runtime semantics: 4 tests, 46 subtests passed.
- Focused witness, lineage, integrity, and tokenizer requirements: 74 passed.
- Repository Ruff: passed.
- Repository mypy: 241 source files passed.
- Full non-Docker regression: 709 passed, 21 Docker-only tests deselected,
  210 subtests passed in 706.57 seconds.

The Docker-only group was not counted as passed. An accidental run reached a
Docker image-inspection timeout; the established non-Docker gate was rerun from
zero and completed successfully.

## Remaining explicit boundaries

- The pinned Swift grammar can parse some closure arithmetic as a single broad
  `simple_identifier` surface. The return owner is proven, but inner operand
  reference resolution for those grammar shapes remains unavailable.
- Rust or-pattern bindings are left unresolved instead of being emitted as
  duplicate definitions; a future repair needs an alternative-consistency
  contract.
- These repairs do not retroactively change already admitted or packed semantic
  corpora. Existing artifacts retain their old runtime identity and require a
  verified migration or regeneration before current-runtime model training.
