# Contained runtime semantic-coverage repair — 2026-11-16

## Boundary

This repair changes the isolated, project-contained BCNLT runtime only. It does
not modify the upstream BCNLM handoff, the frozen tokenizer vocabulary, Common
Coding Core concept IDs, structural token IDs, or any previously published
admission/audit artifact.

The old runtime remains historical evidence:

- parser runtime identity: `7440ef6263537ded8bceec9a3d2fdb72cbaf7a324e6deea552dc18e4d6891dfc`
- source aggregate: `d791c0eb80c9a65191be3b15877f6a7153e434bbe5f8cd4a6060f57af2fca14e`

The repaired candidate is pinned as:

- parser runtime identity: `5b8f0b043d8d749449b468fd479564aba6193785081336f7d011f3c5079f3511`
- source aggregate: `d04cd2cd17d7c53498a11b12e5593d6bcb5c81ee8c6373e4843cc03006eb0e1a`
- runtime manifest SHA-256: `bbdeee4bf553a0fd53d79d3dd709e7563e0f9efb2c59b62e2c873003445b042d`

Any admission database, semantic audit, calibration evidence, packed task
database, or curriculum carrying the old parser identity is not transferable
to the repaired candidate. It must be rebuilt under the new identity.

## Repaired semantics

The repair maps additional concrete grammar forms onto existing universal
binding, reference, and scope concepts. It does not introduce language-specific
concepts into Common Core.

1. Go grouped parameters emit one definition per declared name.
2. Go `range` declarations emit one definition per left-hand binding, while
   range assignments remain references to existing definitions.
3. Ruby destructured, optional, keyword, splat, hash-splat, and block
   parameters emit their actual bindings. Default expressions are evaluated
   before the parameter currently being initialized.
4. Bash positional and special parameters are known implicit references, not
   definitions invented at each use site. The canonical metadata contract now
   carries the bounded Boolean `implicit_binding` marker.
5. JavaScript and TypeScript class static initialization blocks own `var`
   bindings; those bindings neither hoist to module scope nor escape the static
   block.

The affected semantic-spec hashes are:

- Bash: `ab4732dda01f958ec3f06bfb88a8d5b95879e226143bb01a5474587460ee3b36`
- JavaScript: `45cabfe829b613869c0fc49a72fb78a5e58dd9b3d7449fb61560d6e593253c64`
- TypeScript: `26a18950bff376a5403e59d24bef1a469e213f19316cb9556e730a587e394f9b`

Go and Ruby retain their prior declarative spec hashes because their repairs
are grammar-aware repeated-field and binding-pattern execution inside the
identity-pinned semantic engine. The complete runtime identity captures those
engine changes.

## Verification

- Five original defect tests failed before the repair and passed afterward.
- Negative controls cover Go range assignment and Ruby default-expression
  ownership.
- The independent raw-AST witness covers repeated Go parameter fields, Go range
  declaration operators, Ruby parameter binding ownership, and JavaScript
  static-block scope.
- Focused regression: 86 tests and 64 subtests passed before the additional
  negative controls were added.
- Final focused regression: 91 tests and 64 subtests passed.
- Final full regression: 689 tests and 198 subtests passed in 740.12 seconds.
- Ruff passed for the project; strict mypy passed 238 source files; changed
  vendor files passed Ruff undefined-name and syntax checks.

Fresh corpus admission and semantic audits remain required before this runtime
can authorize packing or training.
