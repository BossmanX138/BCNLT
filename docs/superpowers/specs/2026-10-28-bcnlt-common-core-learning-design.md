# BCNLT Common Coding Core and Learned Definitions

## Source of truth

This design returns BCNLT to the architecture in the earliest September 20
prototype record and initial `623a5f4` commit: one language-neutral Common
Coding Core, one uncertainty/UYR/wildcard loop, and syntax front ends that
translate source trees into that shared meaning system. That record explicitly
defines BCNLT as preprocessing, calls wildcard memory a self-growing append-only
notebook, and proposes promoting repeated unknown patterns into named
extensions. It supersedes the language-carousel implementation details in the
later multilingual design. The target languages remain useful corpus inputs,
but they do not become separate BCNLT semantic engines, model adapters, token
ranges, or learned modules.

The historical record describes semantic preprocessing because that was the
v0.1 implementation boundary. The current project deliberately advances that
prototype into BCNLT's native, model-facing tokenizer. Preprocessing remains
the semantic-analysis stage inside the tokenizer; it is not the final product.

## Primary tokenizer destination

BCNLT is the primary code-tokenization path presented to Runeforge. Source is
parsed into canonical semantic records, those records are encoded into stable
BCNLT token IDs, and the model trains dedicated embedding and output rows for
those IDs. The preserved base tokenizer remains available inside the hybrid
artifact for ordinary text, raw-code compatibility, and open-ended field
payloads. That compatibility layer does not make BCNLT an auxiliary model
adapter: semantic code batches use BCNLT IDs as their actual model inputs and
labels.

The tokenizer is promotion-ready only when its artifact serializes the Common
Core vocabulary, grammar/spec lineage, and learned-definition store; reloads
to identical IDs; accepts those IDs through the model embedding and LM head;
and passes held-out semantic, raw-code, and ordinary-text regression gates.

## Architecture

The runtime has one semantic path:

`source + trusted grammar identity -> frozen parser -> shared tree walker -> Common Coding Core -> uncertainty/UYR/wildcard -> canonical BCNLT records`

The Common Coding Core owns the canonical definitions of values, variables,
parameters, functions, calls, assignments, mutable state, loops, branches,
comparisons, data structures, I/O, errors, returns, operators, member access,
imports, modules, scopes, definitions, and references. A concept is defined
once. Each grammar supplies declarative node aliases, source fields, lexical
roles, and scope boundaries that project syntax into those definitions.

Grammar maps are data loaded by one registry. There is no `javascript.py`,
`cpp.py`, or other language implementation module. Python keeps a compatibility
facade for existing callers, but new multilingual ingestion uses the same shared
engine and registry as every other grammar.

Language identity remains provenance. It selects a frozen parser and the
appropriate syntax aliases, and it is retained for evaluation and source
reconstruction. It does not change Common Core meaning or allocate new BCNLT
token IDs.

## Learned outlier loop

This loop runs in semantic preprocessing, before canonical records are admitted
or encoded for training. The shared parser and tree walker emit raw contextual
units; the preprocessor owns uncertainty ratings, UYR, wildcard learning, and
the final knowledge state. Scoring is not recomputed independently by the CLI,
database, tokenizer, and corpus builder.

Constructs that do not map to a Common Core concept enter the knowledge-boundary
loop:

1. The shared engine emits an unknown extension with source, grammar, relation,
   scope, history, and semantic-spec identity.
2. Wildcard proposes a provisional definition and appends its hypotheses,
   evidence, confidence, and sightings to the shared learned-definition store.
3. On later encounters, UYR queries local definitions, nearby semantic context,
   and this learned-definition store.
4. Provisional definitions may be injected as evidence and context, but remain
   explicitly provisional.
5. A validated definition either records a target event and Common Core IR kind
   or assigns a stable named extension to a language-specific oddity. UYR then
   injects that learned definition on matching future constructs and retains the
   learned-definition signature in its evidence trail.
6. Repetition alone never validates a definition. BCNLT promotes a definition
   autonomously when the same target meaning has independent evidence kinds
   and agrees across distinct contexts. Grammar structure, lexical-surface
   agreement, contextual agreement, successful execution/tests, and stable
   downstream predictions are eligible evidence channels. An external
   validation API remains available, but human review is not required for the
   learning loop to advance.

Independent context means evidence from at least two distinct source documents,
not merely repeated positions in one file. A genuinely language-specific named
extension additionally requires three distinct context fingerprints before
autonomous promotion.

The store is append-only. Every revision preserves the original observations,
evidence kinds, distinct-context fingerprints, validation mode,
language/grammar identity, semantic-spec hash, and target Common Core concept.
A changed grammar or spec cannot reuse an unrelated definition accidentally.

## Boundaries

- One Common Coding Core and concept registry.
- One shared parser/runtime and semantic walker.
- One semantic preprocessor that owns uncertainty, retrieval, learning, and the
  final score used by canonical records.
- One uncertainty scorer, UYR resolver, wildcard resolver, and learned store.
- One canonical BCNLT vocabulary and record schema.
- One combined Runeforge adapter/embedding extension.
- Grammar-specific data may describe syntax aliases and irreducible extensions.
- Extensions remain outside the universal core. They can become validated named
  extensions autonomously, project into an existing concept when the evidence
  supports it, or motivate a separately versioned Core change when a genuinely
  universal concept is discovered.

## Verification

- Every required concept is defined exactly once.
- Every registered grammar maps its shared syntax into the same concept IDs.
- The registry rejects unknown events, IR kinds, duplicate aliases, and maps
  that both exclude and emit the same node.
- No per-language implementation modules exist under `bcnlt/adapters`.
- A first unknown observation is provisional and persisted.
- A later UYR pass retrieves that evidence without promoting it to fact.
- Repeating the identical context remains provisional, while independent
  context and evidence channels can promote it autonomously.
- A validated definition is retrieved and projected into the recorded Common
  Core event/IR kind.
- A repeated language oddity can become a validated named extension and be
  reinjected without pretending it is universal.
- Grammar/spec identity changes prevent stale learned-definition reuse.
- Existing Python records and tokenizer IDs remain stable.
