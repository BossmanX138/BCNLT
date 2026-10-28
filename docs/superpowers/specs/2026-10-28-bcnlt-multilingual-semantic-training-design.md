# BCNLT Multilingual Semantic Training Design

## Intent

BCNLT will expand from its Python-first semantic corpus pipeline into a native,
multilingual coding-semantic training system for Runeforge Mk2 7B. The target is
semantic parity across Python, JavaScript, TypeScript, C, C++, C#, Java, Go,
Rust, PHP, Ruby, Kotlin, Swift, and Shell. For every admitted language, BCNLT
must identify definitions, references, calls, scopes, control flow, imports,
operators, member access, literals, and uncertainty through the same canonical
record contract.

This is not a raw-code expansion disguised as semantic data. A language does not
enter the full Runeforge training corpus merely because a parser can produce a
syntax tree. It enters only after its language adapter passes semantic coverage,
golden-fixture, provenance, admission, and mixed-batch training gates.

The existing Runeforge Mk2 7B checkpoint remains immutable. All multilingual
work produces new corpus artifacts, adapters, and derived checkpoints with exact
lineage back to that preserved core and to BCNLT tokenizer v1.

## Goals

- Give all 14 selected languages first-class BCNLT semantic coverage rather than
  Python semantics plus raw-code fallbacks.
- Reuse the existing 73 BCNLT structural tokens and preserve all 32,000 original
  Runeforge token IDs. Language identity is record provenance, not a new token-ID
  range.
- Use one shared parser/runtime layer and explicit per-language semantic maps.
- Produce deterministic canonical records whose source language, parser,
  grammar, source revision, license, and adapter version are auditable.
- Preserve lexical scopes and local definition/reference links without claiming
  whole-program type resolution.
- Quarantine malformed, unsupported, ambiguous, unlicensed, or holdout-colliding
  inputs with stable reason codes.
- Build a balanced multilingual corpus with at least 10,000 admitted packed
  training records per language, at least 140,000 total packed records, and at
  least 50,000,000 encoded training tokens.
- Prove that mixed-language BCNLT sequences train on the RTX 3070 before the
  full corpus run is authorized.
- Evaluate each language independently as well as in a combined aggregate, while
  retaining raw-code and ordinary-language regression checks.

## Non-goals

- Full compiler correctness, complete type inference, or cross-repository symbol
  resolution.
- Byte-for-byte reconstruction of formatting, comments, or preprocessor output
  from semantic records.
- Executing untrusted corpus code to determine its meaning.
- Treating parser success alone as proof of semantic coverage.
- Adding languages beyond the selected 14 in this version.
- Replacing the existing BCNLT structural vocabulary or renumbering any token.
- Overwriting the preserved Runeforge Mk2 7B checkpoint.
- Using a model-generated interpretation where deterministic syntax and lexical
  analysis can supply the answer.

## Anti-goals

- One generic tree walk that labels nearly every named node
  `UNKNOWN_SEMANTIC`.
- Silently routing unsupported languages through the Python adapter.
- Combining JavaScript and TypeScript, or C and C++, under one reported language
  when their grammars and semantic rules differ.
- Training on raw snippets whose language, repository, license, grammar identity,
  or source revision is missing.
- Letting Python dominate the corpus because it already has the most mature
  adapter and an additional NVIDIA source.
- Starting the large Runeforge adaptation while any selected language still
  fails its independent admission gate.

## Current state and constraints

BCNLT's `CommonCodingCore`, event vocabulary, canonical semantic record, corpus
database, curriculum, tokenizer, and packing layers are already mostly
language-neutral. The concrete `PythonAdapter`, Python file discovery, Python
license filter, and Python-oriented source extractors are language-specific.
`LanguageAdapter.render()` is currently mandatory even though the training
pipeline reconstructs code from bounded original-source spans and does not need
every adapter to regenerate code from IR.

Tokenizer v1 is already frozen at 32,073 IDs: 32,000 preserved Runeforge IDs and
73 appended BCNLT structural IDs. The multilingual design must use that artifact
unchanged. Language identity belongs in the corpus record and manifest. Where a
language-specific construct needs IR representation, its extension name is
namespace-prefixed, such as `rust:match_arm` or `cpp:preprocessor_if`, and its
metadata must obey the canonical JSON contract.

The Stack at pinned revision
`17cad72c886a2858e08d4c349a00d6466f54df63` has partitions for all 14 selected
languages. NVIDIA OpenCodeInstruct is admitted only as a supplemental Python
source because the current verified extractor and test-status rules produce
Python code. Dataset licenses and attribution remain attached to every document;
the Runeforge model's distribution terms do not replace upstream dataset terms.

Authenticated Hugging Face streaming works on Windows. WSL network access to
Hugging Face is currently obstructed by Cloudflare WARP, while WSL CUDA and the
training runtime work. The supported pipeline therefore ingests and caches
source data on Windows, then consumes content-addressed shards from WSL for
tokenization and training.

## Approaches considered

### 1. Full semantic parity for every language — selected

Build a shared Tree-sitter runtime and a dedicated semantic specification for
each language. Every selected language receives the same required event classes,
lexical scope tracking, symbol links, uncertainty handling, provenance, tests,
and promotion gates. This takes the most work, but it produces one coherent
BCNLT corpus and prevents weak languages from entering training as mislabeled
semantic data.

### 2. Tiered semantic coverage

Keep Python at full parity, give common languages definitions/calls/scopes, and
admit the rest primarily as raw code. This would generate a large corpus sooner,
but training examples with the same BCNLT mode would carry unequal meaning. It
does not satisfy the requirement to make BCNLT natively useful across all common
languages.

### 3. Raw multilingual corpus with later semantic upgrades

Feed all languages through the original tokenizer as raw code and add semantics
later. This is the cheapest expansion and may improve ordinary multilingual code
exposure, but it does not train the BCNLT semantic tokenizer across languages.

The user selected approach 1.

## Parser and grammar foundation

The parser layer uses `tree-sitter-language-pack==1.22.1` as the distribution
mechanism and the official `tree-sitter` Python bindings as the parse-tree API.
The language pack provides Windows and Linux wheels and an on-demand grammar
cache; the bindings expose grammar ABI compatibility and error/missing-node
state. The implementation must not trust a mutable on-demand cache as corpus
identity.

Before adapters are built, a capability probe downloads exactly these grammar
IDs into a staging cache: `python`, `javascript`, `typescript`, `c`, `cpp`,
`c_sharp`, `java`, `go`, `rust`, `php`, `ruby`, `kotlin`, `swift`, and `bash`.
The probe records the resolved catalog entry, upstream grammar repository,
grammar revision if exposed, ABI version, platform artifact SHA-256, and complete
cache-file SHA-256 list. The cache is then copied into a content-addressed,
read-only BCNLT parser bundle. Windows ingestion and WSL validation must resolve
the same bundle manifest. A missing language, ABI mismatch, changed cache hash,
or platform discrepancy fails closed before corpus generation.

The distribution choice is based on the project's published Python support for
on-demand cached grammars and broad language coverage. Tree-sitter's Python
documentation states that generated-language ABI compatibility is backward but
not forward compatible, which is why runtime, grammar, ABI, and binary hashes
all become corpus identity. References:

- [tree-sitter-language-pack Python package](https://github.com/xberg-io/tree-sitter-language-pack/blob/main/packages/python/README.md)
- [tree-sitter-language-pack 1.22.1 release files](https://pypi.org/project/tree-sitter-language-pack/)
- [official Python Tree-sitter bindings](https://github.com/tree-sitter/py-tree-sitter)

The package release is a delivery mechanism, not semantic authority. BCNLT owns
the language maps and golden fixtures described below.

## Adapter architecture

### One adapter with a language carousel

The public integration is one `BCNLTMultilingualAdapter` connected to the whole
BCNLT tokenizer, corpus, and training pipeline. Inside it, a `LanguageCarousel`
selects one of the 14 semantic cartridges from trusted source metadata. The
cartridges contain grammar IDs and language-specific node mappings; they are not
separate model adapters or separately trained LoRAs.

The flow is therefore:

`source record -> language carousel -> shared semantic engine -> canonical BCNLT records -> one multilingual Runeforge adapter`.

`PythonAdapter` remains as a compatibility facade that selects the Python
cartridge, so existing Python callers and regression tests continue to work.
New corpus code calls the single multilingual facade with an explicit language
ID. Runeforge receives one multilingual LoRA/embedding extension trained across
the combined admitted corpus, not one LoRA per language.

### Shared runtime

`TreeSitterAdapter` implements the language-neutral part of
`LanguageAdapter`:

1. Load a language only through the frozen parser-bundle manifest.
2. Parse UTF-8 source and preserve byte and point spans.
3. Reject trees containing parser `ERROR` or `MISSING` nodes.
4. Walk named nodes in deterministic source order.
5. Maintain an explicit lexical-scope stack.
6. Apply a versioned `LanguageSemanticSpec` to emit contextual units.
7. Build stable local symbol IDs and definition/reference links.
8. Emit explicit uncertainty for language features that cannot be resolved
   lexically.

The adapter never guesses a language from source text. The source registry
selects a language ID, grammar ID, file-extension allowlist, and semantic-spec
version explicitly.

### Per-language semantic specifications

Each language has a checked-in semantic specification containing:

- grammar and adapter identity;
- recognized definition nodes for modules, namespaces/packages, types, methods,
  functions, parameters, local variables, fields/properties, and imports;
- scope-enter and scope-exit rules;
- call, member-access, assignment, return, branch, loop, operator, literal, and
  import mappings;
- identifier roles that distinguish declarations, references, labels, keys,
  types, attributes, and literal-like names;
- lexical name-resolution rules and shadowing behavior;
- language-specific uncertainty rules;
- IR extension mappings for constructs without a common-core equivalent; and
- a deterministic semantic-spec version and content hash.

Shared helpers may be used for language families, but every reported language
has its own spec, golden fixtures, metrics, and promotion decision. TypeScript
may extend JavaScript mappings, C++ may extend C mappings, and Kotlin may share
JVM helpers with Java, but inherited mappings are included in each resolved spec
hash.

### Required parity contract

Every language must emit all common BCNLT event classes that exist in valid
fixtures for that language:

- `DEFINE`, `REFERENCE`, `ASSIGN`, `CALL`, and `RETURN`;
- `BRANCH` and `LOOP`;
- `MEMBER_ACCESS`, `IMPORT`, `OPERATOR`, and `LITERAL`;
- balanced `SCOPE_ENTER` and `SCOPE_EXIT`; and
- `UNKNOWN_SEMANTIC` for valid constructs that are traversed but not yet mapped.

Parity means the contract and quality bar are equal; it does not mean every
language has identical syntax. Shell functions and variable expansions, C
preprocessor branches, Rust lifetimes and macros, PHP namespace imports, Swift
closures, and Kotlin lambdas receive language-appropriate rules and uncertainty.

### Symbol resolution and uncertainty

Version 1 resolves definitions and references inside one source document through
lexical scopes and explicit import/namespace information. It does not infer
runtime dispatch, overload selection, macro expansion, reflection, dynamic
property access, or cross-file types. Unresolved references retain a null
definition link and an uncertainty marker; they are never linked to a plausible
name by guesswork.

The canonical metadata allowlist expands in a new semantic-record
canonicalization version to include language-neutral uncertainty categories,
for example `dynamic_call`, `dynamic_binding`, `macro_expansion`,
`overload_unresolved`, `preprocessor_condition`, and `unknown_node`. Language
identity itself stays on the document/example envelope so identical semantics do
not acquire different structural token IDs solely because of language.

### Rendering contract

`LanguageAdapter.render()` becomes an optional capability rather than an
abstract requirement. Corpus semantic-to-code examples use the original bounded
source span associated with a semantic unit and retain its language envelope.
An adapter may implement IR rendering for diagnostics, but failure to render
arbitrary IR does not block otherwise valid semantic extraction.

## Corpus model and provenance

### License admission

The existing corpus allowlist remains authoritative for The Stack:
`MIT`, `Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`, `ISC`, `0BSD`,
`Unlicense`, and `CC0-1.0`. A row with a missing, custom, mixed, dual, `AND`, or
`OR` license expression is rejected even when one component is permissive.
BossCrafts-owned material is admitted only with the explicit `owner_provided`
designation. NVIDIA OpenCodeInstruct retains its source-specific CC-BY-4.0 rule
and attribution; that exception does not broaden The Stack allowlist.

### Source registry

The source registry has one immutable entry per language/source pairing. Each
entry records:

- provider, dataset, configuration/partition, and immutable revision;
- canonical BCNLT language ID and grammar ID;
- allowed file extensions and any partition-specific path rules;
- upstream URL, ownership, license-field mapping, and attribution rule;
- parser-bundle hash and semantic-spec hash; and
- extraction rule version.

The Stack supplies the primary partition for all 14 languages. NVIDIA
OpenCodeInstruct supplies Python only and must continue to pass its perfect-test
filter. A future source can be added only through a new registry entry and
provenance test.

`SourceDocumentRecord` gains language, parser-bundle hash, grammar identity,
semantic-spec hash, and extraction-rule version. Stable document IDs incorporate
language and the registry-entry hash. Rejections carry the same source identity
even when parsing or licensing fails.

### Version 1 dialect boundaries

The first multilingual corpus admits conservative, unambiguous file forms:

| BCNLT language | Admitted forms | Explicitly deferred |
| --- | --- | --- |
| Python | `.py` | notebook cells and generated stubs |
| JavaScript | `.js`, `.mjs`, `.cjs` | JSX and embedded script |
| TypeScript | `.ts` | TSX and declaration-only `.d.ts` |
| C | `.c` | ambiguous `.h`, Objective-C |
| C++ | `.cc`, `.cpp`, `.cxx`, `.hh`, `.hpp`, `.hxx` | ambiguous `.h`, Objective-C++ |
| C# | `.cs` | generated designer sources |
| Java | `.java` | templated/generated sources |
| Go | `.go` | generated sources |
| Rust | `.rs` | generated expansion output |
| PHP | standalone `.php` whose semantic body is PHP | mixed HTML/PHP templates |
| Ruby | `.rb` | ERB and extensionless DSL files |
| Kotlin | `.kt` | `.kts` scripts |
| Swift | `.swift` | generated interface output |
| Shell | `.sh`, `.bash`, parsed with the Bash grammar | Zsh/Fish and embedded shell strings |

Shell rows record a `posix_sh` or `bash` dialect from extension and shebang, but
both count toward the Shell language gate. A file with an ambiguous extension or
grammar-adjacent syntax is rejected rather than guessed. Later dialect support
requires new registry and semantic-spec versions.

### Document admission

A document is admitted only when all of the following are true:

- content decodes as UTF-8 and meets language-specific extension/path rules;
- repository identity, immutable source revision, license, and attribution are
  present and allowed;
- its content hash does not collide with any protected holdout;
- the selected parser produces no `ERROR` or `MISSING` nodes;
- semantic extraction is deterministic across two clean processes;
- named-node semantic coverage is at least 85%;
- `UNKNOWN_SEMANTIC` events are at most 10% of emitted semantic events; and
- scope events are balanced and all non-null definition links resolve within the
  document index.

Semantic coverage counts named syntax nodes after excluding punctuation,
comments, purely syntactic wrappers, and nodes explicitly classified as
non-semantic in the language spec. Exclusions are versioned and visible in the
coverage report so a spec cannot inflate coverage by ignoring hard constructs.

### Curriculum

The five existing families remain:

1. semantic continuation;
2. semantic-to-code reconstruction;
3. code-to-semantic alignment;
4. structural reasoning; and
5. mixed instruction use.

Every example includes language, source-document ID, repository identity,
parser-bundle hash, semantic-spec hash, source span, task family, and tokenizer
manifest hash outside the model-visible token stream. Prompt wording may mention
the language when required by the task, encoded as ordinary base-token payload.
For version 1, every task family begins with a model-visible language field so
structurally similar records cannot ask the model to infer the output language.
This field uses existing base-token payload encoding and does not add structural
token IDs.

Deterministic task generators are generalized over the semantic contract. A
language can contribute to a family only when its source record contains the
events required for that task. Model-authored wording remains reviewable and its
factual target must match the canonical record.

All generated sequences are capped to the Runeforge training context of 4,096
tokens after packing. Oversized examples are split only on semantic-unit
boundaries; examples that cannot be split without losing their target are
rejected.

One packed training sequence contains examples from exactly one language.
Shards and batches may interleave single-language sequences, but the packer may
not concatenate different languages into one 4,096-token record. Per-language
record and token quotas therefore remain exact and directly attributable.

### Balance and split isolation

The full-corpus gate requires:

- at least 10,000 admitted packed training records for each of 14 languages;
- at least 140,000 admitted packed training records total;
- at least 50,000,000 encoded training tokens total;
- no language above 12% of admitted records or tokens;
- all five task families represented for every language;
- no task family above 40% of one language's records; and
- no language above 7.5% `UNKNOWN_SEMANTIC` events in aggregate;
- no single source document above 0.5% of the complete admitted corpus.

These are minima, not targets to meet through duplication. If a language cannot
meet them with quality data, the full multilingual run remains blocked and the
report names that language.

Split assignment occurs before curriculum expansion using a stable hash of
source registry, repository identity, and content/semantic duplicate cluster:
90% train, 5% validation, and 5% tokenizer/curriculum test. Repository aliases,
exact content duplicates, and canonical semantic near-duplicates stay in one
split. Existing Wizard/Runeforge head-to-head and named evaluation files remain
outside every corpus split.

## Validation strategy

### Parser-bundle validation

- Load all 14 grammars on Windows and WSL from the frozen bundle.
- Verify runtime ABI compatibility, bundle hashes, and resolved grammar IDs.
- Parse one minimal and one representative fixture per language.
- Prove that no network access or mutable cache lookup occurs during validation
  or corpus generation.

### Golden semantic fixtures

Each language receives fixtures for definitions, parameters, shadowing,
references, calls, member access, imports, assignment, branch, loop, return,
operators, literals, nested scopes, and major language-specific constructs. Each
fixture pins the ordered event stream, spans, symbol links, IR projection,
uncertainty metadata, and coverage calculation.

Malformed fixtures prove parse-error quarantine. Valid but unresolved features
prove explicit uncertainty. Metamorphic fixtures rename identifiers and alter
formatting/comments to prove that symbol links and semantic structure remain
stable while spans and surfaces update correctly.

### Corpus sample gate

Before scaling, stream a bounded sample of at least 100 source rows per language
from the exact pinned partitions. Report for each language:

- fetched, licensed, parsed, semantically admitted, and rejected counts;
- every rejection reason;
- parser and semantic coverage distributions;
- event, scope-depth, uncertainty, and task-family distributions;
- exact/near-duplicate rejection;
- packed sequence lengths and token totals; and
- provenance completeness.

No aggregate result can hide a failed language. The sample gate passes only when
every language has at least 25 admitted documents spanning definitions, calls,
references, control flow, and nested scopes, and no unexplained deterministic
mismatch.

### Training smoke gate

The multilingual smoke corpus uses equal record counts per language and covers
all task families. A bounded one-step and short-loss run in WSL must prove:

- every language's semantic sequences stay within token and embedding ranges;
- mixed-language batches collate and checkpoint correctly;
- held-out BCNLT loss decreases globally and does not catastrophically diverge
  for any individual language;
- raw-code and ordinary-language control probes stay within the existing
  material-regression threshold; and
- the adapter, optimizer state, tokenizer, parser manifest, corpus manifest,
  metrics, and exact base checkpoint hashes save and reload independently.

The enforceable smoke uses seed `1729`, microbatch size 1, gradient accumulation
8, and 128 optimizer steps over an equal-language/equal-family sample. Validation
uses 256 fixed examples per language. Every step must have finite loss; mean
global validation loss over the final 32-step window must be at least 2% below
the first 32-step window; and no language's final validation loss may exceed its
initial value by more than 25%. The smoke also enforces the existing two-point
raw-code and ordinary-instruction regression limit. These thresholds are a
compatibility/promotion gate, not a claim of final model quality.

### Evaluation and promotion

Promotion evaluates each language with deterministic held-out probes for event
classification, definition/reference links, scope pairing, control flow,
semantic-to-code reconstruction, and code-to-semantic alignment. It also runs a
macro average across languages so Python volume cannot dominate the result.

Structural accuracy is the unweighted macro average of six exact-scored
categories: event classification, definition/reference linking, scope pairing,
control-flow facts, code-to-semantic alignment, and semantic-to-code
reconstruction. Each category contributes one sixth of the language score and
must individually reach 80%; the language macro must reach 90%. A null or
uncertain resolution earns credit only when the frozen golden target also marks
that fact unresolved. Parseable but semantically different output does not earn
exact credit, though it is reported as a diagnostic secondary metric.

Semantic-to-code reconstruction is exact at the canonical-semantic level rather
than at the source-byte level. The emitted code must parse with the pinned target
grammar, then re-extract to the same normalized semantic projection as the
golden target: ordered events, literal and identifier values, scope topology,
definition/reference graph, AST relations, IR kinds/values, and uncertainty
categories must match. Source spans, comments, line endings, and purely
formatting surface text are excluded from that comparison. A parseable program
with a different semantic projection fails the exact item.

Every language must reach at least 90% structural accuracy on its admitted test
split under that formula. A language failing its threshold blocks promotion of the single
multilingual adapter; it is not silently omitted after training. Raw-code and
ordinary-instruction aggregate regression remains capped at two absolute
percentage points, with any new tokenizer/runtime failure treated as a hard
failure.

Claims compare the preserved Runeforge Mk2 core, the existing Python-only BCNLT
adapter if available, and the multilingual adapter under identical prompts,
decoding settings, quantization, and hardware.

### Authoritative Runeforge recipe

The multilingual run extends the established Runeforge BCNLT recipe rather than
inventing a second adapter shape: QLoRA/Unsloth when the compatibility smoke
passes, LoRA rank 16, alpha 32, dropout 0, learning rate `2e-4`, gradient
accumulation 8, maximum sequence length 4,096, and target modules `q_proj`,
`k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. The sparse
BCNLT embedding and LM-head extension remains trainable while preserved base rows
remain frozen.

The full run uses microbatch size 1, seed `1729`, AdamW, one complete admitted
training epoch, 3% linear warmup followed by cosine decay, validation every 500
optimizer steps, and content-addressed checkpoints every 500 optimizer steps and
at epoch end. The immutable full-run configuration records the resulting exact
step and token budget after the final corpus is frozen. Resume requires exact
configuration and lineage hashes.

## Execution stages

1. **Freeze parser bundle.** Add the dependency only after a capability probe
   confirms all 14 grammar IDs and produces a cross-platform manifest.
2. **Generalize contracts.** Add language/provenance envelopes, optional render,
   parser identities, and canonical uncertainty fields without changing
   tokenizer IDs.
3. **Build shared runtime.** Implement deterministic Tree-sitter traversal,
   scope tracking, symbol tables, and coverage accounting.
4. **Build language specs and goldens.** Complete one language at a time, using
   shared family helpers where safe, while applying the same parity gate.
5. **Generalize ingestion and curriculum.** Add per-language source registry,
   filters, task generation, deduplication, admission, and reports.
6. **Run the 100-row-per-language sample.** Repair semantic gaps and freeze the
   resolved adapter/spec versions.
7. **Run multilingual smoke training and evaluation.** Keep the large corpus
   launcher locked until every gate passes.
8. **Build and validate the full balanced corpus.** Produce content-addressed
   shards and reports.
9. **Run the full Runeforge adaptation.** Preserve the base checkpoint and the
   standalone adapter, then evaluate before any merge or promotion.

Stages are resumable and content-addressed. A changed parser bundle, grammar,
semantic spec, tokenizer manifest, source registry, dataset revision, or base
checkpoint invalidates downstream artifacts instead of reusing them.

The parser probe must resolve and pin each upstream grammar source revision even
when the language-pack API omits it. A platform binary hash alone is insufficient
grammar provenance. If a revision cannot be resolved from the package catalog
and its build metadata, that grammar is blocked until it can be tied to a source
revision.

After LF normalization and UTF-8 parsing, canonical fixture records must be
byte-identical between Windows and WSL. Platform parser binaries may have
different file hashes and are listed separately in one logical bundle manifest,
but they must report the same grammar source revision and emit identical
canonical semantics. Any cross-platform semantic difference blocks the bundle.

## Failure handling and recovery

- Parser errors, missing grammars, ABI mismatches, altered parser binaries,
  missing provenance, unknown licenses, holdout collisions, unbalanced scopes,
  invalid links, and deterministic mismatches are quarantined with reason codes.
- Unknown valid constructs emit `UNKNOWN_SEMANTIC` plus uncertainty and remain
  eligible only below the per-document and per-language thresholds.
- Network loss during Windows ingestion resumes from the content-addressed cache;
  WSL training never fetches source data directly.
- Interrupted corpus runs resume only when all manifest hashes match.
- Interrupted training resumes only from a checkpoint whose tokenizer, parser,
  corpus, semantic-spec, adapter configuration, and base checkpoint hashes match.
- Failure in one language leaves the other language artifacts intact, but the
  full multilingual training gate stays closed.

## Deliverables

- Frozen Windows/WSL parser bundle and manifest for the 14 grammar IDs.
- Shared Tree-sitter adapter runtime and 14 versioned semantic specifications.
- Golden semantic fixtures and coverage reports for every language.
- Multilingual source registry, admission database, rejection log, and
  content-addressed training shards.
- Per-language and aggregate corpus validation reports.
- Equal-language smoke corpus, smoke adapter, logs, and evaluation evidence.
- Guarded full-corpus and Runeforge adaptation launchers.
- Standalone multilingual adapter and, only after evaluation and explicit merge
  authorization, a derived merged Runeforge checkpoint.
- Lineage documentation from every derived artifact back to the preserved
  Runeforge core, tokenizer v1, parser bundle, semantic specs, and source rows.

## Acceptance criteria

The design is complete when a builder can plan and implement the project without
choosing its product behavior. The built system is acceptable only when:

- all 14 language adapters pass identical semantic parity gates;
- parser and semantic identities are frozen and reproducible on Windows and WSL;
- tokenizer v1 IDs and the preserved Runeforge base remain unchanged;
- the bounded multilingual source sample and equal-language training smoke pass;
- the full admitted corpus meets every per-language, global, provenance,
  diversity, duplicate, split, and token threshold;
- every language reaches the stated held-out structural threshold;
- raw-code and ordinary-language regressions remain within the fixed limit; and
- artifacts, rejection reasons, metrics, and lineage are sufficient to rerun or
  audit the result.

## Builder calls

These are implementation choices made in this design rather than requirements
the user separately specified:

- Use `tree-sitter-language-pack==1.22.1` only to obtain parser binaries, then
  freeze and hash the complete cache for offline corpus work.
- Keep language identity in record/example envelopes and ordinary payload text,
  avoiding changes to BCNLT tokenizer v1.
- Make `LanguageAdapter.render()` optional and reconstruct training targets from
  admitted source spans.
- Define first-version symbol resolution as document-local and lexical.
- Require 10,000 packed records per language, 140,000 total records, and 50
  million tokens before the full run.
- Require 85% named-node coverage, at most 10% unknown semantic events, and 90%
  held-out structural accuracy per language.
- Keep Windows as the authenticated ingestion host and WSL as the CUDA training
  host until the WARP networking condition changes and is reverified.
