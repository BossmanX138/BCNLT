# BCNLT Hybrid Tokenizer Design

## Intent

BCNLT is intended to become a real coding tokenizer rather than remain only a
semantic context preprocessor. The first production target is Runeforge Mk2 7B:
it must consume stable BCNLT token IDs directly, learn their embeddings, and use
the semantic stream for code understanding and generation while the preserved
Mk2 checkpoint remains available unchanged.

The current BCNLT v0.1 prototype parses Python into `ContextualUnit` records and
can serialize those records as JSON or model context. It does not yet define a
versioned vocabulary, deterministic token IDs, a decoding contract, or a model
embedding integration. This project supplies those missing parts and generates
an admitted training corpus large enough for a staged Runeforge adaptation.

## Goals

- Define BCNLT vocabulary version 1 with stable, append-only token IDs.
- Encode Python as a canonical sequence of BCNLT structural tokens and payload
  subwords.
- Preserve every existing Runeforge tokenizer ID and embedding row.
- Add trainable embeddings for new BCNLT tokens without modifying the preserved
  Mk2 source checkpoint.
- Retain ordinary text and raw-code modes so Runeforge can still converse and
  emit normal source code.
- Generate at least 100,000 accepted curriculum records with provenance,
  document-level split isolation, duplicate rejection, and immutable evaluation
  holdouts.
- Produce a small smoke adapter before authorizing the full continued-training
  run.
- Measure semantic-token comprehension, raw-code regression, and deterministic
  tokenizer behavior before promotion.

## Non-goals

- Training a new 7B model from random initialization.
- Reassigning, deleting, or changing any existing Runeforge token ID.
- Claiming that AST-derived semantics can reproduce the original source bytes.
- Treating repeated paraphrases of the same semantic unit as independent data.
- Adding languages other than Python in vocabulary version 1.
- Replacing the preserved Runeforge Mk2 checkpoint in place.

## Anti-goals

- A text-only instruction dataset that merely teaches the spelling of BCNLT
  records while the model still sees them entirely as ordinary subwords.
- An unversioned vocabulary whose IDs depend on corpus order.
- A corpus split at the generated-record level that leaks the same source file
  into training and evaluation.
- A nominal tokenizer that cannot save, reload, and reproduce identical IDs.
- Scaling training before a bounded smoke run proves that the new tokens learn
  without unacceptable language or coding regression.

## Chosen approach

The selected approach is a hybrid expansion of the Runeforge tokenizer. It is a
real tokenizer because BCNLT structural concepts receive dedicated token IDs and
enter the model through dedicated embedding rows. It is hybrid because arbitrary
payloads such as identifiers, strings, numbers, paths, and natural-language text
continue to use the original tokenizer.

The alternatives considered were:

1. Keep BCNLT as an external semantic-context layer. This is already working,
   but it does not make BCNLT the model's tokenizer.
2. Train the model to read serialized BCNLT using only its old subword tokens.
   This is inexpensive, but BCNLT concepts still have no stable token IDs or
   dedicated embeddings.
3. Replace the whole vocabulary and train a new model from scratch. This is a
   pure tokenizer replacement, but it discards the main value of the Runeforge
   checkpoint and requires pretraining resources beyond this project.

The hybrid expansion preserves the useful part of option 3 while keeping the
existing model viable.

## Token contract

### Vocabulary identity

The tokenizer artifact contains:

- `schema_version`: `bcnlt-tokenizer-v1`.
- The exact source tokenizer identity and hashes.
- `base_vocab_size`, which is immutable for the artifact.
- An ordered list of added BCNLT tokens.
- An explicit token-to-ID map whose first added ID is `base_vocab_size`.
- A canonicalization version for semantic-unit serialization.
- A SHA-256 digest of the canonical manifest.

All original token IDs remain unchanged. New tokens are append-only. Loading an
artifact with a different base tokenizer hash, duplicate ID, missing required
token, or changed manifest digest fails closed.

### Structural vocabulary

Version 1 reserves dedicated tokens for:

- Stream and record boundaries.
- Modes: ordinary text, raw code, and BCNLT semantic code.
- Every current BCNLT event: `DEFINE`, `REFERENCE`, `ASSIGN`, `CALL`, `RETURN`,
  `BRANCH`, `LOOP`, `MEMBER_ACCESS`, `IMPORT`, `OPERATOR`, `SCOPE_ENTER`,
  `SCOPE_EXIT`, `LITERAL`, and `UNKNOWN_SEMANTIC`.
- Field boundaries for surface, scope, symbol identity, definition link, span,
  AST relation, IR kind, confidence band, uncertainty band, and history.
- Null, boolean, list, and record separators.
- Training-task boundaries used by the curriculum.

Exact spellings and numeric IDs live only in the canonical vocabulary manifest;
code imports them instead of duplicating them.

### Payload encoding

Field values use the preserved base tokenizer between typed BCNLT field
delimiters. This handles arbitrary identifiers and literals without building an
unbounded vocabulary. Integers used for source spans are encoded as base-token
payloads in v1. Confidence and uncertainty use a small fixed set of categorical
bands so floating-point formatting cannot destabilize the sequence.

### Canonicalization

The tokenized object is a closed `SemanticTokenRecord`, not the open-ended
`ContextualUnit` object. Version 1 contains `event`, `surface`, `scope`, `span`,
`symbol_id`, `defines`, `ast_relation`, a recursively typed IR projection,
`recent_history`, `confidence_band`, and `uncertainty_band`. The IR projection
contains only `kind`, `name`, scalar/list/dictionary `value`, ordered `children`,
and sorted scalar/list/dictionary IR metadata. Contextual-unit metadata is limited
to the versioned allow-list `dynamic_call`, `dynamic_binding`, and `unknown_node`.
Adding another field or metadata key requires a new canonicalization version.

A `SemanticTokenRecord` is serialized in one fixed field order. Optional absent
values emit the BCNLT null token. Dictionary keys are sorted, line endings are
normalized to LF, Unicode payloads are normalized to NFC, and all non-allow-listed
runtime metadata is excluded. Recent history is capped at the existing eight
events.

Two round trips are required:

1. Canonical semantic record -> token IDs -> canonical semantic record must be
   exact.
2. Ordinary text/raw code -> base token IDs -> decoded text must retain the base
   tokenizer's existing behavior.

Python source -> semantic tokens -> original Python bytes is not promised because
the current AST representation intentionally omits comments and formatting.

## Runtime interfaces

`BCNLTTokenizer` wraps the preserved base tokenizer and exposes explicit modes:

- `encode_text(text)` for ordinary prompts and responses.
- `encode_raw_code(source)` for byte-equivalent behavior to the base tokenizer.
- `encode_records(records)` for complete `SemanticTokenRecord` objects.
- `records_from_units(units, index)` to score contextual units and produce
  complete records while the required project index is still available.
- `encode_source(source, path)` for Python parsing followed by semantic encoding.
- `decode(ids)` for canonical BCNLT serialization or ordinary base-token text.
- `save_pretrained(path)` and `from_pretrained(path)` for pinned artifacts.

Callers choose the mode explicitly. The tokenizer will not guess that arbitrary
text is Python. A training collator constructs mixed sequences so a causal model
can receive BCNLT semantic tokens and generate normal code or natural language.
Confidence and uncertainty are computed before encoding; they are never silently
recomputed by `encode_records`, and missing bands are a validation error.

## Python semantic coverage gate

Vocabulary v1 will not freeze the current adapter unchanged. Before corpus
identity is established, the Python adapter must explicitly model common scope
and control-flow constructs that the prototype presently traverses only
partially: classes, async functions, lambdas, `with`/`async with`, `try`/`except`
/`finally`, `raise`, comprehensions, generators, `yield`, `await`, and pattern
matching. Unsupported constructs must emit `UNKNOWN_SEMANTIC` rather than
silently appearing complete. The expanded adapter behavior is pinned by the
canonicalization version; later semantic changes require a new version and
corpus manifest.

## Model integration

The preserved Runeforge Mk2 checkpoint is read-only. A derived experimental
checkpoint adds an embedding extension for BCNLT IDs while retaining the frozen
base embedding table. New rows are initialized deterministically from the mean
of base-token embeddings for each token's plain-language definition, with a
recorded seed and initialization manifest.

The initial smoke run trains:

- the new BCNLT embedding rows;
- a small LoRA adapter on attention and MLP projections;
- an output extension only for curriculum tasks that require the model to emit
  BCNLT structural tokens.

The implementation must avoid making the complete base embedding and output
tables trainable, because that is unnecessarily expensive on the RTX 3070. A
merge step may later materialize the added rows into a derived full checkpoint,
but the original core and the standalone adapter remain preserved.

## Training corpus

### Canonical source records

The corpus builder ingests explicitly approved Python roots. Each source document
records its absolute source path, repository-relative path, content hash, license
or ownership designation, parser version, BCNLT version, tokenizer manifest hash,
and split. Files with unknown provenance are rejected rather than silently used.

The existing Wizard-versus-Runeforge head-to-head files remain immutable
evaluation holdouts and cannot enter training or validation through another root
or duplicate path.

### Curriculum families

The admitted corpus contains five complementary families:

1. **Semantic continuation:** predict later BCNLT events from earlier events.
2. **Semantic-to-code reconstruction:** consume BCNLT units and emit normalized
   raw Python for a bounded function or block.
3. **Code-to-semantic alignment:** consume raw Python and emit its canonical
   BCNLT sequence.
4. **Structural reasoning:** answer deterministic questions about definitions,
   references, scopes, calls, branches, loops, and source spans.
5. **Mixed instruction use:** receive natural-language instructions plus BCNLT
   code context and emit explanations, edits, or raw code.

Families 1-4 are generated deterministically from parsed structure. Family 5 may
use reviewed model-generated wording, but its factual target must be derived from
the canonical record and pass semantic checks.

### Size and composition

The full-corpus gate requires at least 100,000 accepted records and at least
50,000,000 encoded training tokens. No single source document may contribute more
than 0.5% of accepted records. No template may contribute more than 10% of a task
family. Exact duplicates, canonical semantic duplicates, and near-duplicate
instruction/answer pairs are removed.

Record count alone is not sufficient. Admission also requires diversity across
event types, scope depths, function sizes, control-flow structures, identifier
shapes, literal types, and uncertainty bands. Rejected records remain in an audit
log with stable reason codes.

### Splits

Splits are assigned before example expansion using a stable hash of repository
identity and source-document hash:

- 90% training candidates.
- 5% validation.
- 5% tokenizer/curriculum test.
- Existing named evaluation holdouts remain separate from all three.

All examples derived from a file, copied file, or canonical semantic duplicate
must remain in one split. Generated transformations inherit the source split.

## Validation and promotion

Tokenizer validation must prove:

- stable IDs across process restarts and save/reload;
- exact canonical semantic round trips;
- unchanged base-token IDs and ordinary decode behavior;
- deterministic handling of Unicode, empty input, malformed manifests, unknown
  IDs, deep scopes, missing optional fields, and `UNKNOWN_SEMANTIC` units;
- no evaluation-holdout hash in admitted training data.

Corpus validation must report counts, encoded-token totals, task distribution,
event coverage, split isolation, provenance coverage, duplicate rejection,
uncertainty distribution, and every rejection reason.

The smoke training gate uses a bounded subset and must demonstrate all of the
following before a full run is authorized:

- the model accepts new BCNLT IDs without an embedding or output-range error;
- held-out BCNLT loss improves over the untrained extension;
- deterministic semantic probes improve or remain perfect;
- raw-code and ordinary-language control evaluations do not show a material
  regression under the thresholds fixed in the training-run manifest;
- the adapter, tokenizer, logs, metrics, and exact source checkpoint hashes are
  saved independently.

Promotion requires a full evaluation against preserved Runeforge Mk2 using the
same prompts, decoding settings, and runtime quantization. Improvements are
reported only for measurements actually run.

The builder will create and freeze the authoritative v1 evaluation suite before
smoke training. It extends the existing definition-line holdout with deterministic
tests for scope resolution, call/definition links, control flow, semantic-to-code
reconstruction, raw-code completion, and ordinary instruction following. Both
the preserved Mk2 checkpoint and the untrained embedding extension are baselined
with identical settings. A material regression is more than two absolute
percentage points on either the raw-code or ordinary-instruction aggregate, or
any newly introduced tokenizer/runtime failure. BCNLT structural accuracy must
reach at least 90% on the tokenizer/curriculum test split before promotion.

## Failure handling

- Parsing failures, unknown provenance, malformed Unicode, unsupported tokenizer
  manifests, and holdout collisions are quarantined with reason codes.
- `UNKNOWN_SEMANTIC` remains representable but is sampled separately and may not
  dominate training.
- Corpus generation is restartable and content-addressed; rerunning unchanged
  inputs produces the same record IDs and split assignments.
- Interrupted training resumes only from a checkpoint whose tokenizer manifest,
  corpus manifest, base checkpoint, and adapter configuration hashes match.

## Deliverables

- A versioned BCNLT vocabulary manifest and tokenizer artifact.
- Tokenizer implementation, CLI inspection commands, and tests.
- A content-addressed canonical corpus database plus exported training shards.
- Corpus validation and rejection reports.
- Runeforge embedding-extension and training integration.
- Smoke adapter and smoke-evaluation evidence.
- A separate full-run launcher that cannot start unless all admission gates pass.
- Documentation describing the exact lineage from preserved Mk2 to every derived
  artifact.

## Builder calls

The following details are implementation calls rather than user-specified
requirements:

- Use the existing Python package and standard-library-first style.
- Store canonical manifests as sorted UTF-8 JSON with LF line endings.
- Use SQLite for corpus indexing and JSONL shards for training interchange.
- Begin with explicit Python-only mode and extend the manifest in later schema
  versions rather than reserving speculative language tokens now.
- Treat the 100,000-record and 50,000,000-token thresholds as admission minima,
  not reasons to duplicate or lower-quality data.
