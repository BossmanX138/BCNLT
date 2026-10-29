# BCNLT Adaptive Vocabulary v5 Experiment

## Intent

This experiment tests whether AdaBPE-style adaptive frequency thresholds can
shorten BCNLT semantic-code streams without replacing the Runeforge Mk2 base
tokenizer, changing existing BCNLT v1 token IDs, or weakening exact
reconstruction. It also measures whether model-prediction entropy identifies
useful code boundaries. Entropy remains a diagnostic signal and does not enter
the uncertainty/UYR/wildcard learning loop in this experiment.

The experiment exists because BCNLT currently expresses semantic structure
with dedicated tokens while encoding open-ended field payloads with the
preserved Runeforge tokenizer. Repeated payload fragments may therefore be
compressible into append-only BCNLT macro tokens. That possibility can be
tested without altering ordinary text, raw-code mode, the Runeforge base
vocabulary, or the Common Coding Core.

The v4 tokenizer, adapter, evaluation evidence, and preserved Runeforge Mk2
checkpoint are immutable inputs. The v5 name identifies a research run, not a
promoted tokenizer release.

## Goals

- Measure repeated base-token sequences inside semantic-field payloads on the
  admitted BCNLT corpus.
- Rank lossless macro-token candidates with adaptive thresholds based on
  sequence length and semantic payload class.
- Compare append-only candidate inventories of 256, 512, and 1,024 macro
  tokens against the unchanged BCNLT v1 baseline.
- Preserve every Runeforge base token ID and every BCNLT v1 structural token
  ID.
- Keep one language-neutral Common Coding Core and one shared macro inventory.
- Require evidence from multiple languages before calling a candidate a shared
  coding token.
- Measure predictive boundary entropy separately from BCNLT knowledge
  uncertainty.
- Produce enough tokenizer-only and bounded-adapter evidence to decide whether
  an adaptive vocabulary deserves a separately versioned implementation.

## Non-goals

- Replacing Runeforge's ordinary-language tokenizer.
- Changing the meaning or IDs of existing Common Coding Core concepts.
- Creating language-specific token ranges, tokenizers, adapters, or LoRAs.
- Implementing true CoBPE compositional embeddings or additional output heads.
- Implementing phonemic English tokenization.
- Feeding predictive entropy into UYR, wildcard resolution, learned-definition
  promotion, or canonical semantic records.
- Starting a full training run or merging experimental weights.
- Claiming better coding ability from compression alone.

## Anti-goals

The experiment has failed if it obtains shorter sequences by making decoding
ambiguous, crossing semantic boundaries, changing existing IDs, favoring one
language while calling the result universal, contaminating held-out data, or
using the v4 artifacts as writable output directories.

## Existing constraints

- The Runeforge Mk2 base checkpoint and vocabulary remain byte-for-byte
  unchanged.
- BCNLT v1 IDs are append-only. Macro IDs begin after the final existing v1
  structural ID.
- The hybrid tokenizer continues to route ordinary text and raw code through
  the base tokenizer. Macro substitution is permitted only inside encoded
  semantic-field payloads.
- The current registered grammar set contains fourteen languages, but the
  current pinned corpus configuration is Python-focused. A Python-only audit
  may discover preliminary candidates but cannot promote them into the shared
  v5 candidate inventory.
- All candidate discovery uses training splits only. Evaluation holdouts are
  never used for frequency counts, thresholds, ranking, or tie-breaking.
- Existing admission, provenance, near-duplicate, and split-isolation rules
  remain in force.
- Every run records the exact source revisions, corpus manifest, base-tokenizer
  identity, v1 manifest hash, semantic-spec hashes, parser-bundle hash, code
  revision, configuration, and random seed.

## Approaches considered

### 1. Measurement-only research spike

Analyze candidate savings and predictive entropy without creating new token
IDs. This has the lowest risk and provides a reliable upper bound, but it
cannot measure tokenizer latency, learned macro embeddings, generation
accuracy, or model behavior with shorter sequences.

### 2. Append-only macro-token experiment

Discover lossless payload macros, append experimental IDs after the v1
inventory, compare fixed vocabulary budgets, and train only a bounded adapter
after tokenizer-only gates pass. This preserves the current model and token
lineage while directly measuring the expected benefit. This is the selected
approach.

### 3. Full compositional model

Add CoBPE-style factorized embeddings and multiple output heads, then pretrain
a matched decoder from scratch. This is the only approach that tests multiple
components inside one contextual position, but it is a different model
architecture and cannot provide a controlled improvement to the existing
Runeforge Mk2 checkpoint. It remains a future research branch.

## Terminology

**Knowledge uncertainty** is BCNLT's current estimate that a parsed construct's
meaning is unknown or weakly supported. It belongs to semantic preprocessing
and may activate UYR and wildcard learning.

**Predictive boundary entropy** is the frozen model's next-token uncertainty at
a particular position. It may correlate with local structural boundaries, but
it is not evidence that BCNLT understands or does not understand a construct.

**Macro token** is one experimental BCNLT ID with a frozen, lossless expansion
to a sequence of two or more Runeforge base-token IDs. A macro token is an
encoding optimization, not a new Common Coding Core concept.

**Shared candidate** is a macro candidate supported by at least three canonical
language IDs and at least three independent source repositories after
deduplication. A candidate that does not meet that rule remains an audit-only
language-local candidate and cannot receive an ID in this experiment.

## Architecture

The existing semantic path remains unchanged through canonical record
construction:

`source -> frozen grammar -> shared walker -> Common Coding Core -> uncertainty/UYR/wildcard -> canonical records`

The experimental encoding path is:

`canonical record -> v1 structural framing -> base-tokenized field payload -> deterministic macro substitution -> v5 experimental IDs`

Macro substitution never runs in text mode or raw-code mode. It never changes
the canonical record and never crosses a field boundary, record boundary, or
stream boundary.

The decoder expands each macro ID to its recorded base-token sequence before
the existing v1 field decoder runs. This makes decoding independent of corpus
frequency, model state, external memory, or language detection.

## Phase 0: freeze lineage

Before analysis, the runner verifies and records:

- the Runeforge Mk2 base hashes in `configs/runeforge_mk2_bcnlt_v1.json`;
- the complete BCNLT v1 tokenizer manifest and structural inventory hash;
- the v4 corpus, adapter, and corrected promotion-gate hashes;
- the corpus source revisions and admitted split manifest;
- the parser bundle, grammar identities, Common Core spec, and code revision;
- the output root, which must be a new directory and must not be any v4 path.

Any mismatch stops the experiment before candidate discovery.

## Multilingual corpus prerequisite

Creating or approving new external dataset sources is outside this v5
experiment. The runner accepts one explicit
`bcnlt-adaptive-vocab-corpus-v1` configuration whose entries contain approved
source names, immutable revisions, licenses, repository-identity rules, and
expected hashes. It never searches for or silently adds another dataset.

The current `configs/bcnlt_corpus_sources_v1.json` input may be audited as a
preliminary snapshot, but it is not a qualifying shared-vocabulary corpus. A
qualifying configuration must admit all fourteen canonical language IDs in
`bcnlt/languages.py`. After deduplication, each language must contribute at
least 100 documents, three independent repositories, and 100,000 semantic
payload base tokens to the training split. The complete admitted-corpus
manifest hash is the corpus snapshot identity used by every later phase.

Preparing that approved multilingual source configuration is a separate data
lineage task. Until it exists and passes admission, the present experiment may
produce only the preliminary audit and the explicit diversity failure report.

## Phase 1: corpus audit

The audit reads admitted training records and records counts by canonical
language, repository, document, semantic event, payload field, uncertainty
band, and base-token length. It writes a report even when diversity is
insufficient.

The audit has two outcomes:

1. If any registered language misses its document, repository, or payload-token
   floor, it writes
   `insufficient_shared_candidate_diversity` and stops before constructing a
   macro inventory.
2. Otherwise it proceeds using normalized language and payload-class rates.
   No random subsampling drives candidate ranking. The full raw counts remain
   in the report, while normalization prevents a large language from receiving
   more ranking weight merely because it supplied more tokens.

This rule allows the current Python-focused configuration to be inspected now
without presenting Python frequency as a universal coding vocabulary.

## Phase 2: candidate discovery

Candidate extraction operates on the existing base-token IDs inside one
semantic-field payload. Eligible sequences contain 2 through 16 contiguous
base tokens and decode deterministically to the identical payload bytes.

Candidate extraction first caps each candidate's count at eight occurrences
per deduplicated document. Candidates are excluded when they:

- cross field, record, task, or stream boundaries;
- contain an existing BCNLT structural token;
- depend on a source path, repository name, document identity, random seed, or
  split label;
- occur only through exact or near-duplicate documents;
- contain a whole high-entropy literal, secret-like value, or unique symbol
  whose usefulness comes from memorizing one project;
- fail UTF-8 and base-token encode/decode round trips;
- are supported by fewer than three languages, three repositories, or five
  deduplicated documents;
- receive more than 50 percent of their capped occurrences from one repository
  or more than 10 percent from one document;
- decode to a configured credential prefix, PEM header, JSON Web Token shape,
  URI containing user information, or hexadecimal/base64-like run of at least
  24 characters;
- exactly cover a literal-event surface of at least 16 characters whose
  character Shannon entropy is at least 3.5 bits per character.

The credential prefixes, regular expressions, and entropy implementation are
versioned configuration data and are included verbatim in the experiment
identity. They begin with the repository's approved secret-detection rules; if
no approved rules exist, implementation planning must add and test the fixed
rules before candidate discovery rather than selecting a third-party service
at runtime.

The AdaBPE-inspired score is exact and deterministic. For candidate `c` of
base-token length `L`, language `l`, and payload class `p`:

`rate(c,l,p) = 1,000,000 * capped_occurrences(c,l,p) / eligible_base_tokens(l,p)`

For each payload class, take the third-highest language rate, so a zero or weak
third language lowers the result. The candidate's score is:

`score(c) = (L - 1) * max_p(third_highest_l(rate(c,l,p)))`

The factor `L - 1` is the number of tokens saved per replacement. A candidate
is eligible only when `score(c) >= 25`, representing at least 25 balanced saved
tokens per million eligible payload tokens in its strongest qualifying payload
class. Ranking is by descending score, then descending number of supporting
languages, repositories, and documents, then lexicographically by the
canonical base-token ID expansion. Language identity never allocates a
separate token budget.

All raw counts, capped counts, denominators, class scores, exclusions, and
tie-break inputs are serialized. Changing the formula, threshold, cap, or
exclusion rules creates a new experiment identity.

## Phase 3: fixed-budget inventories

The experiment attempts to build candidate inventories containing 256, 512,
and 1,024 macros. It builds a budget only when at least that many candidates
pass every eligibility rule. Fewer than 256 eligible candidates stops the
experiment with `insufficient_eligible_candidates`. Between 256 and 511 builds
only the 256 inventory; between 512 and 1,023 builds the 256 and 512
inventories. The runner never fills a budget with below-threshold candidates.

Existing v1 IDs remain identical. New IDs are contiguous and ordered by the
candidate ranking defined above.

Each inventory stores:

- macro ID and stable display name;
- exact base-token ID expansion;
- decoded UTF-8 when valid for display;
- balanced and unbalanced occurrence counts;
- supporting languages, repositories, and document fingerprints;
- payload classes and semantic events;
- estimated training-only token savings;
- candidate-discovery configuration and source-manifest hashes.

Encoding uses deterministic longest-match substitution. Equal-length matches
use the lower macro ID. Decoding is a direct macro-to-base-ID expansion. The
same input, artifacts, and configuration must produce byte-identical
inventories on Windows and WSL.

## Phase 4: tokenizer-only evaluation

The unchanged v1 tokenizer is the baseline. Each candidate inventory is
evaluated on the same frozen training sample, validation split, and untouched
evaluation holdout.

Required measurements are:

- total and per-language semantic-stream token counts;
- bytes per token and base tokens replaced per macro;
- macro coverage by field, event, language, and repository;
- exact canonical-record encode/decode round trips;
- exact Unicode, whitespace, newline, and token-like-payload round trips;
- deterministic output and manifest identity across Windows and WSL;
- p50 and p95 encode/decode latency after warm-up;
- peak host memory during inventory load and encoding;
- unchanged text-mode and raw-code token sequences;
- unchanged v1 token IDs and base-tokenizer hashes;
- malformed, truncated, unknown, and out-of-range macro rejection.

An inventory passes the tokenizer-only gate only when:

- every round trip is exact with zero failures;
- text mode and raw-code mode are byte-for-byte identical to v1;
- every existing v1 ID and base ID is unchanged;
- the held-out semantic stream is at least 10 percent shorter in aggregate;
- no supported language has longer semantic streams than v1;
- p95 encode and decode latency are each no more than 10 percent slower than
  v1 under the recorded benchmark protocol;
- no candidate lacks the required language and repository support;
- the Windows and WSL artifacts are byte-identical.

If multiple inventories pass, choose the smallest inventory whose held-out
compression is within one percentage point of the best passing inventory. If
none pass, the adaptive-vocabulary experiment ends without model training.

## Phase 5: predictive-boundary entropy diagnostic

The diagnostic uses two frozen models: the v4 adapter on v1 semantic streams is
the primary measurement, and the preserved Runeforge Mk2 base on raw code is a
control. The untrained extension is not used for an authoritative entropy
claim. Both run on frozen held-out inputs and record next-token entropy with
source, canonical field, AST boundary, and macro-match alignment. Results are
written to separate model-specific records in one research report.

The diagnostic must distinguish:

- predictive boundary entropy;
- BCNLT knowledge uncertainty;
- parser or grammar unknowns;
- learned-definition state.

It may calculate correlations, precision/recall for registered AST boundaries,
and entropy distributions at macro starts and ends. It must not modify
canonical records, the learned-definition store, wildcard evidence, corpus
admission, macro ranking, or tokenizer IDs. Any future proposal to use entropy
as a learning signal requires a separate causal ablation and design approval.

## Phase 6: bounded adapter experiment

Model training is eligible only after one inventory passes the tokenizer-only
gate. Both training arms start from separate byte-identical copies of the
frozen v4 adapter state. The v1 continuation control retains v1 tokenization.
The v5 arm appends deterministic sparse macro embedding and LM-head rows and
uses the selected macro inventory. Base weights remain frozen; each arm may
update its copied v1 extension rows and LoRA weights under the same audited
method. Outputs go to separate directories beneath the new v5 experiment root.

Both arms use the identical admitted multilingual training examples, order,
curriculum targets, optimizer settings, effective batch size, and seed. The v5
arm changes only tokenization and its new macro rows. The deterministic
training subset is selected from the admitted training split to cover each
selected macro at least eight times when possible, then balanced by language
and curriculum family, with a hard ceiling of 10,000 examples. A selected
macro with fewer than eight available training occurrences is removed and the
inventory gate is rerun. Training runs for one epoch or 512 optimizer steps,
whichever comes first, with no arm-specific early stopping. Validation is
measured every 64 optimizer steps and at the final shared step.

The comparison matrix contains the frozen v4 adapter, the untrained v5 macro
extension attached to v4, the v1 continuation control, and the trained v5 arm.
It records training loss, held-out negative log likelihood, generation
outputs, latency, peak VRAM, and token throughput. Direct loss per generated
token is reported but is not used for cross-tokenizer gating. The authoritative
cross-tokenizer loss is total negative log likelihood in bits divided by the
UTF-8 byte length of the identical canonical held-out targets. The identical
held-out target bytes and case order are hashed in the report.

The model gate requires:

- all existing tokenizer, structural, semantic-to-code, raw-code, and ordinary
  instruction promotion categories to pass their existing thresholds;
- exact Unicode and verbatim reconstruction, including the final newline;
- no regression in held-out bits per canonical byte against both the frozen v4
  adapter and the matched v1 continuation control at the same shared step;
- no regression in parseability or AST equivalence;
- at least 10 percent shorter semantic inputs on the frozen heldout;
- measured end-to-end throughput and VRAM reported without substituting token
  compression for model-quality evidence;
- finite losses, reproducible lineage, and a reloadable standalone adapter.

Passing this gate permits a promotion proposal. It does not merge weights,
replace v1, launch full training, or publish an artifact automatically.

## Artifacts and output boundaries

The default run root is:

`D:\BCNLT-runs\runeforge-mk2-bcnlt-v5-adaptive-vocab`

It contains:

- `lineage.json` with all immutable input identities;
- `corpus-audit.json` and per-language counts;
- `candidates.jsonl` and `candidate-rejections.jsonl`;
- one immutable inventory directory per vocabulary budget;
- tokenizer-only reports and platform reproducibility evidence;
- `boundary-entropy.jsonl` and a summary report;
- a selected-inventory decision or an explicit no-pass result;
- bounded adapter artifacts and evaluation only when the tokenizer gate passes;
- a final promotion proposal that defaults to `promoted: false`.

No experiment stage writes into the preserved Runeforge checkpoint, v4 run
directory, repository fixtures, or v1 tokenizer inventory.

## Failure handling

Every stage writes its manifest and failure code atomically. A failed or
interrupted stage preserves completed reports but cannot be treated as input to
a later stage unless all declared hashes verify. Resume requires an exact
match of code, configuration, corpus, tokenizer, model, and prior-stage
artifacts.

Required failure codes include:

- `lineage_mismatch`;
- `output_overlaps_preserved_artifact`;
- `insufficient_shared_candidate_diversity`;
- `candidate_roundtrip_failure`;
- `existing_token_id_changed`;
- `text_or_raw_code_regression`;
- `compression_below_threshold`;
- `tokenizer_latency_above_threshold`;
- `cross_platform_artifact_mismatch`;
- `no_inventory_passed`;
- the existing per-category model promotion failures.

## Verification

Implementation tests must prove:

- v1 base and structural IDs remain unchanged for every candidate budget;
- macro encoding is deterministic and cannot cross semantic boundaries;
- macro decoding expands to the exact original base-token sequence;
- arbitrary Unicode and token-like payloads round-trip exactly;
- text and raw-code modes do not invoke macro substitution;
- candidates derived from holdouts, one language, or too few repositories are
  rejected;
- duplicated repositories cannot manufacture cross-repository support;
- language balancing affects ranking but not provenance counts;
- all candidate decisions and tie-breaks reproduce byte-for-byte;
- predictive entropy cannot mutate learning, admission, or tokenizer state;
- later phases reject incomplete or mismatched earlier artifacts;
- the runner refuses every preserved v4 and base-model path as an output root;
- a no-pass result exits cleanly without model training.

## Decisions left to implementation planning

The implementation plan may choose module and command names, internal data
classes, streaming/chunking strategy, benchmark repetition counts, and report
visualization. It may reduce memory use without changing discovery results.
It may not alter the candidate budgets, shared-candidate evidence rule,
lineage boundary, gates, or separation between knowledge uncertainty and
predictive entropy without revising this design.
