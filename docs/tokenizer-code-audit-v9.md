# BossCrafts native tokenizer code audit — V9

Date: 2026-11-12

Scope: `bosscrafts_model/tokenizer.py`, its owned `tokenizers` 0.23.2 runtime,
the frozen 65,536-token bundle, and the tokenizer-facing tests. V8 remains
unchanged because it is the active training runtime; all repairs are isolated
in V9.

## Verified architecture

- The model binding is an exact, frozen 65,536-ID byte-level BPE vocabulary.
- IDs are unique and contiguous from 0 through 65,535.
- IDs 0 through 76 are reserved; the 73 BCNLT structural tokens remain atomic
  at IDs 4 through 76.
- Input and output embeddings are tied by the model configuration.
- The tokenizer runtime is project-contained and pinned to `tokenizers` 0.23.2.
- The tokenizer is a lexical codec. `semantic_record_parser_integrated` is
  deliberately false; BCNLT parsing and semantic construction occur in the
  source/data pipeline rather than inside byte-BPE encoding.

## Defects found and repaired in V9

### 1. Public module namespace contamination — repaired

The private owned-runtime import restored `sys.modules["tokenizers"]`, but
left public `tokenizers.decoders`, `tokenizers.models`,
`tokenizers.normalizers`, `tokenizers.pre_tokenizers`, and
`tokenizers.processors` entries behind. A later third-party tokenizer import
could therefore reuse modules from the BossCrafts-owned runtime.

V9 snapshots the complete public `tokenizers` namespace, temporarily removes
it so the owned package cannot consume caller modules, loads the owned runtime,
removes every public module created by that transaction, and restores the
caller's exact namespace, including a deliberate `None` sentinel.

### 2. Runtime provenance was not integrity checked — repaired

Executable runtime files and the native binary were hashed, but
`PROVENANCE.json` was intentionally excluded from the tree hash and was not
separately authenticated. V9 pins and checks its SHA-256 before accepting the
owned runtime.

### 3. Repeated bundle hashing and scalar structural validation — optimized

V9 reuses the already-computed tokenizer file digest rather than hashing
`tokenizer.json` twice, and validates all structural literals with one native
batch encode call. This is a small startup-only optimization; measured bundle
load remains about 0.14 seconds and is not a runtime bottleneck.

### 4. Public return type was implicit — repaired

`bound_native_65k_config` now declares its `ModelConfig` return type. The
tokenizer module passes strict isolated static type checking.

### 5. Autoregressive inference recomputed the full prefix — repaired in V9

The model's greedy generator reran every transformer block across the complete
prompt after each emitted token and materialized prompt-length by 65K logits.
V9 adds a state-dictionary-compatible per-layer key/value cache, applies rotary
positions at the correct cache offset, and projects only the newest hidden
position. A deterministic batched test proves cached greedy output is exactly
equal to the original uncached algorithm on the reference architecture.

### 6. Evaluation projected non-target prompt positions — repaired in V9

Masked evaluation previously produced 65,536-way logits for every prompt and
target position, then discarded prompt losses. V9 preserves the identical
transformer hidden-state computation but projects only positions that predict
target tokens. A direct regression test proves those logits match the
corresponding slice of the full projection.

## Executed evidence

- Complete V9 regression after all repairs and optimizations: 229 tests and
  120 subtests passed in 77.85 seconds.
- Focused tokenizer suite: 20 tests passed.
- Runtime isolation tests include clean import, an existing public root module,
  an existing public submodule tree, and a `None` sentinel.
- Runtime integrity tests reject wrapper drift and provenance drift.
- Lint: clean for the changed tokenizer and tests.
- Static typing: strict, clean for `bosscrafts_model/tokenizer.py` with imported
  modules skipped so unrelated legacy annotations do not obscure this result.
- Dynamic payload check: 507 deterministic multilingual/Unicode cases round
  tripped exactly and emitted no reserved ID.

## Remaining optimization opportunities

These are not correctness blockers and should not be applied to the active V8
training runtime.

1. Cache a verified binding within a process. This would save roughly 0.14
   seconds on repeated loads, but it weakens repeated on-disk tamper checks and
   has little value because production processes normally load once.
2. Always decode-check ordinary `encode_payload` calls. The frozen tokenizer
   has no normalizer and the executed Unicode checks are lossless, so this
   would add cost without changing current behavior; it remains a possible
   defense-in-depth option.

## Promotion conclusion

The V9 tokenizer binding is suitable for post-training evaluation after the
complete V9 regression suite passes. V8 remains the authoritative training
runtime and checkpoint producer. No claim is made that the model is complete
until training, frozen validation/test evaluation, immutable bundle export,
and bundle inference smoke all finish successfully.
