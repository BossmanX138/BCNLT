# BCNLT Tokenizer v1 Contract

`bcnlt-tokenizer-v1` is a hybrid tokenizer contract. It retains every token and
ID from a pinned Hugging Face base tokenizer and appends a fixed inventory of
BCNLT structural tokens. The artifact records the base tokenizer file hashes,
base vocabulary hash, structural inventory hash, and manifest hash so a loader
can reject drift before encoding data.

This contract defines tokenizer behavior and training-data representation. It
does not contain model weights or modify a model embedding matrix. A model that
uses the appended IDs must have its embeddings resized and trained in a later,
separately verified adaptation stage.

## ID and token classes

Base IDs occupy `[0, base_vocab_size)`. Their token strings and IDs must be
contiguous, collision-free, and unchanged from the pinned base tokenizer.
Structural IDs begin at `base_vocab_size` and follow the checked-in inventory
order without gaps.

The structural vocabulary contains:

- stream boundaries: `stream:begin` and `stream:end`
- mode markers: `mode:text`, `mode:raw_code`, and `mode:semantic_code`
- record boundaries: `record:begin` and `record:end`
- one event marker for every supported semantic event
- begin and end markers for every canonical field
- the reserved values `null`, `low`, `medium`, and `high`

Payload text is always encoded by the base tokenizer with
`add_special_tokens=False`. Structural tokens cannot appear inside a base-token
payload because their IDs are outside the base range.

## Modes

### `text`

The stream is `stream:begin`, `mode:text`, base-token payload, `stream:end`.
Decoding returns the base tokenizer's decoded text.

### `raw_code`

The stream has the same shape with `mode:raw_code`. It preserves source as a
plain base-token payload and does not add semantic records.

### `semantic_code`

Python source is parsed into canonical semantic records. Each record is framed
by record boundaries, begins with its event token, and then contains every field
in this fixed order:

1. `surface`
2. `scope`
3. `symbol_id`
4. `defines`
5. `span`
6. `ast_relation`
7. `ir`
8. `recent_history`
9. `metadata`
10. `confidence`
11. `uncertainty`

The decoder requires this exact order and rejects missing, repeated, truncated,
or misplaced structural tokens. Record encoding and decoding is exact: canonical
semantic records round-trip byte-for-byte through the canonical JSONL form.

Semantic mode is not a source-code formatter or archive. Python source does not
round-trip through semantic records because comments, whitespace, quote style,
and other surface choices are not all part of the semantic record contract. Use
`raw_code` when exact source reconstruction is required.

## Canonical values and score bands

Strings are normalized to Unicode NFC and line endings to LF. JSON values use
sorted object keys and compact separators. Only allowed contextual metadata is
admitted; recent history is capped at the last eight entries.

Confidence bands are:

- `low`: value below `0.40`
- `medium`: value from `0.40` up to but excluding `0.72`
- `high`: value at least `0.72`

Uncertainty bands are:

- `low`: value at most `0.25`
- `medium`: value above `0.25` and at most `0.60`
- `high`: value above `0.60`

Using bands keeps the persisted token representation stable when floating-point
scorers change by insignificant amounts.

## Artifact files

A saved tokenizer artifact contains exactly:

- `bcnlt_tokenizer_manifest.json`: canonical manifest with base identity,
  vocabulary hashes, appended ID assignments, and its own digest
- `bcnlt-tokenizer-v1.tokens.json`: exact structural-token inventory used by the
  manifest

The artifact contains no `.bin` or `.safetensors` model weights. The base
tokenizer remains a separate pinned input. Its recognized tokenizer files are
hashed into the manifest identity, including `tokenizer.json`, tokenizer config,
special-token maps, BPE vocabulary and merges, or SentencePiece model files when
present.

Build an artifact from a local base tokenizer directory:

```powershell
python -m bcnlt.tokenizer_cli build `
  --base path\to\base-tokenizer `
  --output path\to\bcnlt-artifact `
  --name organization/model `
  --revision pinned-revision
```

The output directory must be absent or empty. Building twice from byte-identical
base files and the same name and revision produces the same canonical manifest
digest and the same appended IDs.

## Inspection, encoding, and verification

Inspect public manifest facts without loading the base:

```powershell
python -m bcnlt.tokenizer_cli inspect --artifact path\to\bcnlt-artifact
```

Encode literal text or a file:

```powershell
python -m bcnlt.tokenizer_cli encode --artifact path\to\bcnlt-artifact --base path\to\base-tokenizer --mode text --text "hello"
python -m bcnlt.tokenizer_cli encode --artifact path\to\bcnlt-artifact --base path\to\base-tokenizer --mode semantic_code --file sample.py
```

Verify the artifact before corpus generation or model adaptation:

```powershell
python -m bcnlt.tokenizer_cli verify --artifact path\to\bcnlt-artifact --base path\to\base-tokenizer
```

Verification checks the canonical manifest digest, required token inventory,
base file identity, base vocabulary identity, text round-trip, and semantic
record round-trip. Hugging Face loading sets `trust_remote_code=False`; local
artifacts load with `local_files_only=True`.

## Rejection conditions

Loading or verification fails when any of these conditions is observed:

- a manifest field, digest, or canonical serialization differs
- a base tokenizer file or vocabulary differs from the pinned identity
- base IDs are duplicated, sparse, or collide with a structural token
- structural tokens are missing, duplicated, reordered, or altered
- an ID is outside the declared combined vocabulary
- stream, mode, record, or field markers are missing or out of order
- a JSON field is malformed or a score field has the wrong value token
- a save target already contains files
- a round-trip verification changes text or semantic records

These failures are admission gates. An artifact that fails verification must not
be used to generate a training corpus.
