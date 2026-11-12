# Original BCNLT tokenizer: serialized-context accounting

## Scope and provenance

Active project root: `D:\BossCrafts Dev\BossmanX138\BCNLT` on BossCrafts-MSI.
Development branch: `feature/tokenizer-payload-accounting`, based on local
`ab6bdf2` (the original Python prototype plus stewardship history).

This is original-line development, independent of BCNLM. The local main branch
was 7 commits ahead and 2 behind its recorded origin/main. That divergence was
not merged, reset, or published. The newer multilingual source and repaired
runtime used by model-training projects were not modified or replaced by this
prototype. No model calls, source execution, learning-memory mutation, training,
checkpoint changes, push, or deployment were performed.

## Change

`--payload-report` compares the raw Python source, compact JSONL of complete
contextual units, and the concatenated raw-plus-semantic context using one
selected tiktoken encoding. Full payloads, including the augmented string, are
encoded directly. Stable hashes and byte/character sizes allow exact comparison.

If any encoder step fails, all representations are recounted with the explicitly
non-BPE `regex-piece-estimate` fallback. Model-token availability and fallback
reason are exposed. Empty denominators yield null, not an invented savings
percentage. Literal special-token spellings are counted as ordinary text.

The original event benchmark's regex fallback is now correctly named as an
estimate. It no longer silently drops to fallback solely because source contains
a literal special-token spelling. Existing event/count fields remain intact.

Source CRLF newlines are preserved for the new payload comparison, and optional
UTF-8 BOMs are omitted on decoding. Hashes describe the exact compared text
encoded as UTF-8, not original file bytes.

## Measurements

Measured with CPython 3.13.9, tiktoken 0.14.0, `cl100k_base`, from this Windows
checkout. Relative input paths were `fixtures/sample.py` and
`fixtures/ambiguous.py`; symbol identities/path metadata are included in JSONL,
so different paths can change serialized counts.

| Fixture | Events | Raw tokens | Semantic JSONL tokens | Raw + JSONL tokens |
| --- | ---: | ---: | ---: | ---: |
| sample.py | 31 | 49 | 3,614 | 3,672 |
| ambiguous.py | 21 | 37 | 2,387 | 2,432 |

This exposes the overhead of sending **full metadata**, not a result for a
selected, compact fact representation or native BCNLT token IDs. It does not
establish a downstream quality gain or loss, and it is not a promotion gate.

Run from the active root:

```powershell
python -m bcnlt.cli fixtures/sample.py --payload-report .work/sample-payload-cost.json
python -m bcnlt.cli fixtures/ambiguous.py --payload-report .work/ambiguous-payload-cost.json
```

## Verification and contained outputs

- 46 tests passed, including 19 new tests, with no exclusions.
- The same 46 tests passed from an independent clean-folder source copy inside
  `.work/portability`, with no Git metadata or dependency on the original source
  directory. This is source portability under the declared system runtime, not
  a bundled standalone distribution.
- All 25 Python source/test files passed AST compilation checks.
- `git diff --check` passed using Codex's bundled Git 2.56.0. The system Git
  2.54.0 incorrectly reported that this valid working tree was not a working
  tree for diff operations. No Git settings were changed.

Tests used system runtime `C:\ProgramData\anaconda3\python.exe` (pytest 8.4.2).
TEMP, TMP, pytest caches/basetemps, and tiktoken cache were explicitly redirected
under `.work`, with bytecode writing disabled. `.work` is ignored. This run did
not change installed dependencies or download encoder data.

The existing cl100k encoding data was copied from the machine's cached
`data-gym-cache/9b5ad71b2ce5302211f9c61530b329a4922fc6a4` file into the project.
Its SHA-256 is
`223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`.
Tiktoken is MIT-licensed; the runtime/package remains an external prerequisite,
and the cache copy is a local validation input, not a redistributed release.

## Next tokenizer milestone

Evaluate a compact semantic serialization and retrieval selection against this
measured full-metadata baseline. Require independent semantic-correctness and
downstream answer-quality checks at equal model-context budgets before claiming
token savings. Keep native-tokenizer training and BCNLM memory retrieval as
separate experiments with separately pinned sources and result reports.
