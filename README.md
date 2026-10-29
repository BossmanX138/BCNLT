# BCNLT

BCNLT, the Bosscrafts Coding Neural Learning Tokenizer, is a working v0.1 prototype for turning Python source into contextual semantic units instead of plain token streams.

## What It Does

- Parses Python with the built-in `ast` module.
- Emits contextual units for `DEFINE`, `REFERENCE`, `ASSIGN`, `CALL`, `RETURN`, `BRANCH`, `LOOP`, `MEMBER_ACCESS`, `IMPORT`, `OPERATOR`, `SCOPE_ENTER`, `SCOPE_EXIT`, `LITERAL`, and `UNKNOWN_SEMANTIC`.
- Preserves surface text, source span, scope, symbol identity, definition/reference links, AST relation, common IR mapping, and the recent semantic history.
- Maintains a rolling semantic history of up to eight meaningful events.
- Scores uncertainty using structural signals such as missing definitions, dynamic calls, member access, and unknown semantics.
- Runs UYR, Uncertainty Yo-Yo Retrieval, by expanding local context and retrieving targeted symbol definitions until confidence improves, budget is exhausted, or no useful context remains.
- Routes unsupported language features through a wildcard resolver that records category, hypotheses, evidence, and confidence.
- Provides a compact Common Coding Core IR and a Python adapter that can render a limited IR subset back into valid Python.
- Includes a simple project index, parallel file preprocessing helper, and incremental invalidation abstraction.
- Provides a CLI, fixtures, tests, and a benchmark harness.

## Quick Start

```powershell
python -m pytest
python -m bcnlt.cli fixtures/sample.py --jsonl work-units.jsonl --trace trace.txt --benchmark
python -m bcnlt.cli fixtures/ambiguous.py --jsonl ambiguous.jsonl --trace ambiguous-trace.txt
python -m bcnlt.cli fixtures --compare-report comparison.json --workers 2
python -m bcnlt.cli fixtures/sample.py --trace-comparison side-by-side.txt --benchmark
python -m bcnlt.cli fixtures/sample.py --meaning-report meaning.json
python -m bcnlt.cli fixtures/sample.py --model-eval-report model-context.json
python -m bcnlt.cli fixtures/ambiguous.py --jsonl ambiguous.jsonl --wildcard-memory wildcards.jsonl
```

## BCNLT Tokenizer v1

The `bcnlt-tokenizer-v1` artifact combines an immutable Hugging Face base
vocabulary with a deterministic BCNLT structural-token inventory. The three
explicit modes are text, raw_code, and semantic_code. Existing base token IDs
remain unchanged; BCNLT structural IDs are appended contiguously after the base
vocabulary.

Text and raw code preserve their base-token payloads. In semantic mode,
canonical semantic records round-trip exactly through IDs and back to canonical
JSONL. Python source does not round-trip from semantic mode because parsing
intentionally converts source into meaning-bearing records rather than retaining
all formatting, comments, and spelling choices.

Build and verify a local artifact with:

```powershell
python -m bcnlt.tokenizer_cli build --base path\to\base-tokenizer --output path\to\bcnlt-artifact --name organization/model --revision pinned-revision
python -m bcnlt.tokenizer_cli verify --artifact path\to\bcnlt-artifact --base path\to\base-tokenizer
```

The artifact contains a canonical manifest and the exact structural inventory;
it does not copy model weights. See `docs/bcnlt-tokenizer-v1.md` for the token
classes, canonical field order, score bands, persistence contract, and failure
conditions.

## Architecture

- `bcnlt.ir`: common coding core IR. It keeps universal programming concepts in the core and leaves language-specific oddities to extensions.
- `bcnlt.core`: contextual unit and source document models.
- `bcnlt.adapters.base`: language adapter interface.
- `bcnlt.adapters.python`: Python AST adapter and limited IR-to-Python renderer.
- `bcnlt.cache`: symbol and definition/reference index.
- `bcnlt.uncertainty`: structural uncertainty scorer plus a model-entropy interface stub.
- `bcnlt.uyr`: UYR context expansion and targeted retrieval loop.
- `bcnlt.wildcard`: unknown semantic resolver and append-only wildcard memory.
- `bcnlt.preprocessing`: single-file and parallel multi-file preprocessing helpers.
- `bcnlt.incremental`: changed-file and changed-region invalidation hooks.
- `bcnlt.cli`: JSONL and trace command-line interface.
- `bcnlt.tokenizer`: explicit-mode hybrid tokenizer and strict semantic decoder.
- `bcnlt.hf_tokenizer`: local Hugging Face base adapter and artifact persistence.
- `bcnlt.tokenizer_cli`: build, inspect, encode, and verify commands.
- `bcnlt.benchmark`: lexical-token count versus BCNLT event count and latency.

## BCNLT vs Standard Tokenizer Report

The comparison report is the first head-to-head harness for standard tokenization versus BCNLT semantic units.

```powershell
python -m bcnlt.cli path\to\python_project --compare-report comparison.json --workers 4
```

For a single file, use:

```powershell
python -m bcnlt.cli path\to\file.py --compare-report file-comparison.json --trace-comparison file-trace.txt
```

The report includes:

- Standard token count, using `tiktoken:cl100k_base` when installed.
- A deterministic regex BPE-style fallback when `tiktoken` is unavailable.
- Python lexical token count.
- BCNLT semantic event count.
- Event-to-standard-token ratio.
- Uncertain event count.
- UYR activation count.
- Wildcard count.
- Per-file latency and project totals.

The side-by-side trace shows ordinary token pieces next to BCNLT semantic events, which makes the comparison visible instead of only numeric.

## Meaning Retrieval Evaluation

The meaning evaluator is the first deterministic test of whether BCNLT keeps enough structure to retrieve useful code facts. It generates definition-location probes from a Python file, then compares:

- **Standard text retrieval**: find the first raw text occurrence of a symbol and return nearby lines.
- **BCNLT semantic retrieval**: find the matching semantic `DEFINE` unit by symbol identity and source line, then return nearby semantic context.

Run it with:

```powershell
python -m bcnlt.cli path\to\file.py --meaning-report meaning.json --meaning-max-probes 50
```

This is not a full LLM comprehension benchmark yet. It is a deterministic retrieval-quality check that asks whether the context source contains the actual target definition span. It is useful because it catches cases where plain text sees the right word but lands on the wrong meaning.

## Model Context Answer Evaluation

The model-context evaluator is the next step after retrieval. It gives both retrieved contexts the same question and asks a model interface to answer from the supplied context.

```powershell
python -m bcnlt.cli path\to\file.py --model-eval-report model-context.json --model-eval-max-probes 50
```

The default model is `oracle-definition-line`, a deterministic local stand-in that answers definition-line questions only from the context text it receives. This verifies the pipeline before plugging in an API-backed LLM.

To use an OpenAI-compatible chat-completions model:

```powershell
$env:OPENAI_API_KEY = "..."
python -m bcnlt.cli path\to\file.py --model-eval-report model-openai.json --model-eval-provider openai --model-eval-model gpt-4.1-mini --model-eval-max-probes 10
```

To use a local Ollama model:

```powershell
python -m bcnlt.cli path\to\file.py --model-eval-report model-ollama.json --model-eval-provider ollama --model-eval-model qwen2.5-coder:7b --model-eval-max-probes 10
```

The checked-in Wizard-Phi v0.2 profile validates and imports the promoted local release, performs a real readiness completion, and runs this evaluation end to end:

```powershell
.\scripts\connect_wizard_phi_v02.ps1
```

See `integrations/wizard_phi_v02/README.md` for its pinned source lineage and runtime settings.

The Runeforge Mk2 7B profile preserves the merged FP16 source, validates the pinned
Q4_K_M runtime, registers `runeforge-mk2:7b`, and requires a real readiness
completion before evaluation:

```powershell
.\scripts\connect_runeforge_mk2.ps1
```

Run the controlled 46-probe comparison against Wizard-Phi v0.2 with:

```powershell
python .\scripts\run_bcnlt_head_to_head_mk2.py
```

The comparison writes `outputs/head-to-head-wizard-v02-vs-runeforge-mk2/`.
Use `scripts/import_bcnlt_observations.py` to add its answers to the semantic-pair
database without deleting observations from earlier runs.

The output compares:

- Standard raw-text context answer accuracy.
- BCNLT semantic context answer accuracy.
- Expected definition line.
- Standard and BCNLT answers.
- The exact context supplied to each side.

This is still not a full human-grade comprehension benchmark. It is a controlled test of whether the retrieved context contains enough structure for a model-like answerer to produce the right fact.

## Wildcard Memory

The wildcard resolver can now append unfamiliar constructs it has already seen into a JSONL memory file. On later runs, BCNLT reloads that memory, recognizes matching unknown shapes, and returns a stronger provisional meaning with memory-hit evidence.

```powershell
python -m bcnlt.cli fixtures/ambiguous.py --jsonl ambiguous.jsonl --wildcard-memory wildcards.jsonl
```

Each memory row records:

- stable wildcard signature
- source surface text
- AST relation
- provisional category
- hypotheses
- evidence
- confidence
- seen count

This is intentionally append-only in v0.1. The latest in-memory view merges repeated sightings, while the file preserves the learning trail for later review, pruning, or training-data extraction.

## Parallel Compute Strategy

BCNLT adds semantic work before a model sees code, so the scaling plan is to recover that cost by splitting clean compute phases across more cores. The prototype already includes `process_python_files(..., workers=N)` as the first process-ready seam for multi-file preprocessing.

Parallel-friendly stages:

- Parse separate files at the same time with a process pool.
- Build contextual units per file independently, then merge symbol indexes.
- Score uncertainty across many units independently.
- Run UYR for separate high-uncertainty units in parallel.
- Batch embedding, retrieval, and entropy/model-scoring work through GPU-ready batching when those backends are added.
- Benchmark many files at once.

Mostly sequential stages:

- Scope tracking inside a single file.
- Ordered statement walking where later references depend on earlier definitions.
- Data-flow refinement where each pass depends on the previous pass.
- A single model forward pass, unless the model runtime itself supports batching or distributed compute.

The intended execution shape is:

```text
coordinator
  -> per-file process pool
  -> per-unit uncertainty/UYR jobs
  -> batched retrieval or model scoring
  -> deterministic merge into one project graph
```

The key rule is to avoid shared mutable state during parallel work. Workers should emit file-local units, symbol records, uncertainty records, and retrieval evidence; the coordinator performs a deterministic merge afterward.

## Terminology

- **Contextual unit**: one semantic event with surface text, span, scope, symbol links, IR mapping, and recent history.
- **Common Coding Core IR**: compact representation of universal coding concepts such as values, variables, functions, calls, assignment, loops, branches, comparisons, data structures, I/O, and errors.
- **Extension**: a language-specific construct kept outside the core IR.
- **Wildcard**: the resolver path for unfamiliar constructs.
- **Wildcard memory**: append-only JSONL evidence that lets BCNLT recognize and refine unknown constructs it has already encountered.
- **UYR**: repeated uncertainty check, targeted retrieval, and re-evaluation.

## Current Limits

- Python parsing uses `ast`, not Tree-sitter.
- Scope and data-flow links are useful but intentionally lightweight.
- Type inference is not implemented.
- Model entropy is stubbed behind an interface.
- Wildcard memory is local JSONL persistence, not a vetted global knowledge base yet.
- IR rendering supports a narrow subset: modules, assignments, functions, returns, literals, variables, and calls.
- Incremental support is an invalidation abstraction, not a full changed-AST rewriter.
- The benchmark uses Python lexical token counts as the ordinary-token baseline, not a real BPE tokenizer yet.

## Experiment Plan

1. Compare event count and latency against lexical token counts across real BossCrafts Python files.
2. Add a real BPE tokenizer baseline.
3. Add Tree-sitter as an optional parser backend.
4. Expand symbol resolution with import graphs and cross-file definitions.
5. Add type evidence retrieval for high-uncertainty member access and calls.
6. Track whether UYR reduces downstream model mistakes on code explanation and transformation tasks.
7. Ablate semantic history, UYR, wildcard, and IR mapping independently.

## Runeforge Mk2 native tokenizer training

The repository now includes the BCNLT hybrid tokenizer, provenance-gated corpus pipeline, deterministic curriculum and admission gates, packed-dataset loader, sparse embedding/output extensions, and guarded Runeforge Mk2 training handoff. The preserved Mk2 checkpoint remains immutable, and validation-only is the default for the full-run launcher.

See [docs/bcnlt-runeforge-training.md](docs/bcnlt-runeforge-training.md) for the exact WSL build order, artifact boundaries, current Hugging Face access blocker, smoke requirements, and explicit full-run launch gate.

The same guide now includes the gated adaptive-vocabulary v5 experiment: a
read-only legacy-corpus rebuild, preliminary multilingual audit, resumable WSL
phase runner, tokenizer-only default, and explicit GPU opt-in. Its promotion
proposal cannot merge or publish a model.
