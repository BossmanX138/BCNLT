# BCNLT Multilingual Adapter Carousel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the existing Python BCNLT semantic module into one multilingual adapter with a 14-language semantic carousel, then train one balanced multilingual Runeforge adapter from the admitted records.

**Architecture:** `BCNLTMultilingualAdapter` is the only public multilingual integration. Its `LanguageCarousel` selects a versioned semantic cartridge from trusted source metadata, and one shared Tree-sitter engine emits the existing canonical BCNLT records. Python keeps a compatibility facade, all languages use the same tokenizer and corpus pipeline, and one QLoRA/embedding extension is trained across the combined corpus.

**Tech Stack:** Python 3.10+, `tree-sitter-language-pack==1.22.1`, `tree-sitter==0.26.0`, SQLite, pytest, Hugging Face Datasets, PyTorch, Unsloth/PEFT/bitsandbytes, PowerShell, WSL2, CUDA

**Spec:** `docs/superpowers/specs/2026-10-28-bcnlt-multilingual-semantic-training-design.md`

## Global Constraints

- The public integration is one `BCNLTMultilingualAdapter`; the 14 language cartridges are parser/semantic mappings, not separate LoRAs.
- Supported languages are Python, JavaScript, TypeScript, C, C++, C#, Java, Go, Rust, PHP, Ruby, Kotlin, Swift, and Shell.
- Preserve all 32,000 original Runeforge IDs and all 73 existing BCNLT IDs; tokenizer v1 remains exactly 32,073 IDs.
- Preserve the Runeforge Mk2 7B source checkpoint and standalone adapters; never overwrite either.
- Use only explicit language metadata from the source registry; never guess a language from content.
- Freeze parser package, grammar source revision, ABI, cache-file hashes, and per-platform binary hashes before generating corpus identity.
- Canonical fixtures must emit byte-identical semantic records on Windows and WSL after LF normalization.
- Reject parser `ERROR` or `MISSING` nodes, unknown provenance, disallowed licenses, ambiguous dialects, and holdout collisions.
- Map 100% of required shared golden concepts and at least 80% of admitted non-scope/non-unknown units into `CommonCodingCore`; language extensions cannot replace available core meanings.
- Score every unit's knowledge boundary; confidence below 0.72 triggers at most four UYR rounds and eight added context units before wildcard/unknown handling.
- Require at least 85% mapped named-node coverage per document, at most 10% unknown events per document, and at most 7.5% unknown events per language.
- Pack one language per sequence; every model-visible task begins with an existing-base-token language field.
- Full admission requires at least 10,000 packed records per language, 140,000 total records, and 50,000,000 encoded tokens.
- No language may exceed 12% of records or tokens; all six curriculum families, including knowledge-boundary reasoning, must exist for every language.
- Windows performs authenticated Hugging Face ingestion; WSL consumes content-addressed local shards for CUDA training.
- The large corpus run and full training remain locked until all 14 sample gates and the multilingual smoke gate pass.

## Review Focus

- Non-ASCII source positions use UTF-8 byte columns, and Windows/WSL must produce identical canonical spans; Task 4 adds Unicode span tests and Task 5 adds cross-platform fixture comparison.
- Ambiguous forms such as `.h`, JSX/TSX, embedded PHP, ERB, Kotlin scripts, and Zsh must be rejected with stable reason codes; Task 12 adds the dialect matrix tests.
- Nested Tree-sitter `ERROR`/`MISSING` nodes must quarantine a whole document even when the root reports a valid language; Tasks 2 and 4 add recursive diagnostic tests.
- Packing and batching may interleave languages but may never concatenate them into one sequence or omit the model-visible language field; Task 13 tests both conditions.
- Repeated wildcard observations must never turn a provisional hypothesis into a fact without resolving evidence; Tasks 4, 13, and 14 test memory identity, evidence reuse, UYR improvement, and false certainty.
- Resume must fail after any parser, grammar, semantic-spec, source-registry, tokenizer, corpus, training-config, or base-model hash changes; Tasks 2, 13, and 15 add lineage mismatch tests.

---

### Task 1: Preserve and checkpoint the current Python corpus baseline

**Files:**
- Modify: `bcnlt/corpus/admission.py`
- Modify: `bcnlt/corpus/curriculum.py`
- Modify: `bcnlt/corpus/sources.py`
- Modify: `bcnlt/corpus/tasks/code_to_semantic.py`
- Modify: `bcnlt/corpus/tasks/mixed_instruction.py`
- Modify: `bcnlt/corpus/tasks/semantic_continuation.py`
- Modify: `bcnlt/corpus/tasks/semantic_to_code.py`
- Modify: `bcnlt/semantic_record.py`
- Modify: `scripts/export_bcnlt_training_shards.py`
- Modify: `scripts/ingest_bcnlt_corpus.py`
- Modify: `scripts/preflight_bcnlt_runeforge.ps1`
- Create: `scripts/collect_bcnlt_preflight_probe.py`
- Test: `tests/test_corpus_ingestion.py`
- Test: `tests/test_semantic_record.py`

**Interfaces:**
- Consumes: current tokenizer v1, Python adapter, and existing corpus schema.
- Produces: a clean, tested checkpoint containing the already-written live-source schema fixes, bounded curriculum contexts, 4,096-token admission limit, deterministic near-duplicate buckets, and mixed Windows/WSL preflight evidence.

- [ ] **Step 1: Inspect the pending baseline diff**

Run: `git diff --check` and `git status --short`

Expected: no whitespace errors; only the listed pending baseline files plus this plan are present.

- [ ] **Step 2: Run the focused regression suite**

Run: `python -m pytest tests/test_corpus_ingestion.py tests/test_semantic_record.py tests/test_curriculum_generation.py tests/test_corpus_admission.py tests/test_corpus_packing.py tests/test_runeforge_tokenizer_build.py -q`

Expected: PASS, including NVIDIA JSON-string status parsing, JSON-safe bytes/complex/Ellipsis canonicalization, bounded curriculum records, 4,096-token rejection, and deterministic near-duplicate admission.

- [ ] **Step 3: Run the complete Python-first baseline suite**

Run: `python -m pytest -q`

Expected: all existing tests pass before adding parser dependencies or multilingual behavior.

- [ ] **Step 4: Commit the baseline separately**

```powershell
git add bcnlt/corpus bcnlt/semantic_record.py scripts/collect_bcnlt_preflight_probe.py scripts/export_bcnlt_training_shards.py scripts/ingest_bcnlt_corpus.py scripts/preflight_bcnlt_runeforge.ps1 tests/test_corpus_ingestion.py tests/test_semantic_record.py
git commit -m "fix: stabilize live BCNLT corpus generation"
```

### Task 2: Freeze the cross-platform Tree-sitter parser bundle

**Files:**
- Modify: `pyproject.toml`
- Create: `bcnlt/parsers/__init__.py`
- Create: `bcnlt/parsers/bundle.py`
- Create: `configs/bcnlt_parser_bundle_v1.json`
- Create: `scripts/freeze_bcnlt_parser_bundle.py`
- Test: `tests/test_parser_bundle.py`

**Interfaces:**
- Produces: `ParserBundleEntry`, `ParserBundleManifest`, `load_parser_bundle(path: str | Path) -> ParserBundleManifest`, `verify_parser_bundle(manifest, root, platform) -> None`, and `freeze_parser_bundle(config, cache_root, output_root) -> ParserBundleManifest`.
- Produces: one manifest for grammar IDs `python`, `javascript`, `typescript`, `c`, `cpp`, `c_sharp`, `java`, `go`, `rust`, `php`, `ruby`, `kotlin`, `swift`, and `bash`.

- [ ] **Step 1: Write failing manifest and corruption tests**

In `tests/test_parser_bundle.py`, add tests that assert all 14 entries are required, grammar revisions are 40-character source SHAs, ABI versions are recorded, Windows and WSL file hashes are distinct platform entries, recursive `ERROR`/`MISSING` diagnostics fail, and one altered cache byte causes `verify_parser_bundle` to raise.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_parser_bundle.py -q`

Expected: FAIL because `bcnlt.parsers.bundle` does not exist.

- [ ] **Step 3: Add exact parser dependencies and bundle types**

Add a `multilingual` optional dependency containing `tree-sitter-language-pack==1.22.1` and `tree-sitter==0.26.0`. Implement frozen dataclasses whose canonical sorted-JSON digest covers package versions, grammar repositories/revisions, ABI, cache-relative paths, file sizes, and SHA-256 values.

- [ ] **Step 4: Implement the offline bundle freezer**

Implement `freeze_parser_bundle(...)` to download only the 14 configured grammars into a staging cache, resolve each upstream revision from `grammar-provenance-1.22.1.json`, copy cache files to a content-addressed output, and refuse any grammar without a source revision. Limit parser inputs to 200 KiB and 500 ms.

- [ ] **Step 5: Run parser-bundle unit tests**

Run: `python -m pytest tests/test_parser_bundle.py -q`

Expected: PASS.

- [ ] **Step 6: Run the live cross-platform capability check**

Run the freezer once on Windows and verify the WSL platform bundle against the same logical manifest. Expected: all 14 grammars load without network access from the frozen destination.

- [ ] **Step 7: Commit**

```powershell
git add pyproject.toml bcnlt/parsers configs/bcnlt_parser_bundle_v1.json scripts/freeze_bcnlt_parser_bundle.py tests/test_parser_bundle.py
git commit -m "feat: freeze multilingual parser bundle"
```

### Task 3: Define the single adapter facade and language carousel

**Files:**
- Modify: `bcnlt/core.py`
- Modify: `bcnlt/semantic_record.py`
- Modify: `bcnlt/canonical.py`
- Modify: `bcnlt/adapters/base.py`
- Modify: `bcnlt/adapters/__init__.py`
- Create: `bcnlt/semantic_schema.py`
- Create: `bcnlt/languages.py`
- Create: `bcnlt/adapters/spec.py`
- Create: `bcnlt/adapters/coverage.py`
- Create: `bcnlt/adapters/carousel.py`
- Test: `tests/test_adapter_carousel.py`
- Modify: `tests/test_semantic_record.py`

**Interfaces:**
- Produces: `LanguageDefinition`, `LanguageSemanticSpec`, `LanguageCartridge`, `LanguageCarousel`, `SemanticCoverage`, `SemanticExtraction`, and `BCNLTMultilingualAdapter`.
- Produces semantic schema identity `bcnlt-semantic-record-v2` while retaining the existing structural token IDs.
- Produces: `BCNLTMultilingualAdapter.parse(source: str, *, language: str, path: str, dialect: str | None = None) -> SourceDocument` and `extract(document: SourceDocument) -> SemanticExtraction`.
- `LanguageAdapter.render(node: IRNode) -> str` becomes optional; its default raises `NotImplementedError` without making subclasses abstract.

- [ ] **Step 1: Write failing carousel contract tests**

Assert that the carousel lists exactly 14 canonical language IDs, rejects aliases and unknown IDs unless the central registry maps them, selects only from explicit metadata, returns one public adapter object, and never exposes a model/LoRA per cartridge.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_adapter_carousel.py -q`

Expected: FAIL because the carousel types do not exist.

- [ ] **Step 3: Implement language definitions and semantic-spec types**

Define immutable language entries with canonical ID, grammar ID, accepted extensions, deferred extensions, dialect rules, and semantic-spec version. Define immutable node, identifier-role, scope, and uncertainty rules whose resolved canonical hash includes inherited family rules.

- [ ] **Step 4: Extend the canonical semantic schema**

Add the versioned uncertainty keys `dynamic_call`, `dynamic_binding`, `macro_expansion`, `overload_unresolved`, `preprocessor_condition`, `unknown_node`, `knowledge_state`, `uyr_rounds`, `uncertainty_signals`, and `wildcard_category`. Keep language in the document/example envelope and prove all existing token IDs remain unchanged.

- [ ] **Step 5: Implement the public facade**

`LanguageCarousel.for_language(language: str) -> LanguageCartridge` must require an exact registered ID. `BCNLTMultilingualAdapter` delegates parsing/extraction to that cartridge through the shared engine supplied in Task 4.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_adapter_carousel.py tests/test_semantic_record.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add bcnlt/core.py bcnlt/semantic_record.py bcnlt/canonical.py bcnlt/semantic_schema.py bcnlt/languages.py bcnlt/adapters tests/test_adapter_carousel.py tests/test_semantic_record.py
git commit -m "feat: add BCNLT language carousel"
```

### Task 4: Extract Python's reusable semantics into the shared Tree-sitter engine

**Files:**
- Modify: `bcnlt/ir.py`
- Modify: `bcnlt/uncertainty.py`
- Modify: `bcnlt/uyr.py`
- Modify: `bcnlt/wildcard.py`
- Create: `bcnlt/adapters/tree_sitter.py`
- Create: `bcnlt/adapters/symbols.py`
- Test: `tests/test_tree_sitter_adapter.py`
- Modify: `tests/test_uncertainty_uyr_wildcard.py`

**Interfaces:**
- Consumes: `SemanticCoverage`, `SemanticExtraction`, and `LanguageCartridge` from Task 3.
- Produces: `LexicalSymbolTable`, `KnowledgeBoundaryAssessment`, and `TreeSitterSemanticEngine.extract(document: SourceDocument, cartridge: LanguageCartridge) -> SemanticExtraction`.
- `SemanticExtraction.units` is an ordered tuple of existing `ContextualUnit` values; `.coverage` records total, excluded, mapped, and unknown named nodes plus scope/link validation.

- [ ] **Step 1: Write failing traversal, scope, and Unicode tests**

Test deterministic source-order traversal, UTF-8 byte columns for emoji identifiers/strings, balanced scope events, shadowing, unresolved references, stable local symbol IDs, recent-history limit 8, and whole-document rejection for nested `ERROR` or `MISSING` nodes.

- [ ] **Step 2: Write failing coverage tests**

Assert exclusions are spec-hashed, mapped coverage is `mapped / (total - excluded)`, unknown-event percentage is explicit, documents below 85% or above 10% unknown fail admission, required shared golden concepts map 100% to Common Coding Core, corpus core coverage below 80% fails, and a spec cannot exclude a node it also maps.

- [ ] **Step 3: Write failing knowledge-boundary tests**

Assert every unit receives initial/final scores and one state from `known`, `retrieved`, `provisional`, or `unresolved`; confidence below 0.72 invokes no more than four UYR rounds/eight context units; retrieval can improve confidence only with recorded evidence; wildcard signatures include language/spec identity; memory reuse raises provisional confidence without promoting it to fact.

- [ ] **Step 4: Run tests and verify failure**

Run: `python -m pytest tests/test_tree_sitter_adapter.py tests/test_uncertainty_uyr_wildcard.py -q`

Expected: FAIL because the shared engine does not exist.

- [ ] **Step 5: Implement lexical symbols and deterministic emission**

Implement document-local lexical scope stacks, source-ordered definitions/references, explicit null links, complete Common Coding Core factories, namespace-prefixed IR extensions, recursive diagnostics, and coverage accounting. Do not perform runtime dispatch, macro expansion, or cross-file type inference.

- [ ] **Step 6: Implement the shared knowledge-boundary loop**

Generalize heuristic uncertainty signals beyond Python, run bounded UYR on low-confidence units, then use language-aware append-only wildcard memory for unresolved constructs. Preserve the evidence path in `KnowledgeBoundaryAssessment` and the semantic training envelope.

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/test_tree_sitter_adapter.py tests/test_uncertainty_uyr_wildcard.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add bcnlt/ir.py bcnlt/uncertainty.py bcnlt/uyr.py bcnlt/wildcard.py bcnlt/adapters/tree_sitter.py bcnlt/adapters/symbols.py tests/test_tree_sitter_adapter.py tests/test_uncertainty_uyr_wildcard.py
git commit -m "feat: extract shared semantic adapter engine"
```

### Task 5: Convert Python into the first carousel cartridge

**Files:**
- Modify: `bcnlt/adapters/python.py`
- Create: `tests/language_fixtures/python/core.py`
- Create: `tests/language_fixtures/python/core.expected.json`
- Modify: `tests/test_python_adapter.py`
- Modify: `tests/test_python_semantic_coverage.py`
- Create: `tests/test_multilingual_golden.py`

**Interfaces:**
- Produces: `build_python_spec() -> LanguageSemanticSpec` and a compatibility `PythonAdapter` that selects `language="python"` through the carousel.
- Produces: the reusable golden-fixture loader used by Tasks 6-11.

- [ ] **Step 1: Freeze Python compatibility and golden expectations**

Add tests for imports, functions, classes, parameters, shadowing, calls, assignments, branches, loops, returns, member access, advanced async/context/try/match/comprehension events, unknown nodes, and existing symbol-link expectations.

- [ ] **Step 2: Add cross-platform canonical comparison**

The fixture test must write canonical record bytes on Windows and WSL and assert equality after LF normalization, including non-ASCII spans.

- [ ] **Step 3: Run tests and verify failure**

Run: `python -m pytest tests/test_python_adapter.py tests/test_python_semantic_coverage.py tests/test_multilingual_golden.py -q`

Expected: FAIL until the Python cartridge uses the carousel/shared engine.

- [ ] **Step 4: Implement the Python spec and compatibility facade**

Move reusable behavior out of `python.py`; keep Python-only node mappings and the limited IR renderer there. Preserve `PythonAdapter.parse`, `to_units`, and `render` for old callers.

- [ ] **Step 5: Run focused Python compatibility tests**

Run: `python -m pytest tests/test_python_adapter.py tests/test_python_semantic_coverage.py tests/test_multilingual_golden.py -q`

Expected: PASS.

- [ ] **Step 6: Run the complete regression suite**

Run: `python -m pytest -q`

Expected: PASS with unchanged tokenizer IDs and no Python corpus regression.

- [ ] **Step 7: Commit**

```powershell
git add bcnlt/adapters/python.py tests/language_fixtures/python tests/test_python_adapter.py tests/test_python_semantic_coverage.py tests/test_multilingual_golden.py
git commit -m "refactor: make Python the first carousel cartridge"
```

### Task 6: Add JavaScript and TypeScript cartridges

**Files:**
- Create: `bcnlt/adapters/javascript.py`
- Create: `bcnlt/adapters/typescript.py`
- Create: `tests/language_fixtures/javascript/core.js`
- Create: `tests/language_fixtures/javascript/core.expected.json`
- Create: `tests/language_fixtures/typescript/core.ts`
- Create: `tests/language_fixtures/typescript/core.expected.json`
- Test: `tests/test_javascript_typescript_adapters.py`

**Interfaces:**
- Produces: `build_javascript_spec()` and `build_typescript_spec()`; TypeScript inherits JavaScript mappings through a resolved spec hash.

- [ ] **Step 1: Write failing language fixtures and edge tests**

Cover function/class/arrow definitions, lexical `let`/`const`, `var` function scope, imports/exports, calls, properties, branches/loops, async/await, TypeScript interfaces/types/generics, and explicit uncertainty for dynamic property access. Assert JSX/TSX and `.d.ts` are not admitted.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_javascript_typescript_adapters.py tests/test_multilingual_golden.py -q`

Expected: FAIL because both cartridges are absent.

- [ ] **Step 3: Implement both specs and register them**

Keep JavaScript and TypeScript as separate reported languages even where rules are inherited. Namespace extensions with `javascript:` and `typescript:`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_javascript_typescript_adapters.py tests/test_multilingual_golden.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add bcnlt/adapters/javascript.py bcnlt/adapters/typescript.py tests/language_fixtures/javascript tests/language_fixtures/typescript tests/test_javascript_typescript_adapters.py
git commit -m "feat: add JavaScript and TypeScript cartridges"
```

### Task 7: Add C, C++, and C# cartridges

**Files:**
- Create: `bcnlt/adapters/c.py`
- Create: `bcnlt/adapters/cpp.py`
- Create: `bcnlt/adapters/csharp.py`
- Create: `tests/language_fixtures/c/core.c`
- Create: `tests/language_fixtures/c/core.expected.json`
- Create: `tests/language_fixtures/cpp/core.cpp`
- Create: `tests/language_fixtures/cpp/core.expected.json`
- Create: `tests/language_fixtures/csharp/core.cs`
- Create: `tests/language_fixtures/csharp/core.expected.json`
- Test: `tests/test_c_family_adapters.py`

**Interfaces:**
- Produces: `build_c_spec()`, `build_cpp_spec()`, and `build_csharp_spec()`; C++ may inherit C mappings but has a distinct resolved hash and metrics.

- [ ] **Step 1: Write failing family fixtures**

Cover functions/methods, parameters, structs/classes, namespaces, fields, pointers/references, calls, member access, loops/branches, returns, imports/includes/usings, overload uncertainty, C preprocessor uncertainty, C++ templates/lambdas, and C# properties/LINQ/async. Assert ambiguous `.h` and Objective-C forms are rejected.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_c_family_adapters.py tests/test_multilingual_golden.py -q`

Expected: FAIL.

- [ ] **Step 3: Implement and register the three specs**

Use `c:`, `cpp:`, and `csharp:` extension namespaces. Do not claim macro expansion or overload resolution.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_c_family_adapters.py tests/test_multilingual_golden.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add bcnlt/adapters/c.py bcnlt/adapters/cpp.py bcnlt/adapters/csharp.py tests/language_fixtures/c tests/language_fixtures/cpp tests/language_fixtures/csharp tests/test_c_family_adapters.py
git commit -m "feat: add C family cartridges"
```

### Task 8: Add Java and Kotlin cartridges

**Files:**
- Create: `bcnlt/adapters/java.py`
- Create: `bcnlt/adapters/kotlin.py`
- Create: `tests/language_fixtures/java/core.java`
- Create: `tests/language_fixtures/java/core.expected.json`
- Create: `tests/language_fixtures/kotlin/core.kt`
- Create: `tests/language_fixtures/kotlin/core.expected.json`
- Test: `tests/test_jvm_adapters.py`

**Interfaces:**
- Produces: `build_java_spec()` and `build_kotlin_spec()` with shared JVM helpers resolved into separate spec hashes.

- [ ] **Step 1: Write failing JVM fixtures**

Cover packages/imports, classes/interfaces, fields/properties, constructors, methods, parameters, calls, loops/branches, returns, Java lambdas/try-catch, Kotlin functions/classes/data classes/null-safe member access/lambdas/when. Assert `.kts` is rejected.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_jvm_adapters.py tests/test_multilingual_golden.py -q`

Expected: FAIL because both cartridges are absent.

- [ ] **Step 3: Implement both JVM cartridges**

Implement and register both specs, requiring at least 85% fixture coverage and at most 10% fixture unknown events.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_jvm_adapters.py tests/test_multilingual_golden.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add bcnlt/adapters/java.py bcnlt/adapters/kotlin.py tests/language_fixtures/java tests/language_fixtures/kotlin tests/test_jvm_adapters.py
git commit -m "feat: add Java and Kotlin cartridges"
```

### Task 9: Add Go and Rust cartridges

**Files:**
- Create: `bcnlt/adapters/go.py`
- Create: `bcnlt/adapters/rust.py`
- Create: `tests/language_fixtures/go/core.go`
- Create: `tests/language_fixtures/go/core.expected.json`
- Create: `tests/language_fixtures/rust/core.rs`
- Create: `tests/language_fixtures/rust/core.expected.json`
- Test: `tests/test_go_rust_adapters.py`

**Interfaces:**
- Produces: `build_go_spec()` and `build_rust_spec()`.

- [ ] **Step 1: Write failing systems-language fixtures**

Cover packages/modules, imports, structs, functions/methods, parameters, short declarations/`let`, calls, member access, loops/branches, returns, Go interfaces/defer/goroutines, Rust impl/traits/closures/match, and uncertainty for Rust macro expansion and lifetimes when no common-core equivalent exists.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_go_rust_adapters.py tests/test_multilingual_golden.py -q`

Expected: FAIL because both cartridges are absent.

- [ ] **Step 3: Implement both systems-language cartridges**

Implement and register both specs using `go:` and `rust:` extensions.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_go_rust_adapters.py tests/test_multilingual_golden.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add bcnlt/adapters/go.py bcnlt/adapters/rust.py tests/language_fixtures/go tests/language_fixtures/rust tests/test_go_rust_adapters.py
git commit -m "feat: add Go and Rust cartridges"
```

### Task 10: Add PHP and Ruby cartridges

**Files:**
- Create: `bcnlt/adapters/php.py`
- Create: `bcnlt/adapters/ruby.py`
- Create: `tests/language_fixtures/php/core.php`
- Create: `tests/language_fixtures/php/core.expected.json`
- Create: `tests/language_fixtures/ruby/core.rb`
- Create: `tests/language_fixtures/ruby/core.expected.json`
- Test: `tests/test_php_ruby_adapters.py`

**Interfaces:**
- Produces: `build_php_spec()` and `build_ruby_spec()`.

- [ ] **Step 1: Write failing dynamic-language fixtures**

Cover namespaces/modules, imports/includes, classes, functions/methods, parameters, variables, calls, properties, loops/branches, returns, PHP closures/traits, Ruby blocks/modules/singleton calls, and explicit dynamic binding/call uncertainty. Assert mixed HTML/PHP and ERB are rejected.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_php_ruby_adapters.py tests/test_multilingual_golden.py -q`

Expected: FAIL because both cartridges are absent.

- [ ] **Step 3: Implement both dynamic-language cartridges**

Implement both cartridges and retain null definition links where runtime lookup cannot be known lexically.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_php_ruby_adapters.py tests/test_multilingual_golden.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add bcnlt/adapters/php.py bcnlt/adapters/ruby.py tests/language_fixtures/php tests/language_fixtures/ruby tests/test_php_ruby_adapters.py
git commit -m "feat: add PHP and Ruby cartridges"
```

### Task 11: Add Swift and Shell cartridges and close the carousel

**Files:**
- Create: `bcnlt/adapters/swift.py`
- Create: `bcnlt/adapters/shell.py`
- Create: `tests/language_fixtures/swift/core.swift`
- Create: `tests/language_fixtures/swift/core.expected.json`
- Create: `tests/language_fixtures/shell/core.sh`
- Create: `tests/language_fixtures/shell/core.expected.json`
- Test: `tests/test_swift_shell_adapters.py`
- Modify: `tests/test_adapter_carousel.py`

**Interfaces:**
- Produces: `build_swift_spec()` and `build_shell_spec()`; Shell reports dialect `posix_sh` or `bash` under one language ID.

- [ ] **Step 1: Write failing fixtures and complete-carousel test**

Cover Swift imports/types/functions/properties/closures/optionals/guards and Shell functions/variables/expansions/commands/pipelines/conditionals/loops/returns. Assert `.zsh`/Fish are rejected and all 14 carousel slots now resolve.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_swift_shell_adapters.py tests/test_adapter_carousel.py tests/test_multilingual_golden.py -q`

Expected: FAIL because both cartridges are absent and the carousel is incomplete.

- [ ] **Step 3: Implement and register both cartridges**

Use explicit uncertainty for shell command resolution, expansion, and sourced files. Keep Bash and POSIX Shell dialect metadata outside structural token IDs.

- [ ] **Step 4: Run all adapter tests**

Run: `python -m pytest tests/test_*adapter*.py tests/test_multilingual_golden.py -q`

Expected: all 14 language fixtures pass independent coverage, event, scope, and link assertions.

- [ ] **Step 5: Commit**

```powershell
git add bcnlt/adapters/swift.py bcnlt/adapters/shell.py tests/language_fixtures/swift tests/language_fixtures/shell tests/test_swift_shell_adapters.py tests/test_adapter_carousel.py
git commit -m "feat: complete BCNLT language carousel"
```

### Task 12: Generalize provenance-first ingestion across the carousel

**Files:**
- Create: `configs/bcnlt_corpus_sources_v2.json`
- Modify: `bcnlt/corpus/database.py`
- Modify: `bcnlt/corpus/filters.py`
- Modify: `bcnlt/corpus/sources.py`
- Modify: `scripts/ingest_bcnlt_corpus.py`
- Create: `scripts/sample_bcnlt_multilingual_sources.py`
- Modify: `tests/test_corpus_ingestion.py`
- Create: `tests/test_multilingual_ingestion.py`

**Interfaces:**
- Extends `SourceDocumentRecord` with `language`, `dialect`, `registry_entry_sha256`, `parser_bundle_sha256`, `grammar_revision`, `semantic_spec_sha256`, and `extraction_rule_version`.
- Produces: `evaluate_source_document(...) -> FilterDecision`, generic `ingest_stack_stream(...)`, and `sample_languages(..., rows_per_language=100) -> MultilingualSampleReport`.

- [ ] **Step 1: Write failing schema, identity, license, and dialect tests**

Assert document IDs change with language/registry identity; old nonempty v1 databases fail with `rebuild_required`; The Stack allows only MIT, Apache-2.0, BSD-2/3, ISC, 0BSD, Unlicense, and CC0; mixed/dual/custom/missing licenses fail; NVIDIA CC-BY-4.0 remains Python-only; every deferred file form receives a stable rejection reason.

- [ ] **Step 2: Write failing 14-partition registry tests**

Assert exact pinned revision `17cad72c886a2858e08d4c349a00d6466f54df63`, exact `data/<partition>` values, exact grammar/spec hashes, and Python-only OpenCodeInstruct revision `8f3ba5bafe4d6e8db46082cf7ae6741bc370604d`.

- [ ] **Step 3: Run tests and verify failure**

Run: `python -m pytest tests/test_corpus_ingestion.py tests/test_multilingual_ingestion.py -q`

Expected: FAIL until the v2 source envelope exists.

- [ ] **Step 4: Implement v2 database and generic ingestion**

Route each admitted row through `BCNLTMultilingualAdapter` using registry metadata. Quarantine parse/coverage failures before curriculum generation. Keep Windows `truststore` injection and streaming limits.

- [ ] **Step 5: Implement the bounded sample command**

Fetch 100 rows per language, report fetched/licensed/parsed/admitted/rejected counts and reasons, and require at least 25 admitted documents per language spanning definitions, calls, references, control flow, and nested scopes.

- [ ] **Step 6: Run ingestion tests**

Run: `python -m pytest tests/test_corpus_ingestion.py tests/test_multilingual_ingestion.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add configs/bcnlt_corpus_sources_v2.json bcnlt/corpus/database.py bcnlt/corpus/filters.py bcnlt/corpus/sources.py scripts/ingest_bcnlt_corpus.py scripts/sample_bcnlt_multilingual_sources.py tests/test_corpus_ingestion.py tests/test_multilingual_ingestion.py
git commit -m "feat: ingest multilingual BCNLT sources"
```

### Task 13: Make curriculum, admission, and packing language-aware

**Files:**
- Modify: `bcnlt/corpus/curriculum.py`
- Modify: `bcnlt/corpus/tasks/code_to_semantic.py`
- Modify: `bcnlt/corpus/tasks/mixed_instruction.py`
- Modify: `bcnlt/corpus/tasks/semantic_continuation.py`
- Modify: `bcnlt/corpus/tasks/semantic_to_code.py`
- Modify: `bcnlt/corpus/tasks/structural_reasoning.py`
- Create: `bcnlt/corpus/tasks/knowledge_boundary.py`
- Modify: `bcnlt/corpus/admission.py`
- Modify: `bcnlt/corpus/packing.py`
- Modify: `scripts/generate_bcnlt_curriculum.py`
- Modify: `scripts/validate_bcnlt_corpus.py`
- Modify: `scripts/export_bcnlt_training_shards.py`
- Modify: `tests/test_curriculum_generation.py`
- Modify: `tests/test_corpus_admission.py`
- Modify: `tests/test_corpus_packing.py`

**Interfaces:**
- Adds `language: str` and `dialect: str | None` to `CurriculumRecord`.
- Adds per-language distributions and failures to `CorpusValidationReport`.
- `validate_corpus(...)` consumes the single carousel and v2 source records; `pack_training_shards(...)` preserves one language per sequence and writes language in provenance indexes.

- [ ] **Step 1: Write failing language-conditioning tests**

For all six families, assert the prompt begins with `Language: <canonical-id>`, uses ordinary base-token payload, and retains the document language through record ID, split, rejection, and shard provenance.

- [ ] **Step 2: Write failing UYR/wildcard curriculum tests**

Assert knowledge-boundary records preserve initial/final score bands, state, evidence signals, UYR retrievals/rounds, wildcard hypotheses, and an explicit unresolved target. Resolvable examples must select supporting evidence; unresolvable examples must reject fabricated certainty.

- [ ] **Step 3: Write failing balance and split tests**

Using small configurable thresholds, assert failures for a missing language, fewer than the per-language minimum, a language above 12% records/tokens, common-core coverage below 80%, unknown events above 7.5%, a missing family, or a family above 40%. Assert duplicate clusters never cross repository/language splits.

- [ ] **Step 4: Write failing one-language packing tests**

Assert a record cannot cite documents from different languages, packer output never concatenates languages, and resume fails after any parser/spec/registry digest changes.

- [ ] **Step 5: Run tests and verify failure**

Run: `python -m pytest tests/test_curriculum_generation.py tests/test_corpus_admission.py tests/test_corpus_packing.py -q`

Expected: FAIL on the new multilingual assertions.

- [ ] **Step 6: Generalize generators and gates**

Replace Python AST normalization and control-flow inspection with carousel semantic projections. Enforce 10,000 packed records per language, 140,000 total, 50,000,000 tokens, six families per language, Common Coding Core coverage, knowledge-boundary distributions, and the global document/template caps.

- [ ] **Step 7: Run focused and complete tests**

Run the focused command, then `python -m pytest -q`; expected PASS.

- [ ] **Step 8: Commit**

```powershell
git add bcnlt/corpus scripts/generate_bcnlt_curriculum.py scripts/validate_bcnlt_corpus.py scripts/export_bcnlt_training_shards.py tests/test_curriculum_generation.py tests/test_corpus_admission.py tests/test_corpus_packing.py
git commit -m "feat: gate balanced multilingual BCNLT corpus"
```

### Task 14: Freeze per-language semantic evaluation and promotion math

**Files:**
- Create: `bcnlt/multilingual_eval.py`
- Create: `configs/bcnlt_multilingual_eval_v1.json`
- Create: `scripts/run_bcnlt_multilingual_eval.py`
- Create: `tests/test_multilingual_eval.py`
- Modify: `bcnlt/tokenizer_eval.py`

**Interfaces:**
- Produces: `SemanticProjection`, `MultilingualEvaluationCase`, `LanguageEvaluation`, `MultilingualPromotionDecision`, `score_semantic_projection(...)`, and `evaluate_multilingual_promotion(...)`.
- Scores six equal categories per language: event classification, definition/reference linking, scope pairing, control flow, code-to-semantic, and semantic-to-code.

- [ ] **Step 1: Write failing exact-scoring tests**

Assert each category contributes one sixth, every category requires at least 80%, language macro requires 90%, and uncertain/null links pass only when the golden target is uncertain/null.

- [ ] **Step 2: Write failing semantic-to-code equivalence tests**

Assert formatting/comments/span changes pass when re-extracted ordered events, values, scope topology, links, AST relations, IR, and uncertainty match; parseable semantic changes fail.

- [ ] **Step 3: Write failing promotion tests**

Assert one failed language blocks the single multilingual adapter, macro averaging is unweighted by corpus size, runtime failures block promotion, knowledge-state accuracy below 90%, correct UYR resolution below 85%, false certainty above 5%, or raw-code/ordinary-instruction regression above two absolute points fails.

- [ ] **Step 4: Run tests and verify failure**

Run: `python -m pytest tests/test_multilingual_eval.py tests/test_tokenizer_eval.py -q`

Expected: FAIL because the multilingual evaluator does not exist.

- [ ] **Step 5: Implement the frozen evaluator**

Implement the exact category math and semantic-projection comparator while keeping existing Python-only evaluator compatibility.

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_multilingual_eval.py tests/test_tokenizer_eval.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add bcnlt/multilingual_eval.py bcnlt/tokenizer_eval.py configs/bcnlt_multilingual_eval_v1.json scripts/run_bcnlt_multilingual_eval.py tests/test_multilingual_eval.py
git commit -m "feat: evaluate multilingual BCNLT semantics"
```

### Task 15: Connect every existing BCNLT subsystem to the carousel

**Files:**
- Modify: `bcnlt/cache.py`
- Modify: `bcnlt/incremental.py`
- Modify: `bcnlt/semantic_db.py`
- Modify: `bcnlt/preprocessing.py`
- Modify: `bcnlt/benchmark.py`
- Modify: `bcnlt/meaning_eval.py`
- Modify: `bcnlt/model_eval.py`
- Modify: `bcnlt/cli.py`
- Modify: `bcnlt/tokenizer_cli.py`
- Modify: `scripts/build_semantic_pair_database.py`
- Modify: `scripts/import_bcnlt_observations.py`
- Modify: `scripts/run_bcnlt_head_to_head_mk2.py`
- Create: `docs/bcnlt-multilingual-component-matrix.md`
- Modify: `tests/test_cache_incremental.py`
- Modify: `tests/test_uncertainty_uyr_wildcard.py`
- Modify: `tests/test_semantic_db.py`
- Modify: `tests/test_cli_and_benchmark.py`
- Modify: `tests/test_meaning_eval.py`
- Modify: `tests/test_model_eval.py`
- Create: `tests/test_multilingual_subsystem_compatibility.py`

**Interfaces:**
- Consumes: the one `BCNLTMultilingualAdapter`, language-aware `SourceDocumentRecord`, canonical records, and semantic-spec/parser identities from Tasks 2-14.
- Produces: language-aware cache keys, incremental updates, uncertainty/UYR/wildcard analysis, semantic-pair persistence, preprocessing, benchmarks, model evaluation, head-to-head reports, and CLI entry points without duplicating parser selection.

- [ ] **Step 1: Write a failing subsystem inventory test**

Parameterize all 14 golden fixtures through cache/incremental parsing, uncertainty scoring, UYR/wildcard analysis, semantic-pair storage/readback, preprocessing, and benchmark serialization. Assert every artifact retains `language`, `parser_bundle_sha256`, and `semantic_spec_sha256` where persistence occurs.

- [ ] **Step 2: Write failing stale-cache and database tests**

Assert cache keys change when language, dialect, parser bundle, or semantic-spec hash changes; incremental edits preserve the selected cartridge; semantic-pair rows from different languages never collide; and existing Python rows remain readable through an explicit v1 compatibility path.

- [ ] **Step 3: Write failing CLI and evaluator routing tests**

Assert parse/preprocess/benchmark commands require `--language`, reject content guessing, and route through the single facade. Assert model/meaning/head-to-head evaluators preserve their existing transport behavior while recording language and adapter lineage.

- [ ] **Step 4: Run tests and verify failure**

Run: `python -m pytest tests/test_cache_incremental.py tests/test_uncertainty_uyr_wildcard.py tests/test_semantic_db.py tests/test_cli_and_benchmark.py tests/test_meaning_eval.py tests/test_model_eval.py tests/test_multilingual_subsystem_compatibility.py -q`

Expected: FAIL wherever an existing subsystem still constructs `PythonAdapter` directly or omits multilingual identity.

- [ ] **Step 5: Replace direct Python wiring with the facade**

Pass explicit language metadata through existing interfaces, extend persistent schemas/identities without changing event or tokenizer IDs, and keep the OpenAI-compatible/Bifrost transport contract unchanged. Do not create a second parser registry in any subsystem.

- [ ] **Step 6: Write the component coverage matrix**

List every tracked `bcnlt/*.py` and integration module as `carousel-connected`, `language-neutral unchanged`, or `generated-artifact only`, with the exact test that protects it. The tokenizer, manifest, canonical, events, IR, Hugging Face wrapper, embedding extension, and merge path must be explicitly listed even when their implementation is unchanged.

- [ ] **Step 7: Run subsystem regressions**

Run the focused command; expected PASS with all-language facade coverage.

- [ ] **Step 8: Run the complete regression suite**

Run: `python -m pytest -q`

Expected: PASS, including tokenizer, manifest, canonical, semantic database, uncertainty, model transport, embedding, and merge tests.

- [ ] **Step 9: Commit**

```powershell
git add bcnlt/cache.py bcnlt/incremental.py bcnlt/semantic_db.py bcnlt/preprocessing.py bcnlt/benchmark.py bcnlt/meaning_eval.py bcnlt/model_eval.py bcnlt/cli.py bcnlt/tokenizer_cli.py scripts/build_semantic_pair_database.py scripts/import_bcnlt_observations.py scripts/run_bcnlt_head_to_head_mk2.py docs/bcnlt-multilingual-component-matrix.md tests/test_cache_incremental.py tests/test_uncertainty_uyr_wildcard.py tests/test_semantic_db.py tests/test_cli_and_benchmark.py tests/test_meaning_eval.py tests/test_model_eval.py tests/test_multilingual_subsystem_compatibility.py
git commit -m "feat: connect BCNLT subsystems to the carousel"
```

### Task 16: Prove the single multilingual adapter in smoke and guard the full run

**Files:**
- Create: `configs/runeforge_mk2_bcnlt_multilingual_smoke_v1.json`
- Create: `configs/runeforge_mk2_bcnlt_multilingual_full_v1.json`
- Modify: `integrations/runeforge_mk2_tokenizer/train_bcnlt_adapter.py`
- Create: `bcnlt/multilingual_training.py`
- Create: `scripts/run_bcnlt_multilingual_smoke.ps1`
- Create: `scripts/build_bcnlt_multilingual_corpus.ps1`
- Create: `scripts/launch_bcnlt_multilingual_full_run.ps1`
- Modify: `bcnlt/full_run_gate.py`
- Modify: `docs/bcnlt-runeforge-training.md`
- Modify: `tests/test_bcnlt_training_preflight.py`
- Modify: `tests/test_full_run_gate.py`
- Create: `tests/test_multilingual_training.py`

**Interfaces:**
- Produces: `MultilingualTrainingConfig`, `LanguageBalancedSampler`, `SmokeMetrics`, `evaluate_smoke_gate(...)`, and full-run handoff schema v2.
- Trains one LoRA plus one sparse BCNLT embedding/LM-head extension over the combined corpus.

- [ ] **Step 1: Write failing single-adapter and sampler tests**

Assert exactly one PEFT adapter is attached, batches draw equal language/family counts for smoke, base embedding/output rows remain frozen, and no per-language LoRA directories are produced.

- [ ] **Step 2: Write failing smoke-metric tests**

Pin seed 1729, microbatch 1, gradient accumulation 8, 128 optimizer steps, 256 validation examples per language, finite loss every step, final 32-step global validation mean at least 2% below the first 32-step mean, and no language final loss above 125% of its initial value.

- [ ] **Step 3: Write failing full-run lineage tests**

Require parser bundle, 14 semantic-spec hashes, source registry, tokenizer, corpus, per-language validation, base identity, adapter, smoke, evaluation, promotion, and training config. Assert any hash change blocks resume/start.

- [ ] **Step 4: Run tests and verify failure**

Run: `python -m pytest tests/test_bcnlt_training_preflight.py tests/test_multilingual_training.py tests/test_full_run_gate.py -q`

Expected: FAIL until multilingual training/gate types exist.

- [ ] **Step 5: Implement the authoritative recipe**

Use LoRA rank 16, alpha 32, dropout 0, learning rate `2e-4`, target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, 4,096-token sequences, AdamW, one full epoch, 3% linear warmup, cosine decay, and validation/checkpoints every 500 optimizer steps.

- [ ] **Step 6: Implement guarded orchestration**

The corpus script runs Windows ingestion/sample validation and WSL packing/training from content-addressed paths. The launcher defaults to validation-only and may start only when all 14 language gates and smoke promotion pass.

- [ ] **Step 7: Run unit and complete regressions**

Run the focused tests, then `python -m pytest -q`; expected PASS.

- [ ] **Step 8: Run the bounded multilingual source sample**

Run the 100-row-per-language command. Expected: all 14 language reports pass the minimum 25 admitted-document and required-event gates.

- [ ] **Step 9: Run the bounded training smoke**

Only after Step 8 passes, run the one-step VRAM check and 128-step smoke. Expected: every numerical smoke gate passes. Do not start the full 140,000-record training run in this task.

- [ ] **Step 10: Commit code and documentation, excluding generated data/weights**

```powershell
git add configs/runeforge_mk2_bcnlt_multilingual_smoke_v1.json configs/runeforge_mk2_bcnlt_multilingual_full_v1.json integrations/runeforge_mk2_tokenizer/train_bcnlt_adapter.py bcnlt/multilingual_training.py bcnlt/full_run_gate.py scripts/run_bcnlt_multilingual_smoke.ps1 scripts/build_bcnlt_multilingual_corpus.ps1 scripts/launch_bcnlt_multilingual_full_run.ps1 docs/bcnlt-runeforge-training.md tests/test_bcnlt_training_preflight.py tests/test_multilingual_training.py tests/test_full_run_gate.py
git commit -m "feat: prepare one multilingual Runeforge adapter"
```

### Task 17: Build the admitted corpus and run the full multilingual adaptation

**Files:**
- Create: `docs/runs/bcnlt-multilingual-v1-report.md`
- Generated outside Git: full corpus database, packed shards, parser/source/corpus manifests, checkpoints, standalone adapter, training logs, evaluations, and promotion decision

**Interfaces:**
- Consumes: the reviewed Task 16 branch, passing 14-language source sample, passing 128-step smoke promotion, preserved Runeforge Mk2 identity, and guarded handoff schema v2.
- Produces: the complete balanced corpus, one full multilingual BCNLT adapter, per-language and aggregate evaluation evidence, and a reproducible run report. The preserved base and smoke/full standalone adapters remain unchanged.

- [ ] **Step 1: Complete the whole-branch review gate**

Require the complete test suite, component matrix, parser-bundle offline verification, source sample, and smoke promotion to pass with no unresolved high-impact review finding.

- [ ] **Step 2: Build and freeze the full admitted corpus**

Run `scripts/build_bcnlt_multilingual_corpus.ps1`. Expected: at least 10,000 packed records per language, at least 140,000 total records, at least 50,000,000 tokens, no language over 12%, all six families present, Common Coding Core and knowledge-boundary gates passed, all holdouts isolated, and every manifest hash fixed.

- [ ] **Step 3: Validate the exact full-run proposal**

Run `scripts/launch_bcnlt_multilingual_full_run.ps1` without its start switch. Expected: it prints the one-epoch step/token budget, disk estimate, checkpoint cadence, command, and all lineage hashes while reporting `started: false`.

- [ ] **Step 4: Launch the one-epoch WSL training run**

Run the same launcher with its explicit start switch. Expected: one Runeforge LoRA plus sparse BCNLT embedding/LM-head extension trains with finite loss; checkpoints are written every 500 optimizer steps and at epoch end.

- [ ] **Step 5: Resume only from an exact checkpoint if interrupted**

When needed, pass the checkpoint path and SHA-256 through the guarded launcher. Expected: exact-lineage resumes proceed; altered or partial checkpoints fail before loading optimizer state.

- [ ] **Step 6: Run frozen full evaluation**

Evaluate preserved Runeforge Mk2, the smoke adapter, and the full adapter with identical prompts, seed, decoding, quantization, and hardware. Expected: every language has at least 80% in each structural category, at least 90% macro structural accuracy, at least 90% knowledge-state accuracy, at least 85% correct UYR resolution, at most 5% false certainty, and control regressions within two absolute points.

- [ ] **Step 7: Preserve artifacts and write the run report**

Record corpus/parser/tokenizer/base/config/adapter hashes, per-language counts and metrics, rejected-data totals, training loss/checkpoints, exact commands, runtime, and promotion decision. Preserve the full standalone adapter even if later merge work is authorized.

- [ ] **Step 8: Commit evidence only**

```powershell
git add docs/runs/bcnlt-multilingual-v1-report.md
git commit -m "docs: record multilingual BCNLT full run"
```

## Final verification and handoff

- [ ] Run `python -m pytest -q` and record the exact pass count and duration.
- [ ] Verify `git diff --check` and confirm generated parser caches, datasets, logs, checkpoints, and weights are not tracked.
- [ ] Re-run parser-bundle verification on Windows and WSL without network access.
- [ ] Run the bounded 100-row sample for all 14 languages and archive its per-language report.
- [ ] Run the one-step and 128-step multilingual smoke only after the sample report passes.
- [ ] Run the frozen multilingual evaluation against preserved Runeforge Mk2 and the smoke adapter under identical settings.
- [ ] Produce the guarded full-run proposal with exact step count, token budget, disk estimate, and manifest hashes.
- [ ] Complete the whole-branch review before the full corpus build/training phase.
- [ ] Preserve the full standalone multilingual adapter and its promotion evidence after Task 17.
