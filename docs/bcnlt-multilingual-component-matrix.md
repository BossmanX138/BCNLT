# BCNLT multilingual component matrix

This inventory covers every tracked Python module under `bcnlt/`, every Runeforge tokenizer integration module, and the Task 15 command-line integration scripts. `carousel-connected` means the component either selects the one `BCNLTMultilingualAdapter` or carries its explicit language and parser/spec lineage. `language-neutral unchanged` means its contract is shared across languages and contains no parser selection. Generated data is listed separately and is never executable parser wiring.

| Module | Status | Protecting regression |
|---|---|---|
| `bcnlt/__init__.py` | language-neutral unchanged | `tests/test_readme_architecture.py` |
| `bcnlt/adapters/__init__.py` | carousel-connected | `tests/test_adapter_carousel.py` |
| `bcnlt/adapters/base.py` | language-neutral unchanged | `tests/test_python_adapter.py` |
| `bcnlt/adapters/carousel.py` | carousel-connected | `tests/test_adapter_carousel.py` |
| `bcnlt/adapters/coverage.py` | carousel-connected | `tests/test_tree_sitter_adapter.py` |
| `bcnlt/adapters/python.py` | carousel-connected | `tests/test_python_adapter.py` |
| `bcnlt/adapters/spec.py` | carousel-connected | `tests/test_common_core_registry.py` |
| `bcnlt/adapters/symbols.py` | carousel-connected | `tests/test_common_core_language_projection.py` |
| `bcnlt/adapters/tree_sitter.py` | carousel-connected | `tests/test_multilingual_subsystem_compatibility.py` |
| `bcnlt/adaptive_vocab/__init__.py` | language-neutral unchanged | `tests/test_adaptive_vocab_config.py` |
| `bcnlt/adaptive_vocab/boundary_entropy.py` | language-neutral unchanged | `tests/test_boundary_entropy.py` |
| `bcnlt/adaptive_vocab/candidates.py` | language-neutral unchanged | `tests/test_adaptive_vocab_candidates.py` |
| `bcnlt/adaptive_vocab/codec.py` | language-neutral unchanged | `tests/test_adaptive_vocab_codec.py` |
| `bcnlt/adaptive_vocab/config.py` | language-neutral unchanged | `tests/test_adaptive_vocab_config.py` |
| `bcnlt/adaptive_vocab/corpus_audit.py` | language-neutral unchanged | `tests/test_adaptive_vocab_corpus_audit.py` |
| `bcnlt/adaptive_vocab/hf_artifact.py` | language-neutral unchanged | `tests/test_adaptive_vocab_tokenizer.py` |
| `bcnlt/adaptive_vocab/lineage.py` | language-neutral unchanged | `tests/test_adaptive_vocab_lineage.py` |
| `bcnlt/adaptive_vocab/manifest.py` | language-neutral unchanged | `tests/test_adaptive_vocab_manifest.py` |
| `bcnlt/adaptive_vocab/model_eval.py` | language-neutral unchanged | `tests/test_adaptive_vocab_model_eval.py` |
| `bcnlt/adaptive_vocab/runner.py` | language-neutral unchanged | `tests/test_adaptive_vocab_runner.py` |
| `bcnlt/adaptive_vocab/secrets.py` | language-neutral unchanged | `tests/test_adaptive_vocab_secrets.py` |
| `bcnlt/adaptive_vocab/tokenizer.py` | language-neutral unchanged | `tests/test_adaptive_vocab_tokenizer.py` |
| `bcnlt/adaptive_vocab/tokenizer_eval.py` | language-neutral unchanged | `tests/test_adaptive_vocab_tokenizer_eval.py` |
| `bcnlt/adaptive_vocab/training.py` | language-neutral unchanged | `tests/test_adaptive_vocab_training.py` |
| `bcnlt/benchmark.py` | carousel-connected | `tests/test_cli_and_benchmark.py` |
| `bcnlt/cache.py` | carousel-connected | `tests/test_cache_incremental.py` |
| `bcnlt/canonical.py` | language-neutral unchanged | `tests/test_semantic_record.py` |
| `bcnlt/cli.py` | carousel-connected | `tests/test_cli_and_benchmark.py` |
| `bcnlt/common_core_registry.py` | carousel-connected | `tests/test_common_core_registry.py` |
| `bcnlt/core.py` | carousel-connected | `tests/test_multilingual_golden.py` |
| `bcnlt/corpus/__init__.py` | carousel-connected | `tests/test_corpus_ingestion.py` |
| `bcnlt/corpus/admission.py` | carousel-connected | `tests/test_corpus_admission.py` |
| `bcnlt/corpus/curriculum.py` | carousel-connected | `tests/test_curriculum_generation.py` |
| `bcnlt/corpus/database.py` | carousel-connected | `tests/test_corpus_ingestion.py` |
| `bcnlt/corpus/filters.py` | carousel-connected | `tests/test_corpus_admission.py` |
| `bcnlt/corpus/packing.py` | carousel-connected | `tests/test_corpus_packing.py` |
| `bcnlt/corpus/sources.py` | carousel-connected | `tests/test_multilingual_ingestion.py` |
| `bcnlt/corpus/tasks/__init__.py` | carousel-connected | `tests/test_curriculum_generation.py` |
| `bcnlt/corpus/tasks/code_to_semantic.py` | carousel-connected | `tests/test_curriculum_generation.py` |
| `bcnlt/corpus/tasks/knowledge_boundary.py` | carousel-connected | `tests/test_curriculum_generation.py` |
| `bcnlt/corpus/tasks/mixed_instruction.py` | carousel-connected | `tests/test_curriculum_generation.py` |
| `bcnlt/corpus/tasks/semantic_continuation.py` | carousel-connected | `tests/test_curriculum_generation.py` |
| `bcnlt/corpus/tasks/semantic_to_code.py` | carousel-connected | `tests/test_curriculum_generation.py` |
| `bcnlt/corpus/tasks/structural_reasoning.py` | carousel-connected | `tests/test_curriculum_generation.py` |
| `bcnlt/events.py` | language-neutral unchanged | `tests/test_semantic_record.py` |
| `bcnlt/full_run_gate.py` | language-neutral unchanged | `tests/test_full_run_gate.py` |
| `bcnlt/hf_tokenizer.py` | language-neutral unchanged | `tests/test_hf_tokenizer.py` |
| `bcnlt/incremental.py` | carousel-connected | `tests/test_cache_incremental.py` |
| `bcnlt/ir.py` | language-neutral unchanged | `tests/test_common_core_language_projection.py` |
| `bcnlt/languages.py` | carousel-connected | `tests/test_adapter_carousel.py` |
| `bcnlt/meaning_eval.py` | carousel-connected | `tests/test_meaning_eval.py` |
| `bcnlt/model_eval.py` | carousel-connected | `tests/test_model_eval.py` |
| `bcnlt/multilingual_eval.py` | carousel-connected | `tests/test_multilingual_eval.py` |
| `bcnlt/parsers/__init__.py` | carousel-connected | `tests/test_parser_bundle.py` |
| `bcnlt/parsers/bundle.py` | carousel-connected | `tests/test_parser_bundle.py` |
| `bcnlt/preprocessing.py` | carousel-connected | `tests/test_multilingual_subsystem_compatibility.py` |
| `bcnlt/semantic_db.py` | carousel-connected | `tests/test_semantic_db.py` |
| `bcnlt/semantic_record.py` | language-neutral unchanged | `tests/test_semantic_record.py` |
| `bcnlt/semantic_schema.py` | language-neutral unchanged | `tests/test_semantic_record.py` |
| `bcnlt/tokenizer.py` | carousel-connected | `tests/test_tokenizer.py` |
| `bcnlt/tokenizer_base.py` | language-neutral unchanged | `tests/test_tokenizer.py` |
| `bcnlt/tokenizer_cli.py` | carousel-connected | `tests/test_tokenizer_cli.py` |
| `bcnlt/tokenizer_eval.py` | language-neutral unchanged | `tests/test_tokenizer_eval.py` |
| `bcnlt/tokenizer_manifest.py` | language-neutral unchanged | `tests/test_tokenizer_manifest.py` |
| `bcnlt/uncertainty.py` | carousel-connected | `tests/test_uncertainty_uyr_wildcard.py` |
| `bcnlt/uyr.py` | carousel-connected | `tests/test_uncertainty_uyr_wildcard.py` |
| `bcnlt/wildcard.py` | carousel-connected | `tests/test_learned_definition_loop.py` |
| `integrations/runeforge_mk2_tokenizer/__init__.py` | language-neutral unchanged | `tests/test_bcnlt_training_preflight.py` |
| `integrations/runeforge_mk2_tokenizer/embedding_extension.py` | language-neutral unchanged | `tests/test_embedding_extension.py` |
| `integrations/runeforge_mk2_tokenizer/merge_bcnlt_adapter.py` | language-neutral unchanged | `tests/test_embedding_extension.py` |
| `integrations/runeforge_mk2_tokenizer/train_bcnlt_adapter.py` | language-neutral unchanged | `tests/test_bcnlt_training_preflight.py` |

## Command-line integration modules

| Module | Status | Protecting regression |
|---|---|---|
| `scripts/build_semantic_pair_database.py` | carousel-connected | `tests/test_semantic_db.py` |
| `scripts/import_bcnlt_observations.py` | carousel-connected | `tests/test_semantic_db.py` |
| `scripts/run_bcnlt_head_to_head_mk2.py` | carousel-connected | `tests/test_model_eval.py` |

## Generated artifacts

| Artifact | Status | Protecting regression |
|---|---|---|
| `bcnlt/vocab/bcnlt-tokenizer-v1.tokens.json` | generated-artifact only | `tests/test_tokenizer_manifest.py` |
| tokenizer directories and manifests written by `bcnlt/hf_tokenizer.py` | generated-artifact only | `tests/test_hf_tokenizer.py` |
| adaptive-vocabulary manifests and macro-token artifacts | generated-artifact only | `tests/test_adaptive_vocab_manifest.py`, `tests/test_adaptive_vocab_tokenizer.py` |
| semantic-pair SQLite/JSONL exports | generated-artifact only | `tests/test_semantic_db.py` |

The native tokenizer path is `bcnlt/tokenizer.py` plus the language-neutral manifest, canonical-record, event, IR, and Hugging Face wrapper contracts. The Runeforge embedding extension, adapter trainer, and merge path remain one shared downstream path; Task 16 supplies one balanced multilingual adapter rather than one adapter per language.
