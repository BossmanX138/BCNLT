from pathlib import Path


def test_readme_documents_parallel_compute_strategy():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Parallel Compute Strategy" in readme
    assert "process pool" in readme
    assert "GPU-ready batching" in readme
    assert "deterministic merge" in readme


def test_readme_documents_standard_tokenizer_showdown():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "BCNLT vs Standard Tokenizer Report" in readme
    assert "--compare-report" in readme
    assert "--trace-comparison" in readme


def test_readme_documents_meaning_eval():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Meaning Retrieval Evaluation" in readme
    assert "--meaning-report" in readme


def test_readme_documents_model_context_eval():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Model Context Answer Evaluation" in readme
    assert "--model-eval-report" in readme
    assert "--model-eval-provider openai" in readme
    assert "--model-eval-provider ollama" in readme


def test_readme_documents_bcnlt_tokenizer_v1_contract():
    readme = " ".join(Path("README.md").read_text(encoding="utf-8").split())

    assert "bcnlt-tokenizer-v1" in readme
    assert "text, raw_code, and semantic_code" in readme
    assert "base token IDs remain unchanged" in readme
    assert "semantic records round-trip exactly" in readme
    assert "Python source does not round-trip" in readme
    assert "bcnlt.tokenizer_cli verify" in readme
    assert "docs/bcnlt-tokenizer-v1.md" in readme
