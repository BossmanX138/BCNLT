import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from bcnlt import payload_eval
from bcnlt.adapters.python import PythonAdapter
from bcnlt.core import ContextualUnit
from bcnlt.payload_eval import AUGMENTATION_SEPARATOR, evaluate_payload_source, serialize_units


class RecordingEncoding:
    def __init__(self):
        self.inputs = []

    def encode(self, text, *, disallowed_special):
        assert disallowed_special == ()
        self.inputs.append(text)
        return text.encode("utf-8")


@pytest.fixture
def encoding(monkeypatch):
    encoder = RecordingEncoding()
    monkeypatch.setattr(payload_eval, "_load_encoding", lambda name: encoder)
    return encoder


def test_counts_full_serialized_payload_with_one_encoder(encoding):
    source = "x = 1\nprint(x)\n"
    report = evaluate_payload_source(source, path="fixture.py").to_dict()
    assert len(encoding.inputs) == 3
    raw, semantic, augmented = encoding.inputs
    assert raw == source
    assert augmented == raw + AUGMENTATION_SEPARATOR + semantic
    rows = [json.loads(line) for line in semantic.splitlines()]
    assert len(rows) == report["event_count"]
    assert {"surface", "event", "scope", "span", "symbol_id", "ir", "metadata"} <= rows[0].keys()
    for key, text in zip(("raw_source", "semantic_only", "augmented_source"), encoding.inputs):
        size = report[key]
        assert size["units"] == len(text.encode("utf-8"))
        assert size["utf8_bytes"] == len(text.encode("utf-8"))
        assert size["characters"] == len(text)
        assert size["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert report["semantic_only"]["units"] > report["event_count"]
    assert report["model_token_counts_available"] is True
    assert report["promotion_eligible"] is False


def test_augmented_payload_is_encoded_not_added(monkeypatch):
    class BoundaryEncoding:
        def encode(self, text, **kwargs):
            return [0] * (7 if AUGMENTATION_SEPARATOR in text else 3)

    monkeypatch.setattr(payload_eval, "_load_encoding", lambda name: BoundaryEncoding())
    report = evaluate_payload_source("x = 1\n").to_dict()
    assert report["raw_source"]["units"] == 3
    assert report["semantic_only"]["units"] == 3
    assert report["augmented_source"]["units"] == 7
    assert report["augmentation_overhead_units"] == 4


def test_explicit_encoding_used_once(monkeypatch):
    names = []

    def load(name):
        names.append(name)
        return RecordingEncoding()

    monkeypatch.setattr(payload_eval, "_load_encoding", load)
    result = evaluate_payload_source("x = 1\n", encoding_name="o200k_base")
    assert names == ["o200k_base"]
    assert result.tokenizer_name == "tiktoken:o200k_base"


@pytest.mark.parametrize("failure_at", [0, 1, 2, 3])
def test_encoder_failure_recounts_all_payloads_as_estimates(monkeypatch, failure_at):
    class BrokenEncoding:
        calls = 0

        def encode(self, text, **kwargs):
            self.calls += 1
            if self.calls == failure_at:
                raise RuntimeError("cannot encode")
            return [0] * 99999

    def load(name):
        if failure_at == 0:
            raise ImportError("optional tokenizer missing")
        return BrokenEncoding()

    monkeypatch.setattr(payload_eval, "_load_encoding", load)
    source = "x = 1\n"
    adapter = PythonAdapter()
    units = list(adapter.to_units(adapter.parse(source)))
    semantic = serialize_units(units)
    result = evaluate_payload_source(source).to_dict()
    assert result["tokenizer_name"] == "regex-piece-estimate"
    assert result["model_token_counts_available"] is False
    assert result["fallback_reason"] == ("ImportError" if failure_at == 0 else "RuntimeError")
    for key, text in zip(("raw_source", "semantic_only", "augmented_source"),
                         (source, semantic, source + AUGMENTATION_SEPARATOR + semantic)):
        assert result[key]["units"] == len(re.findall(r"\w+|[^\w\s]|\s+", text))


def test_unicode_special_token_spelling_and_crlf(encoding):
    source = 'message = "café 🧪 <|endoftext|>"\r\n'
    report = evaluate_payload_source(source)
    assert report.raw_source.utf8_bytes > report.raw_source.characters
    assert encoding.inputs[0] == source
    assert "café 🧪" in encoding.inputs[1]
    assert "<|endoftext|>" in encoding.inputs[1]


def test_empty_input_has_no_division_by_zero(encoding):
    result = evaluate_payload_source("").to_dict()
    assert result["raw_source"]["units"] == 0
    assert result["semantic_to_raw_ratio"] is None
    assert result["augmented_to_raw_ratio"] is None
    assert "NaN" not in json.dumps(result, allow_nan=False)


def test_empty_serialization_does_not_add_separator():
    assert serialize_units([]) == ""


def test_serialization_is_deterministic_and_preserves_metadata():
    first = ContextualUnit("x", "DEFINE", "module", (1, 0, 1, 1), metadata={"b": 2, "a": 1})
    second = ContextualUnit("x", "DEFINE", "module", (1, 0, 1, 1), metadata={"a": 1, "b": 2})
    assert serialize_units([first]) == serialize_units([second])
    assert json.loads(serialize_units([first]))["metadata"] == {"a": 1, "b": 2}


def test_invalid_metadata_not_silently_dropped():
    unit = ContextualUnit("x", "DEFINE", "module", (1, 0, 1, 1), metadata={"score": float("nan")})
    with pytest.raises(ValueError):
        serialize_units([unit])


def test_syntax_errors_are_not_reported_as_tokenizer_fallback(encoding):
    with pytest.raises(SyntaxError):
        evaluate_payload_source("def broken(:\n")


def test_cli_writes_payload_cost_report(tmp_path: Path):
    source = tmp_path / "sample.py"
    source.write_text("x = 1\nprint(x)\n", encoding="utf-8")
    output = tmp_path / "payload.json"
    result = subprocess.run(
        [sys.executable, "-m", "bcnlt.cli", str(source), "--payload-report", str(output)],
        text=True, capture_output=True, check=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert "BCNLT payload report" in result.stdout
    assert report["schema_version"] == "bcnlt-payload-evaluation-v1"
    assert report["semantic_only"]["units"] > report["event_count"]
    assert report["augmentation_overhead_units"] > 0


def test_cli_preserves_crlf_but_omits_utf8_bom(tmp_path: Path):
    source = tmp_path / "sample.py"
    text = "x = 1\r\nprint(x)\r\n"
    source.write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    output = tmp_path / "payload.json"
    subprocess.run(
        [sys.executable, "-m", "bcnlt.cli", str(source), "--payload-report", str(output)],
        text=True, capture_output=True, check=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["raw_source"]["characters"] == len(text)
    assert report["raw_source"]["sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_real_encoder_counts_literal_special_token_as_text():
    tiktoken = pytest.importorskip("tiktoken")
    # No dependency on a downloaded encoding in the unit test.
    encoder = tiktoken.Encoding(
        name="test-byte-encoder", pat_str=r"[\s\S]",
        mergeable_ranks={bytes([index]): index for index in range(256)},
        special_tokens={"<|endoftext|>": 256},
    )
    text = 'message = "<|endoftext|>"\n'
    assert len(encoder.encode(text, disallowed_special=())) == len(text)


def test_legacy_benchmark_labels_regex_as_estimate(monkeypatch):
    from bcnlt.benchmark import standard_token_count

    monkeypatch.setitem(sys.modules, "tiktoken", None)
    result = standard_token_count("x = 1\n")
    assert result.tokenizer_name == "regex-piece-estimate"


def test_cli_rejects_directory_before_writing_partial_reports(tmp_path: Path):
    output = tmp_path / "payload.json"
    comparison = tmp_path / "comparison.json"
    result = subprocess.run(
        [sys.executable, "-m", "bcnlt.cli", str(tmp_path), "--payload-report", str(output),
         "--compare-report", str(comparison)], text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert "single Python file" in result.stderr
    assert not output.exists()
    assert not comparison.exists()


def test_readme_explains_payload_measurement_limits():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "--payload-report" in readme
    assert "Semantic event counts are not model token counts" in readme
