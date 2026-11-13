"""Run the immutable BCNLM handoff assertions against the V10 runtime.

The handoff tests import ``bcnlm.runtime`` from the verified snapshot.  This
pytest plugin replaces only that transport boundary: test fixtures, selection
logic, and assertions remain byte-for-byte the handoff's, while parsing is
performed by the isolated V10 BCNLT source pipeline.
"""
from __future__ import annotations

import json
from typing import Any

from bosscrafts_model.bcnlt_source_pipeline import (
    BCNLTPreprocessingError,
    BCNLTSourcePipeline,
)
from bosscrafts_model.jsonl import split_jsonl_records
from bosscrafts_model.tokenizer import NativeTokenizerBinding


class _PipelineState:
    value: BCNLTSourcePipeline | None = None


_PIPELINE = _PipelineState()


def _pipeline() -> BCNLTSourcePipeline:
    if _PIPELINE.value is None:
        _PIPELINE.value = BCNLTSourcePipeline(NativeTokenizerBinding.load())
    return _PIPELINE.value


def verify_runtime() -> str:
    """Force V10's contained-runtime identity checks and return its identity."""
    return _pipeline().parser_runtime_identity_sha256


def parse_source(
    source: Any,
    *,
    python: str | None = None,
    timeout: float = 30,
    verify: bool = True,
) -> dict[str, object]:
    """Adapt a BCNLM ``Source`` to V10 without weakening either contract."""
    del python, timeout
    pipeline = _pipeline()
    if verify:
        verify_runtime()
    try:
        prepared = pipeline.prepare_semantic(
            source.text,
            language=source.language,
            path=source.path,
        )
    except (BCNLTPreprocessingError, ValueError) as error:
        raise ValueError(f"parser rejected {source.path}: {error}") from error
    if prepared.canonical_jsonl is None:
        raise ValueError(f"parser omitted canonical records for {source.path}")
    records = [
        json.loads(line)
        for line in split_jsonl_records(prepared.canonical_jsonl)
    ]
    return {
        "protocol": "bosscrafts-bcnlt-source-sidecar-v2",
        "language": prepared.language,
        "path": prepared.path,
        "dialect": None,
        "semantic_schema_version": "bcnlt-semantic-record-v3",
        "semantic_spec_sha256": prepared.semantic_spec_sha256,
        "source_sha256": prepared.source_sha256,
        "canonical_sha256": prepared.canonical_sha256,
        "record_count": prepared.record_count,
        "records": records,
    }


def pytest_configure(config: Any) -> None:
    """Install the V10 transport before pytest imports handoff test modules."""
    del config
    from bcnlm import runtime

    runtime.parse_source = parse_source
    runtime.verify_runtime = verify_runtime


def pytest_unconfigure(config: Any) -> None:
    del config
    if _PIPELINE.value is not None:
        _PIPELINE.value.close()
        _PIPELINE.value = None
