"""Content-addressed raw-AST relation and lexical-scope audit evidence."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard, cast

from .bcnlt_source_pipeline import PARSER_RUNTIME_IDENTITY_SHA256
from .current_task_packing import CurrentCorpusTaskPacker
from .raw_ast_witness import RawAstWitnessClient
from .source_semantic_replay import SourceSemanticReplayAuditor
from .tokenizer import NativeTokenizerBinding

_ITEM_SCHEMA = "bosscrafts-raw-ast-semantic-witness-item-v1"
_MANIFEST_SCHEMA = "bosscrafts-raw-ast-semantic-audit-manifest-v1"
_SIDECAR = Path(__file__).with_name("raw_ast_witness_sidecar.py")


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> TypeGuard[str]:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass(frozen=True, slots=True)
class RawAstSemanticAuditResult:
    output_root: Path
    item_count: int
    failed_item_count: int
    complete_source_queue: bool
    training_ready: bool
    witnesses_sha256: str
    manifest_sha256: str


class RawAstSemanticAuditReport:
    """Audit record relation and scope claims against raw Tree-sitter syntax."""

    def __init__(self, binding: NativeTokenizerBinding) -> None:
        if type(binding) is not NativeTokenizerBinding:
            raise TypeError("binding must be a verified NativeTokenizerBinding")
        self._binding = binding

    def materialize(
        self,
        admission_root: str | Path,
        review_root: str | Path,
        output_root: str | Path,
        *,
        review_ids: tuple[str, ...] | None = None,
    ) -> RawAstSemanticAuditResult:
        output = Path(output_root).resolve()
        witnesses_text, manifest = self._evaluate(
            Path(admission_root).resolve(),
            Path(review_root).resolve(),
            review_ids,
        )
        _atomic_write(output / "witnesses.jsonl", witnesses_text.encode("utf-8"))
        _atomic_write(
            output / "manifest.json", f"{_canonical(manifest)}\n".encode("utf-8")
        )
        return self._result(output, manifest)

    def verify(
        self,
        admission_root: str | Path,
        review_root: str | Path,
        output_root: str | Path,
        *,
        review_ids: tuple[str, ...] | None = None,
    ) -> RawAstSemanticAuditResult:
        output = Path(output_root).resolve()
        expected_witnesses, expected_manifest = self._evaluate(
            Path(admission_root).resolve(),
            Path(review_root).resolve(),
            review_ids,
        )
        try:
            actual_witnesses = (output / "witnesses.jsonl").read_text(
                encoding="utf-8"
            )
            actual_manifest = (output / "manifest.json").read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError("raw AST semantic audit evidence is stale or tampered") from exc
        if (
            actual_witnesses != expected_witnesses
            or actual_manifest != f"{_canonical(expected_manifest)}\n"
        ):
            raise ValueError("raw AST semantic audit evidence is stale or tampered")
        return self._result(output, expected_manifest)

    def _evaluate(
        self,
        admission_root: Path,
        review_root: Path,
        review_ids: tuple[str, ...] | None,
    ) -> tuple[str, dict[str, object]]:
        admission_manifest, admission_rows = CurrentCorpusTaskPacker(
            self._binding
        )._load_admission(admission_root)
        review_manifest, review_rows = SourceSemanticReplayAuditor._load_review(
            review_root
        )
        if (
            review_manifest.get("source_admission_manifest_sha256")
            != admission_manifest.get("manifest_sha256")
            or review_manifest.get("binding_sha256") != self._binding.binding_sha256
            or review_manifest.get("tokenizer_sha256") != self._binding.tokenizer_sha256
            or review_manifest.get("parser_runtime_identity_sha256")
            != PARSER_RUNTIME_IDENTITY_SHA256
        ):
            raise ValueError("raw AST audit review/admission lineage mismatch")
        selected = SourceSemanticReplayAuditor._select(review_rows, review_ids)
        admission_by_id = {str(row["document_id"]): row for row in admission_rows}
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in selected:
            document_id = str(row.get("document_id"))
            record = row.get("canonical_record")
            if document_id not in admission_by_id or type(record) is not dict:
                raise ValueError("raw AST audit review item has no admitted source")
            relation = record.get("ast_relation")
            scope = record.get("scope")
            span = record.get("span")
            if (
                type(relation) is not str
                or not relation
                or type(scope) is not str
                or not scope
                or type(span) is not list
            ):
                raise ValueError("raw AST audit record claims are invalid")
            grouped[document_id].append(
                {
                    "query_id": row["review_id"],
                    "span": span,
                    "ast_relation": relation,
                    "scope": scope,
                }
            )

        client = RawAstWitnessClient()
        observed: dict[str, dict[str, object]] = {}
        for document_id in dict.fromkeys(
            str(row["document_id"]) for row in selected
        ):
            admitted = admission_by_id[document_id]
            source = admitted.get("source")
            language = admitted.get("language")
            relative_path = admitted.get("relative_path")
            dialect = admitted.get("dialect")
            if (
                type(source) is not str
                or type(language) is not str
                or type(relative_path) is not str
                or (dialect is not None and type(dialect) is not str)
            ):
                raise ValueError("raw AST audit admitted source metadata is invalid")
            response = client.inspect(
                source,
                language=language,
                path=relative_path,
                dialect=dialect,
                queries=tuple(grouped[document_id]),
            )
            witnesses = response.get("witnesses")
            if type(witnesses) is not list:
                raise ValueError("raw AST witness response is invalid")
            for witness_value in witnesses:
                if type(witness_value) is not dict:
                    raise ValueError("raw AST witness row is invalid")
                witness = cast(dict[str, object], witness_value)
                review_id = witness.get("query_id")
                if type(review_id) is not str or review_id in observed:
                    raise ValueError("raw AST witness identities are invalid")
                observed[review_id] = witness
        if set(observed) != {str(row["review_id"]) for row in selected}:
            raise ValueError("raw AST witness coverage mismatch")

        output_rows: list[dict[str, object]] = []
        counts_by_code: dict[str, int] = {}
        counts_by_language: dict[str, int] = {}
        failed = 0
        for review in selected:
            review_id = str(review["review_id"])
            language = str(review["language"])
            witness = observed[review_id]
            codes: list[str] = []
            if witness["relation_span_match"] is not True:
                codes.append("ast_relation_span_mismatch")
            if witness["scope_match"] is False:
                codes.append("raw_scope_mismatch")
            if codes:
                failed += 1
                counts_by_language[language] = counts_by_language.get(language, 0) + 1
                for code in codes:
                    counts_by_code[code] = counts_by_code.get(code, 0) + 1
            output_rows.append(
                {
                    "schema_version": _ITEM_SCHEMA,
                    "review_id": review_id,
                    "document_id": review["document_id"],
                    "language": language,
                    "span": witness["span"],
                    "ast_relation": witness["ast_relation"],
                    "claimed_scope": witness["claimed_scope"],
                    "derived_scope": witness["derived_scope"],
                    "exact_named_node_types": witness["exact_named_node_types"],
                    "boundary_start_named_node_types": witness[
                        "boundary_start_named_node_types"
                    ],
                    "boundary_end_named_node_types": witness[
                        "boundary_end_named_node_types"
                    ],
                    "relation_span_match": witness["relation_span_match"],
                    "scope_match": witness["scope_match"],
                    "finding_codes": codes,
                    "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
                }
            )
        witnesses_text = "".join(f"{_canonical(row)}\n" for row in output_rows)
        witnesses_sha256 = _sha256_text(witnesses_text)
        complete = len(selected) == len(review_rows)
        passed = failed == 0
        blockers = ["raw AST semantic audit alone does not authorize training"]
        if not complete:
            blockers.insert(0, "raw AST audit covers only an explicit review subset")
        if not passed:
            blockers.insert(0, "raw AST audit found source-claim mismatches")
        unsigned: dict[str, object] = {
            "schema_version": _MANIFEST_SCHEMA,
            "source_admission_manifest_sha256": admission_manifest["manifest_sha256"],
            "source_review_manifest_sha256": review_manifest["manifest_sha256"],
            "source_queue_sha256": review_manifest["queue_sha256"],
            "binding_sha256": self._binding.binding_sha256,
            "tokenizer_sha256": self._binding.tokenizer_sha256,
            "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
            "witness_sidecar_sha256": _sha256_file(_SIDECAR),
            "source_review_item_count": len(review_rows),
            "witness_item_count": len(selected),
            "failed_item_count": failed,
            "counts_by_code": dict(sorted(counts_by_code.items())),
            "counts_by_language": dict(sorted(counts_by_language.items())),
            "complete_source_queue": complete,
            "raw_ast_audit_passed": passed,
            "witnesses_file": "witnesses.jsonl",
            "witnesses_sha256": witnesses_sha256,
            "training_ready": False,
            "training_blockers": blockers,
        }
        return witnesses_text, {
            **unsigned,
            "manifest_sha256": _sha256_text(_canonical(unsigned)),
        }

    @staticmethod
    def _result(
        output: Path, manifest: dict[str, object]
    ) -> RawAstSemanticAuditResult:
        item_count = manifest.get("witness_item_count")
        failed_item_count = manifest.get("failed_item_count")
        complete_source_queue = manifest.get("complete_source_queue")
        training_ready = manifest.get("training_ready")
        witnesses_sha256 = manifest.get("witnesses_sha256")
        manifest_sha256 = manifest.get("manifest_sha256")
        if (
            type(item_count) is not int
            or item_count < 0
            or type(failed_item_count) is not int
            or not 0 <= failed_item_count <= item_count
            or type(complete_source_queue) is not bool
            or type(training_ready) is not bool
            or not _is_sha256(witnesses_sha256)
            or not _is_sha256(manifest_sha256)
        ):
            raise ValueError("raw AST semantic audit result is invalid")
        return RawAstSemanticAuditResult(
            output_root=output,
            item_count=item_count,
            failed_item_count=failed_item_count,
            complete_source_queue=complete_source_queue,
            training_ready=training_ready,
            witnesses_sha256=witnesses_sha256,
            manifest_sha256=manifest_sha256,
        )
