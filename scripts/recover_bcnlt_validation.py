from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict, deque
from dataclasses import replace
from pathlib import Path
from typing import Any

from bcnlt.corpus.admission import (
    AdmissionDecision,
    AdmissionPolicy,
    _decision_id,
    _encoded_length,
    _gate_failures,
    _pair_hash,
    _precheck_candidate,
    _semantic_payload_hash,
)
from bcnlt.corpus.curriculum import CurriculumRecord
from bcnlt.corpus.database import CorpusDatabase
from bcnlt.corpus.streaming_admission import (
    _DiskDigestSet,
    _document_context,
    _hashed_five_grams,
    _is_near_duplicate,
    _pack_hashes,
    _remember_near_duplicate,
    _streaming_distributions,
    _unpack_hashes,
)
from bcnlt.tokenizer import BCNLTTokenizer
from scripts.validate_bcnlt_corpus import admission_policy_for_profile


CHECKPOINT_INTERVAL = 1_000
CHECKPOINT_SCHEMA = "bcnlt-validation-recovery-v1"


def _curriculum_record(line: str, line_number: int) -> CurriculumRecord:
    raw = json.loads(line)
    try:
        return CurriculumRecord(
            record_id=raw["record_id"],
            family=raw["family"],
            template_id=raw["template_id"],
            source_document_ids=tuple(raw["source_document_ids"]),
            prompt_mode=raw["prompt_mode"],
            target_mode=raw["target_mode"],
            prompt=raw["prompt"],
            target=raw["target"],
            provenance=tuple(sorted(raw["provenance"].items())),
            split=raw["split"],
            language=raw.get("language", raw["provenance"]["canonical_language"]),
            dialect=raw.get("dialect"),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Invalid curriculum row {line_number}: {exc}") from exc


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _policy_fingerprint(policy: AdmissionPolicy, candidate_count: int) -> str:
    return hashlib.sha256(
        f"{candidate_count}:{policy!r}".encode("utf-8")
    ).hexdigest()


def _empty_state() -> dict[str, Any]:
    return {
        "accepted_count": 0,
        "decision_count": 0,
        "document_counts": Counter(),
        "template_counts": Counter(),
        "accepted_document_ids": set(),
        "family_counts": Counter(),
        "split_counts": Counter(),
        "records_by_language": Counter(),
        "family_by_language": defaultdict(Counter),
        "encoded_tokens_by_language": Counter(),
        "rejection_reasons": Counter(),
        "duplicate_reasons": Counter(),
        "recovery_boundary": 0,
    }


def _state_payload(
    state: dict[str, Any],
    *,
    phase: str,
    policy_fingerprint: str,
    curriculum_offset: int,
    accepted_offset: int,
    decisions_offset: int,
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "phase": phase,
        "policy_fingerprint": policy_fingerprint,
        "curriculum_offset": curriculum_offset,
        "accepted_offset": accepted_offset,
        "decisions_offset": decisions_offset,
        "accepted_count": state["accepted_count"],
        "decision_count": state["decision_count"],
        "document_counts": dict(state["document_counts"]),
        "template_counts": [
            [family, template, count]
            for (family, template), count in sorted(state["template_counts"].items())
        ],
        "accepted_document_ids": sorted(state["accepted_document_ids"]),
        "family_counts": dict(state["family_counts"]),
        "split_counts": dict(state["split_counts"]),
        "records_by_language": dict(state["records_by_language"]),
        "family_by_language": {
            language: dict(counts)
            for language, counts in state["family_by_language"].items()
        },
        "encoded_tokens_by_language": dict(state["encoded_tokens_by_language"]),
        "rejection_reasons": dict(state["rejection_reasons"]),
        "duplicate_reasons": dict(state["duplicate_reasons"]),
        "recovery_boundary": state["recovery_boundary"],
    }


def _restore_state(payload: dict[str, Any]) -> dict[str, Any]:
    state = _empty_state()
    state["accepted_count"] = int(payload["accepted_count"])
    state["decision_count"] = int(payload["decision_count"])
    state["recovery_boundary"] = int(payload.get("recovery_boundary", 0))
    state["document_counts"].update(payload["document_counts"])
    state["template_counts"].update(
        {
            (family, template): int(count)
            for family, template, count in payload["template_counts"]
        }
    )
    state["accepted_document_ids"].update(payload["accepted_document_ids"])
    for name in (
        "family_counts",
        "split_counts",
        "records_by_language",
        "encoded_tokens_by_language",
        "rejection_reasons",
        "duplicate_reasons",
    ):
        state[name].update(payload[name])
    for language, counts in payload["family_by_language"].items():
        state["family_by_language"][language].update(counts)
    return state


def _open_state_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS seen_digests (
            kind TEXT NOT NULL,
            digest TEXT NOT NULL,
            PRIMARY KEY (kind, digest)
        ) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS accepted_shingles (
            accepted_index INTEGER PRIMARY KEY,
            family TEXT NOT NULL,
            payload BLOB NOT NULL
        );
        CREATE TABLE IF NOT EXISTS recovery_checkpoint (
            checkpoint_id INTEGER PRIMARY KEY CHECK (checkpoint_id = 1),
            payload TEXT NOT NULL
        );
        """
    )
    return connection


def _load_checkpoint(connection: sqlite3.Connection) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT payload FROM recovery_checkpoint WHERE checkpoint_id = 1"
    ).fetchone()
    return json.loads(row[0]) if row else None


def _save_checkpoint(
    connection: sqlite3.Connection,
    state: dict[str, Any],
    *,
    phase: str,
    policy_fingerprint: str,
    curriculum_offset: int,
    accepted_offset: int,
    decisions_offset: int,
) -> dict[str, Any]:
    payload = _state_payload(
        state,
        phase=phase,
        policy_fingerprint=policy_fingerprint,
        curriculum_offset=curriculum_offset,
        accepted_offset=accepted_offset,
        decisions_offset=decisions_offset,
    )
    connection.execute(
        "INSERT OR REPLACE INTO recovery_checkpoint(checkpoint_id, payload) VALUES (1, ?)",
        (json.dumps(payload, ensure_ascii=False, sort_keys=True),),
    )
    connection.commit()
    connection.execute("BEGIN")
    return payload


def _update_accepted_state(
    state: dict[str, Any],
    candidate: CurriculumRecord,
    *,
    split: str,
    encoded_length: int,
) -> None:
    source_ids = candidate.source_document_ids
    state["accepted_count"] += 1
    state["accepted_document_ids"].update(source_ids)
    state["family_counts"][candidate.family] += 1
    state["split_counts"][split] += 1
    state["records_by_language"][candidate.language] += 1
    state["family_by_language"][candidate.language][candidate.family] += 1
    state["encoded_tokens_by_language"][candidate.language] += encoded_length
    for source_id in source_ids:
        state["document_counts"][source_id] += 1
    state["template_counts"][(candidate.family, candidate.template_id)] += 1


def _verify_decision(
    decision: dict[str, Any], candidate: CurriculumRecord, ordinal: int
) -> None:
    required = {
        "decision_id",
        "record_id",
        "source_document_ids",
        "accepted",
        "reason_code",
        "split",
        "language",
    }
    if not required <= decision.keys():
        raise ValueError(f"Decision {ordinal} is missing required fields")
    if decision["record_id"] != candidate.record_id:
        raise ValueError(f"Decision/curriculum record mismatch at ordinal {ordinal}")
    if tuple(decision["source_document_ids"]) != candidate.source_document_ids:
        raise ValueError(f"Decision source mismatch at ordinal {ordinal}")
    if decision["language"] != candidate.language:
        raise ValueError(f"Decision language mismatch at ordinal {ordinal}")
    expected_id = _decision_id(
        candidate.record_id,
        ordinal,
        str(decision["reason_code"]),
        str(decision["split"]),
    )
    if decision["decision_id"] != expected_id:
        raise ValueError(f"Decision ID mismatch at ordinal {ordinal}")


def _rebuild_partial_state(
    connection: sqlite3.Connection,
    state: dict[str, Any],
    checkpoint: dict[str, Any] | None,
    *,
    curriculum_path: Path,
    accepted_path: Path,
    decisions_path: Path,
    tokenizer: BCNLTTokenizer,
    policy_fingerprint: str,
) -> dict[str, Any]:
    seen_record_ids = _DiskDigestSet(connection, "record")
    seen_pairs = _DiskDigestSet(connection, "pair")
    seen_semantic = _DiskDigestSet(connection, "semantic")
    curriculum_offset = int(checkpoint.get("curriculum_offset", 0)) if checkpoint else 0
    accepted_offset = int(checkpoint.get("accepted_offset", 0)) if checkpoint else 0
    decisions_offset = int(checkpoint.get("decisions_offset", 0)) if checkpoint else 0
    with curriculum_path.open("r", encoding="utf-8", newline="") as curriculum_handle, accepted_path.open(
        "r", encoding="utf-8", newline=""
    ) as accepted_handle, decisions_path.open("r", encoding="utf-8", newline="") as decisions_handle:
        curriculum_handle.seek(curriculum_offset)
        accepted_handle.seek(accepted_offset)
        decisions_handle.seek(decisions_offset)
        while True:
            decision_line = decisions_handle.readline()
            if not decision_line:
                break
            candidate_line = curriculum_handle.readline()
            if not candidate_line:
                raise ValueError("Partial decisions extend beyond the curriculum")
            ordinal = state["decision_count"]
            candidate = _curriculum_record(candidate_line, ordinal + 1)
            decision = json.loads(decision_line)
            _verify_decision(decision, candidate, ordinal)
            seen_record_ids.add(candidate.record_id)
            if decision["accepted"]:
                accepted_line = accepted_handle.readline()
                if not accepted_line:
                    raise ValueError(f"Accepted ledger ended at ordinal {ordinal}")
                admitted_raw = json.loads(accepted_line)
                expected = replace(candidate, split=str(decision["split"])).to_dict()
                if admitted_raw != expected:
                    raise ValueError(f"Accepted ledger mismatch at ordinal {ordinal}")
                pair_hash = _pair_hash(candidate)
                semantic_hash = _semantic_payload_hash(candidate)
                semantic_key = (
                    f"{candidate.family}:{semantic_hash}" if semantic_hash else ""
                )
                shingles = _hashed_five_grams(candidate.prompt + "\n" + candidate.target)
                accepted_index = state["accepted_count"]
                seen_pairs.add(pair_hash)
                if semantic_key:
                    seen_semantic.add(semantic_key)
                connection.execute(
                    "INSERT INTO accepted_shingles(accepted_index, family, payload) VALUES (?, ?, ?)",
                    (accepted_index, candidate.family, _pack_hashes(shingles)),
                )
                _update_accepted_state(
                    state,
                    candidate,
                    split=str(decision["split"]),
                    encoded_length=_encoded_length(candidate, tokenizer),
                )
            else:
                reason = str(decision["reason_code"])
                state["rejection_reasons"][reason] += 1
                if "duplicate" in reason:
                    state["duplicate_reasons"][reason] += 1
            state["decision_count"] += 1
            if state["decision_count"] % CHECKPOINT_INTERVAL == 0:
                checkpoint = _save_checkpoint(
                    connection,
                    state,
                    phase="rebuilding",
                    policy_fingerprint=policy_fingerprint,
                    curriculum_offset=curriculum_handle.tell(),
                    accepted_offset=accepted_handle.tell(),
                    decisions_offset=decisions_handle.tell(),
                )
                print(
                    json.dumps(
                        {
                            "recovery_phase": "rebuilding",
                            "decisions_rebuilt": state["decision_count"],
                            "accepted_rebuilt": state["accepted_count"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if accepted_handle.readline():
            raise ValueError("Accepted ledger has rows without matching accepted decisions")
        state["recovery_boundary"] = state["decision_count"]
        checkpoint = _save_checkpoint(
            connection,
            state,
            phase="continuing",
            policy_fingerprint=policy_fingerprint,
            curriculum_offset=curriculum_handle.tell(),
            accepted_offset=accepted_handle.tell(),
            decisions_offset=decisions_handle.tell(),
        )
    return checkpoint


def _rebuild_near_buckets(
    connection: sqlite3.Connection,
) -> dict[str, dict[bytes, deque[int]]]:
    near_buckets: dict[str, dict[bytes, deque[int]]] = defaultdict(dict)
    rows = connection.execute(
        "SELECT accepted_index, family, payload FROM accepted_shingles ORDER BY accepted_index"
    )
    for accepted_index, family, payload in rows:
        _remember_near_duplicate(
            str(family),
            _unpack_hashes(bytes(payload)),
            int(accepted_index),
            near_buckets,
        )
    return near_buckets


def _continue_validation(
    connection: sqlite3.Connection,
    state: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    curriculum_path: Path,
    accepted_path: Path,
    decisions_path: Path,
    candidate_count: int,
    documents: tuple[Any, ...],
    tokenizer: BCNLTTokenizer,
    policy: AdmissionPolicy,
    policy_fingerprint: str,
) -> tuple[Any, dict[str, Any]]:
    document_map = {document.document_id: document for document in documents}
    split_manifest, holdout_documents, summaries = _document_context(
        documents, policy.holdout_hashes
    )
    document_limit = max(1, int(candidate_count * policy.maximum_document_fraction))
    template_limit = max(
        1, int(candidate_count * policy.maximum_template_family_fraction)
    )
    near_buckets = _rebuild_near_buckets(connection)
    seen_record_ids = _DiskDigestSet(connection, "record")
    seen_pairs = _DiskDigestSet(connection, "pair")
    seen_semantic = _DiskDigestSet(connection, "semantic")
    curriculum_offset = int(checkpoint["curriculum_offset"])
    accepted_offset = int(checkpoint["accepted_offset"])
    decisions_offset = int(checkpoint["decisions_offset"])
    with curriculum_path.open("r", encoding="utf-8", newline="") as curriculum_handle, accepted_path.open(
        "r+", encoding="utf-8", newline="\n"
    ) as accepted_handle, decisions_path.open("r+", encoding="utf-8", newline="\n") as decisions_handle:
        curriculum_handle.seek(curriculum_offset)
        accepted_handle.seek(accepted_offset)
        accepted_handle.truncate()
        decisions_handle.seek(decisions_offset)
        decisions_handle.truncate()
        while True:
            candidate_line = curriculum_handle.readline()
            if not candidate_line:
                break
            ordinal = state["decision_count"]
            candidate = _curriculum_record(candidate_line, ordinal + 1)
            source_ids = candidate.source_document_ids
            assigned_splits = {
                split_manifest.get(source_id, "unknown") for source_id in source_ids
            }
            split = next(iter(assigned_splits)) if len(assigned_splits) == 1 else "conflict"
            reason = _precheck_candidate(
                candidate,
                document_map,
                holdout_documents,
                split,
                assigned_splits,
                seen_record_ids,  # type: ignore[arg-type]
            )
            pair_hash = ""
            semantic_key = ""
            shingles: tuple[int, ...] = ()
            encoded_length = 0
            if reason is None:
                pair_hash = _pair_hash(candidate)
                semantic_hash = _semantic_payload_hash(candidate)
                semantic_key = (
                    f"{candidate.family}:{semantic_hash}" if semantic_hash else ""
                )
                shingles = _hashed_five_grams(candidate.prompt + "\n" + candidate.target)
                if pair_hash in seen_pairs:
                    reason = "exact_pair_duplicate"
                elif semantic_key and semantic_key in seen_semantic:
                    reason = "canonical_semantic_duplicate"
                elif _is_near_duplicate(
                    candidate.family,
                    shingles,
                    near_buckets,
                    connection,
                    policy.near_duplicate_threshold,
                ):
                    reason = "near_duplicate_pair"
                elif any(
                    state["document_counts"][source_id] >= document_limit
                    for source_id in source_ids
                ):
                    reason = "document_cap_exceeded"
                elif (
                    state["template_counts"][(candidate.family, candidate.template_id)]
                    >= template_limit
                ):
                    reason = "template_family_cap_exceeded"
                else:
                    encoded_length = _encoded_length(candidate, tokenizer)
                    if encoded_length > policy.maximum_sequence_tokens:
                        reason = "sequence_too_long"
            accepted = reason is None
            reason_code = "accepted" if accepted else str(reason)
            decision = AdmissionDecision(
                decision_id=_decision_id(candidate.record_id, ordinal, reason_code, split),
                record_id=candidate.record_id,
                source_document_ids=source_ids,
                accepted=accepted,
                reason_code=reason_code,
                split=split,
                language=candidate.language,
            )
            decisions_handle.write(
                json.dumps(decision.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
            seen_record_ids.add(candidate.record_id)
            if not accepted:
                state["rejection_reasons"][reason_code] += 1
                if "duplicate" in reason_code:
                    state["duplicate_reasons"][reason_code] += 1
            else:
                admitted = replace(candidate, split=split)
                accepted_handle.write(
                    json.dumps(admitted.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                )
                seen_pairs.add(pair_hash)
                if semantic_key:
                    seen_semantic.add(semantic_key)
                accepted_index = state["accepted_count"]
                connection.execute(
                    "INSERT INTO accepted_shingles(accepted_index, family, payload) VALUES (?, ?, ?)",
                    (accepted_index, candidate.family, _pack_hashes(shingles)),
                )
                _remember_near_duplicate(
                    candidate.family,
                    shingles,
                    accepted_index,
                    near_buckets,
                )
                _update_accepted_state(
                    state,
                    candidate,
                    split=split,
                    encoded_length=encoded_length,
                )
            state["decision_count"] += 1
            if state["decision_count"] % CHECKPOINT_INTERVAL == 0:
                accepted_handle.flush()
                decisions_handle.flush()
                checkpoint = _save_checkpoint(
                    connection,
                    state,
                    phase="continuing",
                    policy_fingerprint=policy_fingerprint,
                    curriculum_offset=curriculum_handle.tell(),
                    accepted_offset=accepted_handle.tell(),
                    decisions_offset=decisions_handle.tell(),
                )
                print(
                    json.dumps(
                        {
                            "recovery_phase": "continuing",
                            "candidates_processed": state["decision_count"],
                            "candidates_total": candidate_count,
                            "accepted": state["accepted_count"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        accepted_handle.flush()
        decisions_handle.flush()
        if state["decision_count"] != candidate_count:
            raise ValueError(
                f"Candidate count mismatch: expected {candidate_count}, got {state['decision_count']}"
            )
        checkpoint = _save_checkpoint(
            connection,
            state,
            phase="complete",
            policy_fingerprint=policy_fingerprint,
            curriculum_offset=curriculum_handle.tell(),
            accepted_offset=accepted_handle.tell(),
            decisions_offset=decisions_handle.tell(),
        )
    encoded_tokens = sum(state["encoded_tokens_by_language"].values())
    distributions = _streaming_distributions(
        accepted_document_ids=state["accepted_document_ids"],
        documents=document_map,
        summaries=summaries,
        accepted_count=state["accepted_count"],
        family_counts=state["family_counts"],
        split_counts=state["split_counts"],
        records_by_language=state["records_by_language"],
        family_by_language=state["family_by_language"],
        encoded_tokens=encoded_tokens,
        encoded_tokens_by_language=state["encoded_tokens_by_language"],
        rejection_reasons=state["rejection_reasons"],
        duplicate_reasons=state["duplicate_reasons"],
        holdout_collisions=len(holdout_documents),
    )
    failures = _gate_failures(
        state["accepted_count"], encoded_tokens, distributions, policy
    )
    result = {
        "schema_version": "bcnlt-streaming-corpus-validation-v1",
        "passed": not failures,
        "failure_codes": list(failures),
        "accepted_record_count": state["accepted_count"],
        "decision_count": state["decision_count"],
        "split_manifest": dict(sorted(split_manifest.items())),
        "distributions": distributions,
    }
    return result, checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Recover and continue an interrupted BCNLT streaming validation"
    )
    parser.add_argument("--database", required=True)
    parser.add_argument("--curriculum", required=True)
    parser.add_argument("--candidate-count", required=True, type=int)
    parser.add_argument("--tokenizer-artifact", required=True)
    parser.add_argument("--base-tokenizer", required=True)
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--state-database", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--recovery-report", required=True)
    parser.add_argument("--interruption-manifest", required=True)
    parser.add_argument("--policy-profile", choices=("full", "smoke"), default="smoke")
    args = parser.parse_args(argv)

    curriculum_path = Path(args.curriculum)
    accepted_path = Path(args.accepted)
    decisions_path = Path(args.decisions)
    state_path = Path(args.state_database)
    report_path = Path(args.report)
    recovery_report_path = Path(args.recovery_report)
    interruption_manifest_path = Path(args.interruption_manifest)
    for path in (curriculum_path, accepted_path, decisions_path):
        if not path.is_file():
            parser.error(f"required artifact does not exist: {path}")
    if not interruption_manifest_path.is_file():
        parser.error(
            f"interruption manifest does not exist: {interruption_manifest_path}"
        )
    if report_path.exists():
        parser.error(f"report already exists: {report_path}")

    policy = admission_policy_for_profile(args.policy_profile)
    fingerprint = _policy_fingerprint(policy, args.candidate_count)
    tokenizer = BCNLTTokenizer.from_pretrained(
        args.tokenizer_artifact, args.base_tokenizer
    )
    connection = _open_state_database(state_path)
    try:
        checkpoint = _load_checkpoint(connection)
        if checkpoint:
            if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
                raise ValueError("Unsupported recovery checkpoint schema")
            if checkpoint.get("policy_fingerprint") != fingerprint:
                raise ValueError("Recovery checkpoint policy or candidate count changed")
            state = _restore_state(checkpoint)
        else:
            state = _empty_state()
            connection.execute("BEGIN")
        if not checkpoint or checkpoint["phase"] == "rebuilding":
            checkpoint = _rebuild_partial_state(
                connection,
                state,
                checkpoint,
                curriculum_path=curriculum_path,
                accepted_path=accepted_path,
                decisions_path=decisions_path,
                tokenizer=tokenizer,
                policy_fingerprint=fingerprint,
            )
        database = CorpusDatabase(args.database)
        documents = tuple(database.documents())
        database.close()
        result, checkpoint = _continue_validation(
            connection,
            state,
            checkpoint,
            curriculum_path=curriculum_path,
            accepted_path=accepted_path,
            decisions_path=decisions_path,
            candidate_count=args.candidate_count,
            documents=documents,
            tokenizer=tokenizer,
            policy=policy,
            policy_fingerprint=fingerprint,
        )
    finally:
        connection.close()

    script_path = Path(__file__).resolve()
    recovery_payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "recovery_script": {"path": str(script_path), "sha256": _sha256(script_path)},
        "state_database": {"path": str(state_path), "sha256": _sha256(state_path)},
        "interruption_manifest": {
            "path": str(interruption_manifest_path),
            "sha256": _sha256(interruption_manifest_path),
        },
        "resumed_after_decision": state["recovery_boundary"],
        "final_decision_count": result["decision_count"],
        "final_accepted_record_count": result["accepted_record_count"],
    }
    recovery_report_path.write_text(
        json.dumps(recovery_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    result["policy_profile"] = args.policy_profile
    result["decisions_artifact"] = {
        "path": str(decisions_path),
        "sha256": _sha256(decisions_path),
    }
    result["accepted_artifact"] = {
        "path": str(accepted_path),
        "sha256": _sha256(accepted_path),
    }
    result["recovery_artifact"] = {
        "path": str(recovery_report_path),
        "sha256": _sha256(recovery_report_path),
    }
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps({"passed": result["passed"], "report": str(report_path)}, sort_keys=True)
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
