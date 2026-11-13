from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SAMPLE_SCHEMA_VERSION = "bcnlt-native-foundation-sample-v1"


@dataclass(slots=True)
class _ByteReservoir:
    budget: int
    retain_payloads: bool = True
    candidates: dict[str, tuple[int, int, bytes | None]] = field(default_factory=dict)
    selected: dict[str, int] = field(default_factory=dict)
    entries: list[tuple[int, str, bytes]] = field(default_factory=list)
    total_bytes: int = 0
    _finalized: bool = False
    _collected: set[str] = field(default_factory=set)

    def consider(self, *, key: str, payload: bytes) -> None:
        if self._finalized:
            raise RuntimeError("cannot add candidates after selection is finalized")
        if not payload or len(payload) > self.budget:
            return
        score = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest(), "big")
        retained = payload if self.retain_payloads else None
        candidate = (score, len(payload), retained)
        existing = self.candidates.get(key)
        if existing is None or candidate[:2] < existing[:2]:
            self.candidates[key] = candidate

    def finalize(self) -> None:
        if self._finalized:
            return
        used = 0
        for key, (score, size, payload) in sorted(
            self.candidates.items(), key=lambda item: (item[1][0], item[0])
        ):
            if size > self.budget - used:
                continue
            self.selected[key] = score
            used += size
            if payload is not None:
                self.entries.append((score, key, payload))
                self._collected.add(key)
        self.total_bytes = used
        self.candidates.clear()
        self._finalized = True

    def collect(self, *, key: str, payload: bytes) -> None:
        self.finalize()
        if key not in self.selected or key in self._collected:
            return
        self.entries.append((self.selected[key], key, payload))
        self._collected.add(key)

    def ordered_payloads(self) -> Iterable[bytes]:
        self.finalize()
        for _, _, payload in sorted(self.entries, key=lambda entry: (entry[0], entry[1])):
            yield payload


def materialize_balanced_sample(
    *,
    accepted_corpus_path: str | Path,
    english_dataset_path: str | Path,
    output_dir: str | Path,
    languages: tuple[str, ...],
    bytes_per_language: int,
    english_bytes: int,
    progress_every: int = 5_000,
) -> dict[str, Any]:
    if not languages or len(set(languages)) != len(languages):
        raise ValueError("languages must be a non-empty unique tuple")
    if bytes_per_language <= 0 or english_bytes <= 0:
        raise ValueError("sample byte budgets must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")

    accepted_path = Path(accepted_corpus_path)
    english_path = Path(english_dataset_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    progress_path = destination / "progress.json"
    if manifest_path.exists():
        raise FileExistsError(f"sample manifest already exists: {manifest_path}")

    reservoirs = {
        language: _ByteReservoir(bytes_per_language, retain_payloads=False)
        for language in languages
    }
    reservoirs["english"] = _ByteReservoir(english_bytes, retain_payloads=False)
    accepted_digest = hashlib.sha256()
    scanned = 0
    eligible = 0
    start = time.monotonic()
    _write_progress(
        progress_path,
        stage="scan_multilingual",
        status="running",
        records_scanned=0,
        eligible_records=0,
        selected_bytes_by_domain={domain: 0 for domain in reservoirs},
        started_at=_now(),
    )

    try:
        with accepted_path.open("rb") as stream:
            for raw_line in stream:
                accepted_digest.update(raw_line)
                scanned += 1
                try:
                    record = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(record, dict) or record.get("split") != "train":
                    continue
                language = record.get("language")
                prompt = record.get("prompt")
                target = record.get("target")
                if language not in reservoirs or language == "english":
                    continue
                if not _nonempty_text(prompt) or not _nonempty_text(target):
                    continue
                eligible += 1
                payload = _record_payload(prompt, target)
                key = _record_key(language, record.get("record_id"), payload)
                reservoirs[language].consider(
                    key=key,
                    payload=payload,
                )
                if scanned % progress_every == 0:
                    elapsed = max(time.monotonic() - start, 1e-9)
                    _write_progress(
                        progress_path,
                        stage="scan_multilingual",
                        status="running",
                        records_scanned=scanned,
                        eligible_records=eligible,
                        elapsed_seconds=round(elapsed, 3),
                        records_per_second=round(scanned / elapsed, 3),
                        candidate_records_by_domain={
                            domain: len(reservoir.candidates)
                            for domain, reservoir in reservoirs.items()
                        },
                    )

        for language in languages:
            reservoirs[language].finalize()
        _write_progress(
            progress_path,
            stage="collect_multilingual",
            status="running",
            records_scanned=0,
            eligible_records=eligible,
            selected_bytes_by_domain={
                domain: reservoir.total_bytes for domain, reservoir in reservoirs.items()
            },
        )
        collected_scan = 0
        with accepted_path.open("rb") as stream:
            for raw_line in stream:
                collected_scan += 1
                try:
                    record = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(record, dict) or record.get("split") != "train":
                    continue
                language = record.get("language")
                prompt = record.get("prompt")
                target = record.get("target")
                if language not in reservoirs or language == "english":
                    continue
                if not _nonempty_text(prompt) or not _nonempty_text(target):
                    continue
                payload = _record_payload(prompt, target)
                key = _record_key(language, record.get("record_id"), payload)
                reservoirs[language].collect(key=key, payload=payload)
                if collected_scan % progress_every == 0:
                    elapsed = max(time.monotonic() - start, 1e-9)
                    _write_progress(
                        progress_path,
                        stage="collect_multilingual",
                        status="running",
                        records_scanned=collected_scan,
                        eligible_records=eligible,
                        elapsed_seconds=round(elapsed, 3),
                        records_per_second=round((scanned + collected_scan) / elapsed, 3),
                        collected_records_by_domain={
                            domain: len(reservoir.entries)
                            for domain, reservoir in reservoirs.items()
                        },
                    )

        _write_progress(
            progress_path,
            stage="scan_english",
            status="running",
            records_scanned=scanned,
            eligible_records=eligible,
            selected_bytes_by_domain={
                domain: reservoir.total_bytes for domain, reservoir in reservoirs.items()
            },
        )
        english_raw = english_path.read_bytes()
        english_digest = hashlib.sha256(english_raw).hexdigest()
        english_records = json.loads(english_raw)
        if not isinstance(english_records, list):
            raise ValueError("English dataset must be a JSON list")
        english_eligible = 0
        for record in english_records:
            if not isinstance(record, dict) or record.get("review_status") != "approved":
                continue
            instruction = record.get("instruction")
            output = record.get("output")
            if not _nonempty_text(instruction) or not _nonempty_text(output):
                continue
            english_eligible += 1
            payload = _record_payload(instruction, output)
            content_key = _record_key("english", None, payload)
            reservoirs["english"].consider(
                key=content_key,
                payload=payload,
            )
        reservoirs["english"].finalize()
        for record in english_records:
            if not isinstance(record, dict) or record.get("review_status") != "approved":
                continue
            instruction = record.get("instruction")
            output = record.get("output")
            if not _nonempty_text(instruction) or not _nonempty_text(output):
                continue
            payload = _record_payload(instruction, output)
            reservoirs["english"].collect(
                key=_record_key("english", None, payload),
                payload=payload,
            )

        output_hash = hashlib.sha256()
        file_hashes: dict[str, str] = {}
        selected_records: dict[str, int] = {}
        selected_bytes: dict[str, int] = {}
        for domain, reservoir in reservoirs.items():
            file_path = destination / f"{domain}.txt"
            temporary = file_path.with_name(f".{file_path.name}.tmp")
            digest = hashlib.sha256()
            count = 0
            with temporary.open("wb") as output_stream:
                for payload in reservoir.ordered_payloads():
                    output_stream.write(payload)
                    digest.update(payload)
                    output_hash.update(domain.encode("utf-8") + b"\0" + payload)
                    count += 1
            temporary.replace(file_path)
            file_hashes[file_path.name] = digest.hexdigest()
            selected_records[domain] = count
            selected_bytes[domain] = file_path.stat().st_size

        elapsed = max(time.monotonic() - start, 1e-9)
        manifest: dict[str, Any] = {
            "schema_version": SAMPLE_SCHEMA_VERSION,
            "status": "complete",
            "created_at": _now(),
            "languages": list(languages),
            "bytes_per_language_budget": bytes_per_language,
            "english_bytes_budget": english_bytes,
            "records_scanned": scanned,
            "eligible_multilingual_train_records": eligible,
            "eligible_approved_english_records": english_eligible,
            "selected_records_by_domain": selected_records,
            "selected_bytes_by_domain": selected_bytes,
            "total_selected_bytes": sum(selected_bytes.values()),
            "accepted_corpus_sha256": accepted_digest.hexdigest(),
            "english_dataset_sha256": english_digest,
            "file_sha256": file_hashes,
            "sample_sha256": output_hash.hexdigest(),
            "elapsed_seconds": round(elapsed, 3),
            "records_per_second": round(scanned / elapsed, 3),
        }
        _atomic_write_json(manifest_path, manifest)
        _write_progress(progress_path, **manifest)
        return manifest
    except Exception as error:
        _write_progress(
            progress_path,
            stage="failed",
            status="failed",
            records_scanned=scanned,
            eligible_records=eligible,
            error=f"{type(error).__name__}: {error}",
        )
        raise


def _record_payload(prompt: str, target: str) -> bytes:
    return (
        "<|bcnlt:record:begin|>\n"
        + prompt
        + "\n"
        + target
        + "\n<|bcnlt:record:end|>\n"
    ).encode("utf-8")


def _record_key(language: str, record_id: Any, payload: bytes) -> str:
    identity = record_id if isinstance(record_id, str) and record_id else ""
    content_digest = hashlib.sha256(payload).hexdigest()
    return f"{language}:{identity}:{content_digest}"


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_progress(path: Path, **payload: Any) -> None:
    payload["updated_at"] = _now()
    _atomic_write_json(path, payload)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))
