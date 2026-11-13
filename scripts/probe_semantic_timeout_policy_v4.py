"""Measure previously timed-out sources under a bounded diagnostic ceiling."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sqlite3
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _probe(item: tuple[str, str, str | None, str, float]) -> dict[str, object]:
    language, source, dialect, relative_path, timeout_seconds = item
    from bosscrafts_model.bcnlt_source_pipeline import BCNLTSourcePipeline
    from bosscrafts_model.tokenizer import NativeTokenizerBinding

    pipeline = BCNLTSourcePipeline(
        NativeTokenizerBinding.load(),
        persistent_sidecars=True,
        timeout_seconds=timeout_seconds,
    )
    started = time.perf_counter()
    try:
        response = pipeline._invoke_sidecar(
            source,
            language=language,
            path=relative_path,
            dialect=dialect,
        )
        return {
            "relative_path": relative_path,
            "completed": True,
            "seconds": time.perf_counter() - started,
            "record_count": response["record_count"],
            "canonical_bytes": len(str(response["canonical_jsonl"]).encode("utf-8")),
            "error": None,
        }
    except Exception as error:
        return {
            "relative_path": relative_path,
            "completed": False,
            "seconds": time.perf_counter() - started,
            "record_count": 0,
            "canonical_bytes": 0,
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        pipeline.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("relative_paths", nargs="+")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 16:
        parser.error("workers must be between 1 and 16")
    if not 1 <= arguments.timeout_seconds <= 60:
        parser.error("timeout-seconds must be between 1 and 60")
    connection = sqlite3.connect(arguments.database)
    try:
        items = []
        for relative_path in arguments.relative_paths:
            row = connection.execute(
                "SELECT canonical_language, content, dialect FROM documents "
                "WHERE relative_path=?",
                (relative_path,),
            ).fetchone()
            if row is None:
                raise ValueError(f"catalog path not found: {relative_path}")
            items.append(
                (
                    str(row[0]),
                    str(row[1]),
                    str(row[2]) if row[2] is not None else None,
                    relative_path,
                    arguments.timeout_seconds,
                )
            )
    finally:
        connection.close()
    with ProcessPoolExecutor(
        max_workers=arguments.workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        results = list(executor.map(_probe, items))
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
