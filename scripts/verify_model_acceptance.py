"""Verify held-out and inference evidence for the final BossCrafts checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bosscrafts_model.model_acceptance import (
    ACCEPTANCE_CATEGORIES,
    verify_model_acceptance,
)


def _read_payload(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"acceptance evidence is unreadable: {path}") from error
    if type(payload) is not dict:
        raise ValueError(f"acceptance evidence is not a JSON object: {path}")
    return payload


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-evaluation", type=Path, required=True)
    parser.add_argument("--test-evaluation", type=Path, required=True)
    parser.add_argument("--code-to-semantic-inference", type=Path, required=True)
    parser.add_argument("--semantic-continuation-inference", type=Path, required=True)
    parser.add_argument("--semantic-to-code-inference", type=Path, required=True)
    for category in ACCEPTANCE_CATEGORIES:
        parser.add_argument(
            f"--{category.replace('_', '-')}-evidence",
            dest=f"{category}_evidence",
            type=Path,
            required=True,
        )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    result = verify_model_acceptance(
        (
            _read_payload(arguments.validation_evaluation),
            _read_payload(arguments.test_evaluation),
        ),
        (
            _read_payload(arguments.code_to_semantic_inference),
            _read_payload(arguments.semantic_continuation_inference),
            _read_payload(arguments.semantic_to_code_inference),
        ),
        tuple(
            _read_payload(getattr(arguments, f"{category}_evidence"))
            for category in ACCEPTANCE_CATEGORIES
        ),
    )
    payload = asdict(result)
    payload["acceptance_manifest_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    destination = arguments.output.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(destination)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
