"""Freeze source-controlled regression and deployment-budget category cases."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .category_case_sources import _build_record, _read_verified_source
from .immutable_artifacts import publish_new_files

ACCEPTANCE_POLICY_SCHEMA = "bosscrafts-acceptance-policy-v1"
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class FrozenGateCategoryCases:
    regression_output: Path
    resources_output: Path
    policy_sha256: str
    revision: str
    regression_example_id: str
    resources_example_id: str


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def freeze_gate_category_cases(
    policy_source: str | Path,
    regression_output: str | Path,
    resources_output: str | Path,
    *,
    revision: str,
    expected_source_sha256: str,
) -> FrozenGateCategoryCases:
    """Freeze exact regression and resource cases from a pinned policy file."""

    content, source_sha256 = _read_verified_source(
        policy_source,
        revision=revision,
        expected_source_sha256=expected_source_sha256,
    )
    try:
        policy = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("BossCrafts acceptance policy is invalid JSON") from error
    if type(policy) is not dict or set(policy) != {
        "schema_version",
        "license",
        "attribution",
        "regression",
        "resources",
    }:
        raise ValueError("BossCrafts acceptance policy fields are invalid")
    if (
        policy.get("schema_version") != ACCEPTANCE_POLICY_SCHEMA
        or policy.get("license") != "BossCrafts-Proprietary-Evaluation"
        or policy.get("attribution") != "BossCrafts pre-model acceptance policy"
        or _GIT_REVISION.fullmatch(revision) is None
    ):
        raise ValueError("BossCrafts acceptance policy lineage is invalid")
    regression = policy.get("regression")
    resources = policy.get("resources")
    if (
        type(regression) is not dict
        or set(regression)
        != {"suite_id", "minimum_test_count", "minimum_subtest_count"}
        or type(resources) is not dict
        or set(resources)
        != {
            "vram_limit_bytes",
            "ram_limit_bytes",
            "minimum_tokens_per_second",
        }
    ):
        raise ValueError("BossCrafts acceptance policy cases are invalid")
    provenance: dict[str, object] = {
        "source": "bosscrafts://benchmarks/bosscrafts-acceptance-policy-v1.json",
        "revision": revision,
        "source_sha256": source_sha256,
        "license": policy["license"],
        "attribution": policy["attribution"],
    }
    regression_record = _build_record(
        category="regression",
        case_kind="regression_suite",
        case=regression,
        provenance=provenance,
    )
    resources_record = _build_record(
        category="resources",
        case_kind="resource_budget",
        case=resources,
        provenance=provenance,
    )
    regression_destination = Path(regression_output).resolve()
    resources_destination = Path(resources_output).resolve()
    if regression_destination == resources_destination:
        raise ValueError("gate category outputs must be distinct")
    publish_new_files(
        (
            (regression_destination, _canonical(regression_record) + b"\n"),
            (resources_destination, _canonical(resources_record) + b"\n"),
        ),
        artifact_label="BossCrafts gate category cases",
    )
    return FrozenGateCategoryCases(
        regression_output=regression_destination,
        resources_output=resources_destination,
        policy_sha256=hashlib.sha256(content).hexdigest(),
        revision=revision,
        regression_example_id=str(regression_record["example_id"]),
        resources_example_id=str(resources_record["example_id"]),
    )
