from __future__ import annotations

from enum import Enum
from typing import Any


SEMANTIC_SCHEMA_VERSION = "bcnlt-semantic-record-v2"
UNCERTAINTY_METADATA_KEYS = (
    "dynamic_call",
    "dynamic_binding",
    "macro_expansion",
    "overload_unresolved",
    "preprocessor_condition",
    "unknown_node",
    "knowledge_state",
    "uyr_rounds",
    "uncertainty_signals",
    "wildcard_category",
)


class KnowledgeState(str, Enum):
    KNOWN = "known"
    RETRIEVED = "retrieved"
    PROVISIONAL = "provisional"
    UNRESOLVED = "unresolved"


def validate_uncertainty_metadata(key: str, value: Any) -> None:
    if key not in UNCERTAINTY_METADATA_KEYS:
        raise ValueError(f"Unsupported contextual metadata key: {key}")
    if key in {"dynamic_call", "dynamic_binding", "macro_expansion", "overload_unresolved"}:
        if not isinstance(value, bool):
            raise ValueError(f"{key} metadata must be bool")
        return
    if key in {"preprocessor_condition", "unknown_node", "wildcard_category"}:
        if not isinstance(value, str):
            raise ValueError(f"{key} metadata must be str")
        return
    if key == "knowledge_state":
        if not isinstance(value, str) or value not in {state.value for state in KnowledgeState}:
            raise ValueError("knowledge_state metadata is invalid")
        return
    if key == "uyr_rounds":
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 4:
            raise ValueError("uyr_rounds metadata must be an integer from 0 through 4")
        return
    if key == "uncertainty_signals":
        if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
            raise ValueError("uncertainty_signals metadata must be a string sequence")
        if len(value) > 16:
            raise ValueError("uncertainty_signals metadata is limited to 16 entries")
        return
    raise AssertionError(f"Unhandled uncertainty metadata key: {key}")
