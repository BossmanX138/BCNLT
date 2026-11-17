from __future__ import annotations

from pathlib import Path

import pytest
from runtime_import import select_bcnlt_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE = (
    PROJECT_ROOT / "vendor" / "bcnlt_runtime_v10_learning_candidate" / "source"
)
select_bcnlt_runtime(RUNTIME_SOURCE)

from bcnlt.cache import ProjectIndex  # noqa: E402
from bcnlt.core import ContextualUnit, SourceDocument  # noqa: E402
from bcnlt.uncertainty import HeuristicUncertaintyScorer  # noqa: E402
from bcnlt.uyr import UYRResolver  # noqa: E402


class _NoIndexList(list[ContextualUnit]):
    def index(self, *args: object, **kwargs: object) -> int:
        raise AssertionError("quadratic list.index path was used")


def _fixture() -> tuple[ContextualUnit, SourceDocument, _NoIndexList]:
    unit = ContextualUnit(
        surface="value",
        event="REFERENCE",
        scope="module",
        span=(1, 0, 1, 5),
        ast_relation="identifier.reference",
    )
    document = SourceDocument(
        path="indexed.py",
        source="value\n",
        tree=None,
        index=ProjectIndex(),
    )
    return unit, document, _NoIndexList([unit])


def test_supplied_target_index_avoids_quadratic_unit_search() -> None:
    unit, document, units = _fixture()
    result = UYRResolver(HeuristicUncertaintyScorer()).resolve(
        unit,
        document,
        units,
        target_index=0,
    )
    assert result.unit is unit


def test_compatibility_search_uses_identity_not_dataclass_equality() -> None:
    unit, document, _ = _fixture()
    equal_but_distinct = ContextualUnit(
        surface=unit.surface,
        event=unit.event,
        scope=unit.scope,
        span=unit.span,
        ast_relation=unit.ast_relation,
    )
    units = _NoIndexList([equal_but_distinct, unit])
    result = UYRResolver(HeuristicUncertaintyScorer()).resolve(
        unit,
        document,
        units,
    )
    assert result.unit is unit


def test_compatibility_search_rejects_equal_but_distinct_unit() -> None:
    unit, document, _ = _fixture()
    equal_but_distinct = ContextualUnit(
        surface=unit.surface,
        event=unit.event,
        scope=unit.scope,
        span=unit.span,
        ast_relation=unit.ast_relation,
    )
    with pytest.raises(ValueError, match="absent.*identity"):
        UYRResolver(HeuristicUncertaintyScorer()).resolve(
            unit,
            document,
            _NoIndexList([equal_but_distinct]),
        )


@pytest.mark.parametrize("target_index", [-1, 1, True])
def test_supplied_target_index_is_strictly_validated(target_index: object) -> None:
    unit, document, units = _fixture()
    with pytest.raises(ValueError, match="target_index"):
        UYRResolver(HeuristicUncertaintyScorer()).resolve(
            unit,
            document,
            units,
            target_index=target_index,  # type: ignore[arg-type]
        )
