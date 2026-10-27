from __future__ import annotations

from bcnlt.adapters.python import PythonAdapter
from bcnlt.events import CURRENT_EVENT_VALUES, SemanticEvent


ADVANCED_SOURCE = '''
class Worker:
    async def run(self, items):
        try:
            async with self.session() as session:
                values = [await session.fetch(item) for item in items if item]
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        finally:
            self.finished = True
        return values


def make_incrementer():
    return lambda value: value + 1


def countdown(limit):
    while limit:
        yield limit
        limit -= 1


def classify(value):
    match value:
        case 0:
            return "zero"
        case _:
            return "other"
'''


def _units(source: str = ADVANCED_SOURCE):
    adapter = PythonAdapter()
    document = adapter.parse(source, path="advanced.py")
    return list(adapter.to_units(document))


def test_event_values_are_stable_and_complete_for_v1_tokenization():
    assert CURRENT_EVENT_VALUES == (
        "IMPORT",
        "DEFINE",
        "REFERENCE",
        "ASSIGN",
        "CALL",
        "RETURN",
        "BRANCH",
        "LOOP",
        "MEMBER_ACCESS",
        "OPERATOR",
        "SCOPE_ENTER",
        "SCOPE_EXIT",
        "LITERAL",
        "UNKNOWN_SEMANTIC",
        "CLASS",
        "ASYNC",
        "CONTEXT_ENTER",
        "CONTEXT_EXIT",
        "TRY",
        "EXCEPT",
        "FINALLY",
        "RAISE",
        "COMPREHENSION",
        "YIELD",
        "AWAIT",
        "MATCH",
    )
    assert tuple(event.value for event in SemanticEvent) == CURRENT_EVENT_VALUES


def test_advanced_constructs_emit_explicit_events_in_their_lexical_scopes():
    units = _units()
    events = {str(unit.event) for unit in units}

    assert {
        "CLASS",
        "ASYNC",
        "CONTEXT_ENTER",
        "CONTEXT_EXIT",
        "TRY",
        "EXCEPT",
        "FINALLY",
        "RAISE",
        "COMPREHENSION",
        "YIELD",
        "AWAIT",
        "MATCH",
    } <= events

    class_unit = next(unit for unit in units if unit.event == "CLASS")
    async_unit = next(unit for unit in units if unit.event == "ASYNC")
    await_unit = next(unit for unit in units if unit.event == "AWAIT")
    comprehension = next(unit for unit in units if unit.event == "COMPREHENSION")

    assert class_unit.surface == "Worker"
    assert class_unit.scope == "module"
    assert class_unit.symbol_id == "module.Worker"
    assert async_unit.surface == "run"
    assert async_unit.scope == "module.Worker"
    assert async_unit.symbol_id == "module.Worker.run"
    assert await_unit.scope.startswith("module.Worker.run.comprehension@")
    assert comprehension.scope == "module.Worker.run"


def test_scope_producers_exit_cleanly_and_do_not_leak_definitions():
    units = _units()
    entries = [unit for unit in units if unit.event == "SCOPE_ENTER"]
    exits = [unit for unit in units if unit.event == "SCOPE_EXIT"]

    assert len(entries) == len(exits)
    assert any(unit.surface.startswith("lambda@") for unit in entries)
    assert any(unit.surface.startswith("comprehension@") for unit in entries)

    classify = next(
        unit for unit in units if unit.event == "DEFINE" and unit.surface == "classify"
    )
    assert classify.scope == "module"
    assert classify.symbol_id == "module.classify"
    assert all(not unit.scope.startswith("module.Worker") for unit in units if unit is classify)


def test_unhandled_semantic_node_is_visible_as_unknown_and_still_traverses_children():
    units = _units("value = 1\ndel value\n")

    unknown = next(unit for unit in units if unit.event == "UNKNOWN_SEMANTIC")
    reference = next(
        unit for unit in units if unit.event == "REFERENCE" and unit.surface == "value"
    )

    assert unknown.surface == "del value"
    assert unknown.metadata == {"unknown_node": "Delete"}
    assert reference.defines == "module.value"


def test_empty_module_has_no_semantic_units():
    assert _units("") == []
