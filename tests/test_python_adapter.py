import ast

from bcnlt.adapters.python import PythonAdapter
from bcnlt.ir import IRKind


SAMPLE = """
import math

GLOBAL_RATE = 2

def total_price(items):
    subtotal = 0
    for item in items:
        if item.enabled:
            subtotal = subtotal + item.price()
    return math.ceil(subtotal * GLOBAL_RATE)
"""


def test_python_adapter_emits_semantic_units_with_links_and_history():
    adapter = PythonAdapter()

    document = adapter.parse(SAMPLE, path="sample.py")
    units = list(adapter.to_units(document))

    events = [unit.event for unit in units]
    assert "IMPORT" in events
    assert "DEFINE" in events
    assert "ASSIGN" in events
    assert "LOOP" in events
    assert "BRANCH" in events
    assert "CALL" in events
    assert "MEMBER_ACCESS" in events
    assert "RETURN" in events
    assert "OPERATOR" in events
    assert "SCOPE_ENTER" in events
    assert "SCOPE_EXIT" in events
    assert "LITERAL" in events

    subtotal_define = next(
        unit for unit in units if unit.event == "DEFINE" and unit.surface == "subtotal"
    )
    subtotal_refs = [
        unit for unit in units if unit.event == "REFERENCE" and unit.surface == "subtotal"
    ]
    assert subtotal_define.symbol_id
    assert any(ref.defines == subtotal_define.symbol_id for ref in subtotal_refs)

    call = next(unit for unit in units if unit.event == "CALL" and unit.surface == "ceil")
    assert call.scope == "module.total_price"
    assert call.ast_relation == "Call.func"
    assert 0 < len(call.recent_history) <= 8
    assert call.ir.kind == IRKind.CALL


def test_python_adapter_renders_limited_ir_back_to_valid_python():
    adapter = PythonAdapter()
    module_ir = adapter.common_core.module(
        [
            adapter.common_core.assignment("answer", adapter.common_core.literal(42)),
            adapter.common_core.function(
                "read_answer",
                [],
                [adapter.common_core.return_value(adapter.common_core.variable("answer"))],
            ),
        ]
    )

    rendered = adapter.render(module_ir)

    ast.parse(rendered)
    compile(rendered, "<bcnlt-rendered>", "exec")
    assert "answer = 42" in rendered
    assert "def read_answer():" in rendered
