import pytest

from bcnlt.adapters.carousel import BCNLTMultilingualAdapter
from bcnlt.adapters.tree_sitter import TreeSitterSemanticEngine
from bcnlt.ir import IRKind


SOURCES = {
    "python": """def compute(x):
    value = 1
    value = value + x
    if value > 0:
        for i in range(1):
            value = value + i
    return value
compute(2)
""",
    "javascript": """function compute(x) { let value = 1; value = value + x; if (value > 0) { for (let i = 0; i < 1; i++) { value = value + i; } } return value; } compute(2);""",
    "typescript": """function compute(x: number): number { let value = 1; value = value + x; if (value > 0) { for (let i = 0; i < 1; i++) { value = value + i; } } return value; } compute(2);""",
    "c": """int compute(int x) { int value = 1; value = value + x; if (value > 0) { for (int i = 0; i < 1; i++) { value = value + i; } } return value; } int main(void) { return compute(2); }""",
    "cpp": """int compute(int x) { int value = 1; value = value + x; if (value > 0) { for (int i = 0; i < 1; i++) { value = value + i; } } return value; } int main() { return compute(2); }""",
    "c_sharp": """class Program { static int Compute(int x) { int value = 1; value = value + x; if (value > 0) { for (int i = 0; i < 1; i++) { value = value + i; } } return value; } static void Main() { Compute(2); } }""",
    "java": """class Main { static int compute(int x) { int value = 1; value = value + x; if (value > 0) { for (int i = 0; i < 1; i++) { value = value + i; } } return value; } public static void main(String[] args) { compute(2); } }""",
    "go": """package main
func compute(x int) int { value := 1; value = value + x; if value > 0 { for i := 0; i < 1; i++ { value = value + i } }; return value }
func main() { compute(2) }
""",
    "rust": """fn compute(x: i32) -> i32 { let mut value = 1; value = value + x; if value > 0 { for i in 0..1 { value = value + i; } } return value; } fn main() { compute(2); }""",
    "php": """<?php function compute($x) { $value = 1; $value = $value + $x; if ($value > 0) { for ($i = 0; $i < 1; $i++) { $value = $value + $i; } } return $value; } compute(2);""",
    "ruby": """def compute(x)
  value = 1
  value = value + x
  if value > 0
    for i in 0..0
      value = value + i
    end
  end
  return value
end
compute(2)
""",
    "kotlin": """fun compute(x: Int): Int { var value = 1; value = value + x; if (value > 0) { for (i in 0..0) { value = value + i } }; return value } fun main() { compute(2) }""",
    "swift": """func compute(_ x: Int) -> Int {
    var value = 1
    value = value + x
    if value > 0 {
        for i in 0...0 {
            value = value + i
        }
    }
    return value
}
compute(2)
""",
    "bash": """compute() { local x=$1; local value=1; value=$((value + x)); if (( value > 0 )); then for i in 0; do value=$((value + i)); done; fi; return "$value"; }
compute 2
""",
}


@pytest.mark.parametrize("language", tuple(SOURCES))
def test_each_grammar_projects_shared_constructs_into_one_common_core(language: str):
    adapter = BCNLTMultilingualAdapter(engine=TreeSitterSemanticEngine())
    document = adapter.parse(
        SOURCES[language],
        language=language,
        path=f"core.{language}",
    )
    extraction = adapter.extract(document)
    events = {str(unit.event) for unit in extraction.units}
    kinds = {unit.ir.kind for unit in extraction.units if unit.ir is not None}

    assert {"DEFINE", "CALL", "LOOP", "BRANCH", "RETURN"} <= events
    assert {IRKind.FUNCTION, IRKind.CALL, IRKind.LOOP, IRKind.BRANCH, IRKind.RETURN} <= kinds
    assert extraction.coverage.scopes_balanced
    assert 0.0 <= extraction.coverage.mapped_ratio <= 1.0
    assert extraction.coverage.admitted


def test_c_function_names_and_parameters_are_symbol_surfaces():
    adapter = BCNLTMultilingualAdapter(engine=TreeSitterSemanticEngine())
    document = adapter.parse(SOURCES["c"], language="c", path="core.c")
    extraction = adapter.extract(document)

    function = next(
        unit
        for unit in extraction.units
        if unit.event == "DEFINE"
        and unit.ir is not None
        and unit.ir.kind is IRKind.FUNCTION
        and unit.surface == "compute"
    )
    parameter = next(
        unit
        for unit in extraction.units
        if unit.ir is not None and unit.ir.kind is IRKind.PARAMETER
    )

    assert function.symbol_id
    assert parameter.surface == "x"
    assert not any(
        unit.event == "REFERENCE"
        and unit.span == function.span
        and unit.surface == function.surface
        for unit in extraction.units
    )


def test_java_calls_use_the_called_name_and_resolve_the_definition():
    adapter = BCNLTMultilingualAdapter(engine=TreeSitterSemanticEngine())
    document = adapter.parse(SOURCES["java"], language="java", path="Main.java")
    extraction = adapter.extract(document)
    call = next(
        unit
        for unit in extraction.units
        if unit.event == "CALL" and unit.surface == "compute"
    )

    assert call.defines
    assert call.defines in document.index.definitions


def test_type_declarations_project_to_shared_type_definitions():
    adapter = BCNLTMultilingualAdapter(engine=TreeSitterSemanticEngine())
    document = adapter.parse("class Example {}", language="java", path="Example.java")
    extraction = adapter.extract(document)

    type_definition = next(
        unit
        for unit in extraction.units
        if unit.event == "DEFINE"
        and unit.surface == "Example"
        and unit.ir is not None
        and unit.ir.kind is IRKind.TYPE
    )
    assert type_definition.symbol_id
    assert extraction.coverage.scopes_balanced


def test_bash_return_is_not_also_emitted_as_a_call():
    adapter = BCNLTMultilingualAdapter(engine=TreeSitterSemanticEngine())
    document = adapter.parse(SOURCES["bash"], language="bash", path="core.sh")
    extraction = adapter.extract(document)

    assert any(unit.event == "RETURN" and unit.surface == "return" for unit in extraction.units)
    assert not any(unit.event == "CALL" and unit.surface == "return" for unit in extraction.units)


def test_bash_command_substitution_records_a_balanced_subshell_scope():
    adapter = BCNLTMultilingualAdapter(engine=TreeSitterSemanticEngine())
    document = adapter.parse(
        'value=$(printf "%s" "hello")',
        language="bash",
        path="scope.sh",
    )
    extraction = adapter.extract(document)

    assert any(unit.event == "SCOPE_ENTER" for unit in extraction.units)
    assert any(unit.event == "SCOPE_EXIT" for unit in extraction.units)
    assert extraction.coverage.scopes_balanced


def test_surface_child_consumed_by_common_core_rule_is_not_unknown():
    adapter = BCNLTMultilingualAdapter(engine=TreeSitterSemanticEngine())
    document = adapter.parse(
        "const value = object.property;",
        language="javascript",
        path="surface.js",
    )
    extraction = adapter.extract(document)

    assert any(
        unit.event == "MEMBER_ACCESS" and unit.surface == "property"
        for unit in extraction.units
    )
    assert not any(
        unit.event == "UNKNOWN_SEMANTIC"
        and unit.metadata.get("unknown_node") == "property_identifier"
        for unit in extraction.units
    )


def test_lexical_children_of_value_literal_are_covered_by_parent_literal():
    adapter = BCNLTMultilingualAdapter(engine=TreeSitterSemanticEngine())
    document = adapter.parse(
        'message = "hello world"',
        language="python",
        path="literal.py",
    )
    extraction = adapter.extract(document)

    assert any(unit.event == "LITERAL" for unit in extraction.units)
    assert not any(
        unit.event == "UNKNOWN_SEMANTIC"
        and unit.metadata.get("unknown_node")
        in {"string_start", "string_content", "string_end"}
        for unit in extraction.units
    )


@pytest.mark.parametrize(
    ("language", "source", "forbidden_event", "relation_prefix"),
    (
        ("c", "int main(void) { goto done; done: return 0; }", "RAISE", "goto_statement"),
        ("go", "package main\nfunc main() { defer println(1) }", "RAISE", "defer_statement"),
        ("java", "class Example {}", "LITERAL", "class_declaration"),
    ),
)
def test_language_oddities_are_not_forced_into_false_common_core_labels(
    language: str,
    source: str,
    forbidden_event: str,
    relation_prefix: str,
):
    adapter = BCNLTMultilingualAdapter(engine=TreeSitterSemanticEngine())
    document = adapter.parse(source, language=language, path=f"oddity.{language}")
    extraction = adapter.extract(document)

    related = [
        unit for unit in extraction.units if (unit.ast_relation or "").startswith(relation_prefix)
    ]
    assert related
    assert all(unit.event != forbidden_event for unit in related)
