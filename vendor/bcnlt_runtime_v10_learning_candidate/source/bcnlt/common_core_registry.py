from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .adapters.spec import IdentifierRoleRule, LanguageSemanticSpec, NodeRule, ScopeRule
from .ir import COMMON_CODING_CONCEPTS
from .languages import CANONICAL_LANGUAGE_IDS, LANGUAGE_BY_ID


COMMON_CORE_REGISTRY_VERSION = "bcnlt-common-core-v1"
REQUIRED_GRAMMAR_CONCEPTS = frozenset(
    {
        "literal",
        "variable_definition",
        "reference",
        "function",
        "call",
        "assignment",
        "loop",
        "branch",
        "return",
        "operator",
    }
)


@dataclass(frozen=True, slots=True)
class SyntaxAlias:
    concept_id: str
    language: str
    node_type: str
    surface_field: str | None = None
    role: str | None = None
    scope_kind: str | None = None
    surface_equals: str | None = None
    surface_prefix: str | None = None


_CONCEPT_BY_ID = {item.concept_id: item for item in COMMON_CODING_CONCEPTS}


def _per_language(**values: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    missing = set(CANONICAL_LANGUAGE_IDS).difference(values)
    if missing:
        raise ValueError(f"Common Core alias table is missing canonical languages: {sorted(missing)}")
    return values


_NODES = {
    "literal": _per_language(
        python=("string", "integer", "float", "true", "false", "none"),
        javascript=("string", "number", "true", "false", "null"),
        typescript=("string", "number", "true", "false", "null"),
        c=("string_literal", "char_literal", "number_literal", "true", "false", "null"),
        cpp=("string_literal", "char_literal", "number_literal", "true", "false", "null"),
        c_sharp=("string_literal", "character_literal", "integer_literal", "real_literal", "boolean_literal", "null_literal"),
        java=("string_literal", "character_literal", "decimal_integer_literal", "decimal_floating_point_literal", "true", "false", "null_literal"),
        go=("interpreted_string_literal", "raw_string_literal", "rune_literal", "int_literal", "float_literal", "true", "false", "nil"),
        rust=("string_literal", "char_literal", "integer_literal", "float_literal", "boolean_literal"),
        php=("string", "integer", "float", "boolean", "null"),
        ruby=("string", "integer", "float", "true", "false", "nil"),
        kotlin=("string_literal", "character_literal", "integer_literal", "real_literal", "boolean_literal", "null_literal"),
        swift=("line_string_literal", "multiline_string_literal", "integer_literal", "real_literal", "boolean_literal", "nil"),
        bash=("string", "raw_string", "number", "word"),
    ),
    "variable_definition": _per_language(
        python=("assignment",), javascript=("variable_declarator",), typescript=("variable_declarator",),
        c=("init_declarator",), cpp=("init_declarator",), c_sharp=("variable_declarator",),
        java=("variable_declarator",), go=("var_spec", "short_var_declaration"), rust=("let_declaration",),
        php=("assignment_expression",), ruby=("assignment",), kotlin=("variable_declaration",),
        swift=("property_declaration",), bash=("variable_assignment",),
    ),
    "reference": _per_language(
        python=("identifier",), javascript=("identifier",), typescript=("identifier",), c=("identifier",),
        cpp=("identifier",), c_sharp=("identifier",), java=("identifier",), go=("identifier",),
        rust=("identifier",), php=("variable_name",), ruby=("identifier",), kotlin=("simple_identifier",),
        swift=("simple_identifier",), bash=("variable_name",),
    ),
    "parameter": _per_language(
        python=("default_parameter", "typed_parameter"), javascript=("required_parameter",),
        typescript=("required_parameter",), c=("parameter_declaration",), cpp=("parameter_declaration",),
        c_sharp=("parameter",), java=("formal_parameter",),
        go=("parameter_declaration", "variadic_parameter_declaration"),
        rust=("parameter",), php=("simple_parameter",), ruby=(),
        kotlin=("parameter",), swift=("parameter",), bash=("special_variable_name",),
    ),
    "function": _per_language(
        python=("function_definition", "lambda"), javascript=("function_declaration", "method_definition", "arrow_function"),
        typescript=("function_declaration", "method_definition", "arrow_function"), c=("function_definition",),
        cpp=("function_definition", "lambda_expression"), c_sharp=("method_declaration", "constructor_declaration", "lambda_expression"),
        java=("method_declaration", "constructor_declaration", "lambda_expression"), go=("function_declaration", "method_declaration", "func_literal"),
        rust=("function_item", "closure_expression"), php=("function_definition", "method_declaration", "anonymous_function"),
        ruby=("method", "singleton_method", "lambda"), kotlin=("function_declaration", "lambda_literal"),
        swift=("function_declaration", "initializer_declaration", "lambda_literal"), bash=("function_definition",),
    ),
    "call": _per_language(
        python=("call",), javascript=("call_expression", "new_expression"), typescript=("call_expression", "new_expression"),
        c=("call_expression",), cpp=("call_expression",), c_sharp=("invocation_expression", "object_creation_expression"),
        java=("method_invocation", "object_creation_expression"), go=("call_expression",), rust=("call_expression",),
        php=("function_call_expression", "member_call_expression"), ruby=("call", "method_call"),
        kotlin=("call_expression",), swift=("call_expression",), bash=("command",),
    ),
    "assignment": _per_language(
        python=("assignment", "augmented_assignment"), javascript=("assignment_expression", "augmented_assignment_expression"),
        typescript=("assignment_expression", "augmented_assignment_expression"), c=("assignment_expression",),
        cpp=("assignment_expression",), c_sharp=("assignment_expression",), java=("assignment_expression",),
        go=("assignment_statement", "short_var_declaration"), rust=("assignment_expression",),
        php=("assignment_expression", "augmented_assignment_expression"), ruby=("assignment", "operator_assignment"),
        kotlin=("assignment",), swift=("assignment",), bash=("variable_assignment",),
    ),
    "loop": _per_language(
        python=("for_statement", "while_statement"), javascript=("for_statement", "for_in_statement", "while_statement", "do_statement"),
        typescript=("for_statement", "for_in_statement", "while_statement", "do_statement"), c=("for_statement", "while_statement", "do_statement"),
        cpp=("for_statement", "for_range_loop", "while_statement", "do_statement"), c_sharp=("for_statement", "foreach_statement", "while_statement", "do_statement"),
        java=("for_statement", "enhanced_for_statement", "while_statement", "do_statement"), go=("for_statement",),
        rust=("for_expression", "while_expression", "loop_expression"), php=("for_statement", "foreach_statement", "while_statement", "do_statement"),
        ruby=("for", "while", "until"), kotlin=("for_statement", "while_statement", "do_while_statement"),
        swift=("for_statement", "while_statement", "repeat_while_statement"), bash=("for_statement", "while_statement", "c_style_for_statement"),
    ),
    "branch": _per_language(
        python=("if_statement", "match_statement"), javascript=("if_statement", "switch_statement", "ternary_expression"),
        typescript=("if_statement", "switch_statement", "ternary_expression"), c=("if_statement", "switch_statement", "conditional_expression"),
        cpp=("if_statement", "switch_statement", "conditional_expression"), c_sharp=("if_statement", "switch_statement", "switch_expression", "conditional_expression"),
        java=("if_statement", "switch_expression", "ternary_expression"), go=("if_statement", "expression_switch_statement", "type_switch_statement"),
        rust=("if_expression", "match_expression"), php=("if_statement", "switch_statement", "conditional_expression"),
        ruby=("if", "unless", "case"), kotlin=("if_expression", "when_expression"),
        swift=("if_statement", "switch_statement", "guard_statement"), bash=("if_statement", "case_statement"),
    ),
    "return": _per_language(
        python=("return_statement",), javascript=("return_statement",), typescript=("return_statement",), c=("return_statement",),
        cpp=("return_statement",), c_sharp=("return_statement",), java=("return_statement",), go=("return_statement",),
        rust=("return_expression",), php=("return_statement",), ruby=("return",), kotlin=("jump_expression",),
        swift=("control_transfer_statement",), bash=("command",),
    ),
    "operator": _per_language(
        python=("binary_operator", "unary_operator"), javascript=("binary_expression", "unary_expression", "update_expression"),
        typescript=("binary_expression", "unary_expression", "update_expression"), c=("binary_expression", "unary_expression", "update_expression"),
        cpp=("binary_expression", "unary_expression", "update_expression"), c_sharp=("binary_expression", "prefix_unary_expression", "postfix_unary_expression"),
        java=("binary_expression", "unary_expression", "update_expression"), go=("binary_expression", "unary_expression", "inc_statement"),
        rust=("binary_expression", "unary_expression"), php=("binary_expression", "unary_op_expression"),
        ruby=("binary", "unary"), kotlin=("binary_expression", "additive_expression", "comparison_expression", "unary_expression"),
        swift=("binary_expression", "additive_expression", "comparison_expression", "prefix_expression"), bash=("binary_expression", "unary_expression"),
    ),
    "comparison": _per_language(
        python=("comparison_operator",), javascript=("relational_expression",), typescript=("relational_expression",),
        c=("conditional_expression",), cpp=("conditional_expression",), c_sharp=("relational_pattern",),
        java=("instanceof_expression",), go=("expression_switch_statement",), rust=("range_expression",),
        php=("relational_expression",), ruby=("comparison",), kotlin=("range_expression",),
        swift=("comparison_operator",), bash=("test_operator",),
    ),
    "data_structure": _per_language(
        python=("list", "dictionary", "set", "tuple"), javascript=("array", "object"), typescript=("array", "object"),
        c=("initializer_list",), cpp=("initializer_list",), c_sharp=("array_creation_expression", "collection_expression"),
        java=("array_creation_expression", "array_initializer"), go=("composite_literal",), rust=("array_expression", "tuple_expression", "struct_expression"),
        php=("array_creation_expression",), ruby=("array", "hash"), kotlin=("collection_literal",),
        swift=("array_literal", "dictionary_literal", "range_expression"), bash=("array",),
    ),
    "error": _per_language(
        python=("raise_statement",), javascript=("throw_statement",), typescript=("throw_statement",),
        c=(), cpp=("throw_statement",), c_sharp=("throw_statement",),
        java=("throw_statement",), go=(), rust=(),
        php=("throw_expression",), ruby=(), kotlin=("throw_expression",),
        swift=("throw_statement",), bash=(),
    ),
    "member_access": _per_language(
        python=("attribute",), javascript=("member_expression", "subscript_expression"), typescript=("member_expression", "subscript_expression"),
        c=("field_expression", "subscript_expression"), cpp=("field_expression", "subscript_expression"), c_sharp=("member_access_expression", "element_access_expression"),
        java=("field_access", "array_access"), go=("selector_expression", "index_expression"), rust=("field_expression", "index_expression"),
        php=("member_access_expression", "subscript_expression"), ruby=("scope_resolution",), kotlin=("navigation_expression", "indexing_expression"),
        swift=("navigation_expression", "subscript_expression"), bash=("subscript",),
    ),
    "import": _per_language(
        python=("import_statement", "import_from_statement"), javascript=("import_statement",), typescript=("import_statement",),
        c=("preproc_include",), cpp=("preproc_include", "using_declaration"), c_sharp=("using_directive",),
        java=("import_declaration",), go=("import_declaration", "package_clause"), rust=("use_declaration", "mod_item"),
        php=("namespace_use_declaration", "include_expression"), ruby=(), kotlin=("import_header",),
        swift=("import_declaration",), bash=("source_command",),
    ),
}


# Grammar-only containers and punctuation-bearing wrappers do not express a
# second language-specific semantic system. Excluding them keeps coverage
# focused on concepts projected through the shared catalog above.
_EXCLUDED_NODES = _per_language(
    python=(
        "module", "block", "parameters", "argument_list", "string_start",
        "string_content", "string_end", "escape_sequence", "concatenated_string",
    ),
    javascript=(
        "program", "statement_block", "formal_parameters", "arguments",
        "expression_statement", "lexical_declaration", "parenthesized_expression",
        "string_fragment", "escape_sequence",
    ),
    typescript=(
        "program", "statement_block", "formal_parameters", "arguments",
        "expression_statement", "lexical_declaration", "parenthesized_expression",
        "predefined_type", "type_annotation", "string_fragment", "escape_sequence",
    ),
    c=(
        "translation_unit", "compound_statement", "parameter_list", "argument_list",
        "declaration", "expression_statement", "function_declarator",
        "parenthesized_expression", "pointer_declarator", "primitive_type",
        "string_content", "escape_sequence", "system_lib_string", "concatenated_string",
    ),
    cpp=(
        "translation_unit", "compound_statement", "parameter_list", "argument_list",
        "condition_clause", "declaration", "expression_statement",
        "function_declarator", "pointer_declarator", "primitive_type",
        "string_content", "escape_sequence", "system_lib_string",
    ),
    c_sharp=(
        "compilation_unit", "declaration_list", "block", "parameter_list",
        "argument_list", "argument", "expression_statement",
        "local_declaration_statement", "modifier",
        "predefined_type", "class_declaration", "variable_declaration",
        "string_literal_content", "string_content", "interpolation_brace",
    ),
    java=(
        "program", "class_body", "block", "formal_parameters", "argument_list",
        "expression_statement", "local_variable_declaration", "modifiers",
        "dimensions", "integral_type", "parenthesized_expression", "type_identifier",
        "void_type", "string_fragment", "escape_sequence",
    ),
    go=(
        "source_file", "statement_list", "block", "parameter_list", "argument_list",
        "expression_list", "expression_statement", "for_clause",
        "package_identifier", "type_identifier", "interpreted_string_literal_content",
        "raw_string_literal_content", "escape_sequence",
    ),
    rust=(
        "source_file", "block", "parameters", "arguments", "expression_statement",
        "mutable_specifier", "primitive_type", "string_content", "escape_sequence",
    ),
    php=(
        "program", "php_tag", "compound_statement", "formal_parameters", "arguments",
        "argument", "expression_statement", "name", "parenthesized_expression",
        "string_content", "escape_sequence",
    ),
    ruby=(
        "program", "body_statement", "argument_list", "do", "in", "then", "range",
        "string_content", "escape_sequence",
    ),
    kotlin=(
        "source_file", "function_body", "function_value_parameters", "statements",
        "control_structure_body", "call_suffix", "value_arguments", "value_argument",
        "directly_assignable_expression", "binding_pattern_kind", "type_identifier",
        "user_type", "property_declaration", "string_content",
    ),
    swift=(
        "source_file", "function_body", "statements", "call_suffix",
        "value_arguments", "value_argument", "directly_assignable_expression",
        "pattern", "value_binding_pattern", "type_identifier", "user_type", "line_str_text",
    ),
    bash=(
        "program", "compound_statement", "do_group", "declaration_command",
        "string_content",
        "command_name", "arithmetic_expansion", "simple_expansion",
    ),
)


_SURFACE_FIELDS = {
    "variable_definition": "name",
    "function": "name",
    "call": "function",
    "assignment": "left",
    "member_access": "property",
}
_DEFINITION_CONCEPTS = {"variable_definition", "parameter", "function"}
_ALIAS_DETAILS = {
    ("variable_definition", "python", "assignment"): {"surface_field": "left"},
    ("variable_definition", "c", "init_declarator"): {"surface_field": "declarator.declarator|declarator"},
    ("variable_definition", "cpp", "init_declarator"): {"surface_field": "declarator.declarator|declarator"},
    ("variable_definition", "go", "var_spec"): {"surface_field": "name"},
    ("variable_definition", "go", "short_var_declaration"): {"surface_field": "left"},
    ("variable_definition", "rust", "let_declaration"): {"surface_field": "pattern"},
    ("variable_definition", "php", "assignment_expression"): {"surface_field": "left"},
    ("variable_definition", "ruby", "assignment"): {"surface_field": "left"},
    ("variable_definition", "kotlin", "variable_declaration"): {"surface_field": "#0"},
    ("variable_definition", "swift", "property_declaration"): {"surface_field": "name.bound_identifier"},
    ("parameter", "python", "default_parameter"): {"surface_field": "name"},
    ("parameter", "python", "typed_parameter"): {"surface_field": "name"},
    ("parameter", "javascript", "required_parameter"): {"surface_field": "pattern"},
    ("parameter", "typescript", "required_parameter"): {"surface_field": "pattern"},
    ("parameter", "c", "parameter_declaration"): {"surface_field": "declarator"},
    ("parameter", "cpp", "parameter_declaration"): {"surface_field": "declarator"},
    ("parameter", "c_sharp", "parameter"): {"surface_field": "name"},
    ("parameter", "java", "formal_parameter"): {"surface_field": "name"},
    ("parameter", "go", "parameter_declaration"): {"surface_field": "name"},
    ("parameter", "go", "variadic_parameter_declaration"): {"surface_field": "name"},
    ("parameter", "rust", "parameter"): {"surface_field": "pattern"},
    ("parameter", "php", "simple_parameter"): {"surface_field": "name"},
    ("parameter", "kotlin", "parameter"): {"surface_field": "#0"},
    ("parameter", "swift", "parameter"): {"surface_field": "name"},
    ("function", "c", "function_definition"): {"surface_field": "declarator.declarator"},
    ("function", "cpp", "function_definition"): {"surface_field": "declarator.declarator"},
    ("function", "kotlin", "function_declaration"): {"surface_field": "@simple_identifier"},
    ("call", "java", "method_invocation"): {"surface_field": "name"},
    ("call", "ruby", "call"): {"surface_field": "method"},
    ("call", "kotlin", "call_expression"): {"surface_field": "#0"},
    ("call", "swift", "call_expression"): {"surface_field": "#0"},
    ("call", "bash", "command"): {"surface_field": "name"},
    ("member_access", "python", "attribute"): {"surface_field": "attribute"},
    ("member_access", "c", "field_expression"): {"surface_field": "field"},
    ("member_access", "cpp", "field_expression"): {"surface_field": "field"},
    ("member_access", "c_sharp", "member_access_expression"): {"surface_field": "name"},
    ("member_access", "java", "field_access"): {"surface_field": "field"},
    ("member_access", "go", "selector_expression"): {"surface_field": "field"},
    ("member_access", "rust", "field_expression"): {"surface_field": "field"},
    ("member_access", "php", "member_access_expression"): {"surface_field": "name"},
    ("member_access", "ruby", "scope_resolution"): {"surface_field": "name"},
    ("member_access", "kotlin", "navigation_expression"): {"surface_field": "#1.#0"},
    ("member_access", "swift", "navigation_expression"): {"surface_field": "suffix.suffix"},
    ("return", "kotlin", "jump_expression"): {"surface_prefix": "return"},
    ("return", "swift", "control_transfer_statement"): {"surface_prefix": "return"},
    ("return", "bash", "command"): {
        "surface_field": "name",
        "surface_equals": "return",
    },
}


def syntax_aliases_for(language: str) -> tuple[SyntaxAlias, ...]:
    if language not in LANGUAGE_BY_ID:
        raise ValueError(f"Use an exact registered canonical language id, got: {language}")
    aliases: list[SyntaxAlias] = []
    for concept_id, by_language in _NODES.items():
        for node_type in by_language[language]:
            details = _ALIAS_DETAILS.get((concept_id, language, node_type), {})
            surface_field = details.get(
                "surface_field", _SURFACE_FIELDS.get(concept_id)
            )
            aliases.append(
                SyntaxAlias(
                    concept_id=concept_id,
                    language=language,
                    node_type=node_type,
                    surface_field=surface_field,
                    role=("definition" if concept_id in _DEFINITION_CONCEPTS else "reference" if concept_id in {"call", "reference"} else None),
                    scope_kind=("function" if concept_id == "function" else None),
                    surface_equals=details.get("surface_equals"),
                    surface_prefix=details.get("surface_prefix"),
                )
            )
    if language == "ruby":
        aliases.extend(
            SyntaxAlias(
                concept_id="import",
                language="ruby",
                node_type="call",
                surface_field="method",
                surface_equals=loader,
            )
            for loader in ("require", "require_relative", "load")
        )
    return tuple(aliases)


def build_semantic_spec(language: str) -> LanguageSemanticSpec:
    aliases = syntax_aliases_for(language)
    node_rules = []
    identifier_roles = []
    scope_rules = []
    for alias in aliases:
        concept = _CONCEPT_BY_ID[alias.concept_id]
        node_rules.append(
            NodeRule(
                alias.node_type,
                concept.event,
                concept.ir_kind.value,
                alias.surface_field,
                alias.surface_equals,
                alias.surface_prefix,
            )
        )
        if alias.role is not None:
            identifier_roles.append(
                IdentifierRoleRule(alias.node_type, alias.surface_field or "", alias.role)
            )
        if alias.scope_kind is not None:
            scope_rules.append(ScopeRule(alias.node_type, alias.scope_kind, alias.surface_field))
    if language == "python":
        identifier_roles.append(
            IdentifierRoleRule("identifier", "", "definition", parent_type="parameters")
        )
    # Parameter containers are not symbols. Direct identifiers are handled
    # here; the runtime's grammar-aware binding-pattern walker owns defaults
    # and destructuring.
    parameter_parent = {"javascript": "formal_parameters", "ruby": "method_parameters"}.get(language)
    if parameter_parent is not None:
        identifier_roles.append(
            IdentifierRoleRule("identifier", "", "definition", parent_type=parameter_parent)
        )
    if language == "python":
        # Preserve class ownership of methods with otherwise identical names.
        scope_rules.append(ScopeRule("class_definition", "class", "name"))
    type_scope_rules = {
        "c_sharp": (("class_declaration", "name"),),
        "cpp": (
            ("class_specifier", "name"),
            ("struct_specifier", "name"),
            ("union_specifier", "name"),
        ),
        "java": (
            ("class_declaration", "name"),
            ("interface_declaration", "name"),
            ("enum_declaration", "name"),
        ),
        "javascript": (("class_declaration", "name"),),
        "typescript": (("class_declaration", "name"),),
        "kotlin": (("class_declaration", "@type_identifier"),),
        "php": (
            ("class_declaration", "name"),
            ("interface_declaration", "name"),
            ("trait_declaration", "name"),
        ),
        "ruby": (("class", "name"), ("module", "name")),
        "rust": (("impl_item", "type"), ("trait_item", "name")),
        "swift": (
            ("class_declaration", "name"),
            ("struct_declaration", "name"),
            ("protocol_declaration", "name"),
        ),
    }
    scope_rules.extend(
        ScopeRule(node_type, "class", name_field)
        for node_type, name_field in type_scope_rules.get(language, ())
    )
    if language == "go":
        scope_rules.append(ScopeRule("block", "block"))
        # Go control statements own implicit lexical scopes that begin before
        # their initializer. A short declaration in ``if x := ...`` must not
        # be mistaken for assignment to an outer-frame ``x``.
        scope_rules.extend(
            ScopeRule(node_type, "block")
            for node_type in (
                "if_statement",
                "for_statement",
                "expression_switch_statement",
                "type_switch_statement",
                "select_statement",
            )
        )
    if language in {"c", "cpp"}:
        # Braced lexical blocks own bindings; pop them before subsequent siblings.
        scope_rules.append(ScopeRule("compound_statement", "block"))
    if language in {"javascript", "typescript"}:
        scope_rules.append(ScopeRule("statement_block", "block"))
        scope_rules.append(ScopeRule("catch_clause", "block"))
    return LanguageSemanticSpec(
        semantic_spec_version=LANGUAGE_BY_ID[language].semantic_spec_version,
        node_rules=tuple(node_rules),
        identifier_roles=tuple(identifier_roles),
        scope_rules=tuple(scope_rules),
        excluded_node_types=tuple((*_EXCLUDED_NODES[language], "comment")),
    )


def registry_digest() -> str:
    payload = {
        "schema_version": COMMON_CORE_REGISTRY_VERSION,
        "concepts": [
            {
                "concept_id": item.concept_id,
                "event": item.event,
                "ir_kind": item.ir_kind.value,
            }
            for item in COMMON_CODING_CONCEPTS
        ],
        "aliases": [
            asdict(alias)
            for language in CANONICAL_LANGUAGE_IDS
            for alias in syntax_aliases_for(language)
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
