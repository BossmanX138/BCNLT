"""Isolated raw Tree-sitter witness sidecar; semantic extraction is not invoked."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "vendor" / "bcnlt_runtime_v10_learning_candidate"
sys.path.insert(0, str(RUNTIME_ROOT / "site-packages"))
sys.path.insert(0, str(RUNTIME_ROOT / "source"))
os.environ["BCNLT_PARSER_CACHE"] = str(RUNTIME_ROOT / "grammars")

PROTOCOL = "bosscrafts-raw-ast-witness-v2"
SIDECAR_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
PARSER_RUNTIME_IDENTITY_SHA256 = (
    "18992265d9cc6337aac12cab6c88307cbe5a1b855c39844e7681c6f7df2885b8"
)
MAX_SOURCE_BYTES = 200 * 1024
MAX_QUERIES = 16_384
MAX_REQUEST_BYTES = MAX_SOURCE_BYTES * 6 + 4 * 1024 * 1024


class _Node(Protocol):
    @property
    def children(self) -> Sequence[_Node]: ...

    @property
    def start_point(self) -> tuple[int, int]: ...

    @property
    def end_point(self) -> tuple[int, int]: ...

    @property
    def start_byte(self) -> int: ...

    @property
    def end_byte(self) -> int: ...

    @property
    def type(self) -> str: ...

    @property
    def is_named(self) -> bool: ...

    @property
    def parent(self) -> _Node | None: ...

    def child_by_field_name(self, name: str) -> _Node | None: ...

    def children_by_field_name(self, name: str) -> Sequence[_Node]: ...


class _ScopeRule(Protocol):
    @property
    def name_field(self) -> str | None: ...

    @property
    def scope_kind(self) -> str: ...


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _load_request(raw: bytes | None = None) -> dict[str, object]:
    if raw is None:
        request_bytes = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    else:
        request_bytes = raw
    if len(request_bytes) > MAX_REQUEST_BYTES:
        raise ValueError("request exceeds bounded transport")
    payload = json.loads(
        request_bytes.decode("utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    if type(payload) is not dict or set(payload) != {
        "protocol",
        "source",
        "language",
        "path",
        "dialect",
        "queries",
    }:
        raise ValueError("request keys mismatch")
    if payload["protocol"] != PROTOCOL:
        raise ValueError("protocol mismatch")
    source = payload["source"]
    queries = payload["queries"]
    if type(source) is not str or len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("source is invalid")
    if (
        type(payload["language"]) is not str
        or not payload["language"]
        or type(payload["path"]) is not str
        or not payload["path"]
        or (payload["dialect"] is not None and type(payload["dialect"]) is not str)
        or type(queries) is not list
        or not queries
        or len(queries) > MAX_QUERIES
    ):
        raise ValueError("request metadata is invalid")
    seen: set[str] = set()
    for query in queries:
        required_query_keys = {"query_id", "span", "ast_relation"}
        allowed_query_keys = {*required_query_keys, "scope", "surface"}
        if (
            type(query) is not dict
            or not required_query_keys.issubset(query)
            or not set(query).issubset(allowed_query_keys)
        ):
            raise ValueError("query keys mismatch")
        query_id = query["query_id"]
        span = query["span"]
        relation = query["ast_relation"]
        if (
            type(query_id) is not str
            or not query_id
            or query_id in seen
            or type(span) is not list
            or len(span) != 4
            or any(type(value) is not int or value < 0 for value in span)
            or span[0] < 1
            or span[2] < span[0]
            or (span[2] == span[0] and span[3] < span[1])
            or type(relation) is not str
            or not relation
            or (
                "scope" in query
                and (type(query["scope"]) is not str or not query["scope"])
            )
            or ("surface" in query and type(query["surface"]) is not str)
        ):
            raise ValueError("query is invalid")
        seen.add(query_id)
    return payload


def _run(payload: dict[str, object]) -> dict[str, object]:
    from bcnlt.adapters import (
        BCNLTMultilingualAdapter,
        LanguageCarousel,
    )
    from bcnlt.adapters.tree_sitter import (
        TreeSitterSemanticEngine,
    )

    source = str(payload["source"])
    language = str(payload["language"])
    path = str(payload["path"])
    dialect_value = payload["dialect"]
    dialect = dialect_value if isinstance(dialect_value, str) else None
    carousel = LanguageCarousel.default()
    cartridge = carousel.for_language(language)
    declared_node_relations = {
        f"{rule.node_type}.{rule.surface_field or 'self'}"
        for rule in cartridge.spec.resolved_node_rules
    }
    declared_role_fallbacks: dict[str, set[str | None]] = {}
    for rule in cartridge.spec.resolved_identifier_roles:
        relation_role = rule.field_name or (
            "parameter"
            if rule.parent_type is not None and "parameter" in rule.parent_type
            else rule.role
        )
        declared_role_fallbacks.setdefault(
            f"{rule.node_type}.{relation_role}", set()
        ).add(rule.parent_type)
    scope_rules_by_type: dict[str, _ScopeRule] = {}
    for rule in cartridge.spec.resolved_scope_rules:
        scope_rules_by_type.setdefault(rule.node_type, cast(_ScopeRule, rule))
    adapter = BCNLTMultilingualAdapter(
        engine=TreeSitterSemanticEngine(), carousel=carousel
    )
    document = adapter.parse(source, language=language, path=path, dialect=dialect)

    exact: dict[tuple[int, int, int, int], set[str]] = {}
    starts: dict[tuple[int, int], set[str]] = {}
    ends: dict[tuple[int, int], set[str]] = {}
    nodes_by_span: dict[tuple[int, int, int, int], list[_Node]] = {}
    named_nodes_by_type: dict[str, list[_Node]] = {}
    named_starts: dict[tuple[int, int], list[_Node]] = {}
    named_ends: dict[tuple[int, int], list[_Node]] = {}
    stack: list[_Node] = [cast(_Node, document.tree.root_node)]
    while stack:
        node = stack.pop()
        stack.extend(reversed(node.children))
        start_row, start_col = node.start_point
        end_row, end_col = node.end_point
        span = (start_row + 1, start_col, end_row + 1, end_col)
        nodes_by_span.setdefault(span, []).append(node)
        if not bool(getattr(node, "is_named", False)):
            continue
        node_type = str(node.type)
        named_nodes_by_type.setdefault(node_type, []).append(node)
        named_starts.setdefault((span[0], span[1]), []).append(node)
        named_ends.setdefault((span[2], span[3]), []).append(node)
        exact.setdefault(span, set()).add(node_type)
        starts.setdefault((span[0], span[1]), set()).add(node_type)
        ends.setdefault((span[2], span[3]), set()).add(node_type)

    def node_span(node: _Node) -> tuple[int, int, int, int]:
        start_row, start_col = node.start_point
        end_row, end_col = node.end_point
        return (start_row + 1, start_col, end_row + 1, end_col)

    def node_key(node: _Node) -> tuple[int, int, str]:
        return (int(node.start_byte), int(node.end_byte), str(node.type))

    def resolve_field(node: _Node, field_path: str) -> _Node | None:
        if (
            field_path == "name"
            and language in {"javascript", "typescript"}
            and str(node.type) in {"arrow_function", "function_expression"}
            and node.child_by_field_name("name") is None
        ):
            parent = getattr(node, "parent", None)
            if parent is not None and str(parent.type) == "variable_declarator":
                value = parent.child_by_field_name("value")
                binding = parent.child_by_field_name("name")
                if (
                    value is not None
                    and node_key(value) == node_key(node)
                    and binding is not None
                    and str(binding.type) == "identifier"
                ):
                    return cast(_Node, binding)
        for alternative in field_path.split("|"):
            current: _Node | None = node
            for segment in alternative.split("."):
                if current is None:
                    break
                if segment.startswith("#"):
                    try:
                        index = int(segment[1:])
                    except ValueError:
                        current = None
                        break
                    children = [
                        child
                        for child in current.children
                        if bool(getattr(child, "is_named", False))
                    ]
                    current = children[index] if 0 <= index < len(children) else None
                elif segment.startswith("@"):
                    expected_type = segment[1:]
                    current = next(
                        (
                            child
                            for child in current.children
                            if bool(getattr(child, "is_named", False))
                            and str(child.type) == expected_type
                        ),
                        None,
                    )
                else:
                    current = current.child_by_field_name(segment)
                if current is None:
                    break
            if current is not None:
                return current
        return None

    source_bytes = source.encode("utf-8")

    def canonical_surface(node: _Node) -> str:
        raw = source_bytes[node.start_byte : node.end_byte].decode(
            "utf-8", errors="replace"
        )
        return unicodedata.normalize(
            "NFC", raw[:4096].replace("\r\n", "\n").replace("\r", "\n")
        )

    def is_within(node: _Node, ancestor: _Node) -> bool:
        ancestor_key = node_key(ancestor)
        current: _Node | None = node
        while current is not None:
            if node_key(current) == ancestor_key:
                return True
            current = getattr(current, "parent", None)
        return False

    def synthetic_relation_matches(
        owner: _Node,
        target: _Node,
        relation: str,
    ) -> bool:
        owner_type = str(owner.type)
        target_type = str(target.type)
        if relation == "import_clause.binding":
            target_parent = getattr(target, "parent", None)
            return (
                target_type == "identifier"
                and target_parent is not None
                and node_key(target_parent) == node_key(owner)
            )
        if relation == "namespace_import.binding":
            return target_type == "identifier" and is_within(target, owner)
        if relation == "import_specifier.binding":
            imported = owner.child_by_field_name("name")
            alias = owner.child_by_field_name("alias")
            selected = alias or imported
            return selected is not None and node_key(selected) == node_key(target)
        if relation == "short_var_declaration.binding":
            left = owner.child_by_field_name("left")
            return (
                target_type == "identifier"
                and left is not None
                and is_within(target, left)
            )
        if relation == "range_clause.binding":
            left = owner.child_by_field_name("left")
            return (
                target_type == "identifier"
                and left is not None
                and is_within(target, left)
                and any(str(child.type) == ":=" for child in owner.children)
            )
        if (
            relation in {"parameters.binding", "lambda_parameters.binding"}
            and language != "ruby"
        ):
            if target_type != "identifier":
                return False
            current: _Node | None = target
            while current is not None:
                current_parent = getattr(current, "parent", None)
                if (
                    current_parent is not None
                    and node_key(current_parent) == node_key(owner)
                ):
                    break
                current = getattr(current, "parent", None)
            if current is None:
                return False
            wrapper_type = str(current.type)
            if wrapper_type == "identifier":
                return node_key(current) == node_key(target)
            if wrapper_type in {"list_splat_pattern", "dictionary_splat_pattern"}:
                return is_within(target, current)
            if wrapper_type not in {
                "default_parameter",
                "typed_default_parameter",
                "typed_parameter",
            }:
                return False
            name = current.child_by_field_name("name")
            if name is not None:
                return node_key(name) == node_key(target)
            type_node = current.child_by_field_name("type")
            value = current.child_by_field_name("value")
            return is_within(target, current) and not (
                (type_node is not None and is_within(target, type_node))
                or (value is not None and is_within(target, value))
            )
        if relation == "parameter_declaration.binding":
            declarator = owner.child_by_field_name("declarator")
            if declarator is None or target_type != "identifier":
                return False

            def declarator_contains(binding: _Node) -> bool:
                if str(binding.type) == "identifier":
                    return node_key(binding) == node_key(target)
                nested = binding.child_by_field_name("declarator")
                if nested is None and str(binding.type) in {
                    "abstract_pointer_declarator",
                    "array_declarator",
                    "parenthesized_declarator",
                    "pointer_declarator",
                    "reference_declarator",
                }:
                    nested = next(
                        (
                            child
                            for child in binding.children
                            if bool(getattr(child, "is_named", False))
                            and str(child.type) != "comment"
                        ),
                        None,
                    )
                return nested is not None and declarator_contains(nested)

            return declarator_contains(declarator)
        if relation == "parameter_list.binding":
            return any(
                node_key(binding) == node_key(target)
                for binding in owner.children_by_field_name("name")
            )
        if relation == "spread_parameter.binding":
            declarator = next(
                (
                    child
                    for child in owner.children
                    if bool(getattr(child, "is_named", False))
                    and str(child.type) == "variable_declarator"
                ),
                None,
            )
            name = declarator.child_by_field_name("name") if declarator is not None else None
            return name is not None and node_key(name) == node_key(target)
        if relation == "variadic_parameter.binding":
            name = owner.child_by_field_name("name")
            return name is not None and node_key(name) == node_key(target)
        if relation == "lambda_expression.parameter":
            parameters = owner.child_by_field_name("parameters")
            return (
                parameters is not None
                and (
                    node_key(target) == node_key(parameters)
                    or (target_type == "identifier" and is_within(target, parameters))
                )
            )
        if relation == "lambda_parameter.binding":
            return (
                target_type == "simple_identifier"
                and any(
                    node_key(binding) == node_key(target)
                    for binding in owner.children_by_field_name("name")
                )
            )
        if relation == "parameter.pattern_binding":
            pattern = owner.child_by_field_name("pattern")
            if pattern is None or target_type not in {
                "identifier",
                "shorthand_field_identifier",
            }:
                return False

            def rust_pattern_contains(binding: _Node) -> bool:
                binding_type = str(binding.type)
                if binding_type in {"identifier", "shorthand_field_identifier"}:
                    return node_key(binding) == node_key(target)
                if binding_type not in {
                    "captured_pattern",
                    "field_pattern",
                    "mut_pattern",
                    "ref_pattern",
                    "reference_pattern",
                    "slice_pattern",
                    "struct_pattern",
                    "tuple_pattern",
                    "tuple_struct_pattern",
                }:
                    return False
                if binding_type == "field_pattern":
                    nested = binding.child_by_field_name("pattern")
                    selected = nested or binding.child_by_field_name("name")
                    return selected is not None and rust_pattern_contains(selected)
                if binding_type == "struct_pattern":
                    children = [
                        child
                        for child in binding.children
                        if bool(getattr(child, "is_named", False))
                        and str(child.type) == "field_pattern"
                    ]
                elif binding_type == "tuple_struct_pattern":
                    type_node = binding.child_by_field_name("type")
                    children = [
                        child
                        for child in binding.children
                        if bool(getattr(child, "is_named", False))
                        and (type_node is None or node_key(child) != node_key(type_node))
                    ]
                else:
                    children = [
                        child
                        for child in binding.children
                        if bool(getattr(child, "is_named", False))
                    ]
                return any(rust_pattern_contains(child) for child in children)

            return rust_pattern_contains(pattern)
        if relation in {"method_parameters.binding", "lambda_parameters.binding"}:
            if target_type != "identifier":
                return False

            def ruby_binding_contains(binding: _Node) -> bool:
                binding_type = str(binding.type)
                if binding_type == "identifier":
                    return node_key(binding) == node_key(target)
                if binding_type == "destructured_parameter":
                    return any(
                        ruby_binding_contains(child)
                        for child in binding.children
                        if bool(getattr(child, "is_named", False))
                    )
                if binding_type in {
                    "block_parameter",
                    "hash_splat_parameter",
                    "keyword_parameter",
                    "keyword_splat_parameter",
                    "optional_parameter",
                    "splat_parameter",
                }:
                    name = binding.child_by_field_name("name")
                    return name is not None and is_within(target, name)
                return False

            return any(
                ruby_binding_contains(child)
                for child in owner.children
                if bool(getattr(child, "is_named", False))
            )
        if (
            relation
            in {
                "parameter_declaration.name",
                "variadic_parameter_declaration.name",
            }
            and owner_type
            in {"parameter_declaration", "variadic_parameter_declaration"}
        ):
            return any(
                node_key(binding) == node_key(target)
                for binding in owner.children_by_field_name("name")
            )
        if relation in {
            "variable_declarator.binding",
            "required_parameter.binding",
            "optional_parameter.binding",
        }:
            field_name = "name" if owner_type == "variable_declarator" else "pattern"
            binding = owner.child_by_field_name(field_name)
            return (
                binding is not None
                and target_type
                in {"identifier", "shorthand_property_identifier_pattern"}
                and is_within(target, binding)
            )
        if relation == "formal_parameters.binding":
            return target_type in {
                "identifier",
                "shorthand_property_identifier_pattern",
            } and is_within(target, owner)
        if relation == "catch_clause.parameter":
            parameter = owner.child_by_field_name("parameter")
            return parameter is not None and is_within(target, parameter)
        if relation == "arrow_function.parameter":
            parameter = owner.child_by_field_name("parameter")
            return parameter is not None and node_key(parameter) == node_key(target)
        if relation == "arrow_function.implicit_return":
            body = owner.child_by_field_name("body")
            return body is not None and node_key(body) == node_key(target)
        if relation == "lambda.implicit_return":
            body = owner.child_by_field_name("body")
            if language == "ruby" and body is not None:
                block_body = (
                    body.child_by_field_name("body")
                    if str(body.type) == "block"
                    else body
                )
                children = (
                    [
                        child
                        for child in block_body.children
                        if bool(getattr(child, "is_named", False))
                    ]
                    if block_body is not None
                    else []
                )
                tail = children[-1] if children else None
                return (
                    tail is not None
                    and str(tail.type) != "return"
                    and node_key(tail) == node_key(target)
                )
            return body is not None and node_key(body) == node_key(target)
        if relation == "lambda_expression.implicit_return":
            body = owner.child_by_field_name("body")
            return body is not None and node_key(body) == node_key(target)
        if relation == "closure_expression.implicit_return":
            body = owner.child_by_field_name("body")
            if body is None:
                return False
            if str(body.type) != "block":
                return node_key(body) == node_key(target)
            children = [
                child
                for child in body.children
                if bool(getattr(child, "is_named", False))
                and str(child.type) not in {"line_comment", "block_comment"}
            ]
            tail = children[-1] if children else None
            if tail is not None and str(tail.type) == "expression_statement":
                if any(str(child.type) == ";" for child in tail.children):
                    tail = None
                else:
                    tail = next(
                        (
                            child
                            for child in tail.children
                            if bool(getattr(child, "is_named", False))
                        ),
                        None,
                    )
            return (
                tail is not None
                and str(tail.type) != "return_expression"
                and (
                    str(tail.type).endswith(("_expression", "_literal"))
                    or str(tail.type) in {"identifier", "block", "self"}
                )
                and node_key(tail) == node_key(target)
            )
        if relation == "lambda_literal.implicit_return":
            statements = next(
                (
                    child
                    for child in owner.children
                    if bool(getattr(child, "is_named", False))
                    and str(child.type) == "statements"
                ),
                None,
            )
            children = (
                [
                    child
                    for child in statements.children
                    if bool(getattr(child, "is_named", False))
                ]
                if statements is not None
                else []
            )
            tail = children[-1] if children else None
            tail_type = str(tail.type) if tail is not None else ""
            if language == "kotlin":
                expression_like = tail_type.endswith(("_expression", "_literal")) or tail_type in {
                    "boolean_literal",
                    "integer_literal",
                    "null_literal",
                    "real_literal",
                    "simple_identifier",
                    "string_literal",
                }
            elif language == "swift":
                expression_like = tail_type.endswith(("_expression", "_literal")) or tail_type in {
                    "boolean_literal",
                    "nil",
                    "simple_identifier",
                }
            else:
                expression_like = False
            return (
                tail is not None
                and expression_like
                and node_key(tail) == node_key(target)
            )
        if relation == "method_declaration.implicit_return":
            body = owner.child_by_field_name("body")
            target_expression = (
                next(
                    (
                        child
                        for child in body.children
                        if bool(getattr(child, "is_named", False))
                    ),
                    None,
                )
                if body is not None and str(body.type) == "arrow_expression_clause"
                else None
            )
            return (
                target_expression is not None
                and node_key(target_expression) == node_key(target)
            )
        if relation == "method.implicit_return":
            body = owner.child_by_field_name("body")
            children = [
                child
                for child in body.children
                if bool(getattr(child, "is_named", False))
            ] if body is not None else []
            tail = children[-1] if children else None
            return tail is not None and str(tail.type) != "return" and node_key(tail) == node_key(target)
        if relation == "function_declaration.implicit_return":
            if language == "kotlin":
                body = next(
                    (
                        child
                        for child in owner.children
                        if bool(getattr(child, "is_named", False))
                        and str(child.type) == "function_body"
                    ),
                    None,
                )
                if body is None or not canonical_surface(body).lstrip().startswith("="):
                    return False
                expression = next(
                    (
                        child
                        for child in body.children
                        if bool(getattr(child, "is_named", False))
                    ),
                    None,
                )
                return expression is not None and node_key(expression) == node_key(target)
            if language == "swift":
                body = owner.child_by_field_name("body")
                statements = (
                    next(
                        (
                            child
                            for child in body.children
                            if bool(getattr(child, "is_named", False))
                            and str(child.type) == "statements"
                        ),
                        None,
                    )
                    if body is not None
                    else None
                )
                expressions = (
                    [
                        child
                        for child in statements.children
                        if bool(getattr(child, "is_named", False))
                    ]
                    if statements is not None
                    else []
                )
                expression_type = (
                    str(expressions[0].type) if len(expressions) == 1 else ""
                )
                return (
                    len(expressions) == 1
                    and (
                        expression_type.endswith(("_expression", "_literal"))
                        or expression_type
                        in {
                            "boolean_literal",
                            "nil",
                            "simple_identifier",
                        }
                    )
                    and node_key(expressions[0]) == node_key(target)
                )
            return False
        if relation == "function_item.implicit_tail":
            body = owner.child_by_field_name("body")
            if body is None or not is_within(target, body):
                return False
            children = [
                child
                for child in body.children
                if bool(getattr(child, "is_named", False))
                and str(child.type) not in {"line_comment", "block_comment"}
            ]
            tail = children[-1] if children else None
            if tail is not None and str(tail.type) == "expression_statement":
                if any(str(child.type) == ";" for child in tail.children):
                    tail = None
                else:
                    tail = next(
                        (
                            child
                            for child in tail.children
                            if bool(getattr(child, "is_named", False))
                        ),
                        None,
                    )
            return tail is not None and node_key(tail) == node_key(target)
        return False

    def derived_scope(
        owner: _Node,
        relation: str,
        span: tuple[int, int, int, int],
    ) -> str:
        ancestry: list[_Node] = []
        current: _Node | None = owner
        while current is not None:
            ancestry.append(current)
            current = getattr(current, "parent", None)
        ancestry.reverse()
        components = ["module"]
        owner_key = node_key(owner)
        relation_kind = relation.partition(".")[2]
        include_owner_scope = relation in {
            "arrow_function.parameter",
            "arrow_function.implicit_return",
            "catch_clause.parameter",
            "closure_expression.implicit_return",
            "function_declaration.implicit_return",
            "function_item.implicit_tail",
            "lambda.implicit_return",
            "lambda_literal.implicit_return",
            "lambda_expression.implicit_return",
            "lambda_expression.parameter",
            "method.implicit_return",
            "method_declaration.implicit_return",
        }
        parent = getattr(owner, "parent", None)
        javascript_var_hoist = (
            language in {"javascript", "typescript"}
            and str(owner.type) == "variable_declarator"
            and parent is not None
            and str(parent.type) == "variable_declaration"
            and source_bytes[parent.start_byte : parent.end_byte]
            .decode("utf-8", errors="replace")
            .lstrip()
            .startswith("var ")
        )
        javascript_var_owner_index = -1
        if javascript_var_hoist:
            for index in range(len(ancestry) - 1, -1, -1):
                candidate_rule = scope_rules_by_type.get(str(ancestry[index].type))
                if candidate_rule is not None and candidate_rule.scope_kind == "function":
                    javascript_var_owner_index = index
                    break
        for index, node in enumerate(ancestry):
            rule = scope_rules_by_type.get(str(node.type))
            if rule is None:
                continue
            node_parent = getattr(node, "parent", None)
            if (
                language == "go"
                and str(node.type) == "block"
                and node_parent is not None
                and str(node_parent.type)
                in {"function_declaration", "method_declaration", "func_literal"}
            ):
                continue
            if (
                javascript_var_hoist
                and rule.scope_kind == "block"
                and index > javascript_var_owner_index
            ):
                continue
            if node_key(node) == owner_key:
                if relation_kind == "scope":
                    if span[0:2] != node_span(node)[2:4]:
                        continue
                elif not include_owner_scope:
                    continue
            name_field = rule.name_field
            name_node = (
                resolve_field(node, name_field)
                if isinstance(name_field, str) and name_field
                else None
            )
            name = (
                canonical_surface(name_node)
                if name_node is not None
                else str(node.type)
            )
            components.append(name)
        return ".".join(components)

    def declared_role_fallback(node: _Node, relation: str) -> bool:
        requirements = declared_role_fallbacks.get(relation)
        if not requirements:
            return False
        parent = getattr(node, "parent", None)
        parent_type = str(parent.type) if parent is not None else None
        return None in requirements or parent_type in requirements

    witnesses: list[dict[str, object]] = []
    queries = cast(list[dict[str, object]], payload["queries"])
    resolved_field_indexes: dict[
        tuple[str, str],
        dict[tuple[int, int, int, int], list[_Node]],
    ] = {}

    def field_owners(
        owner_type: str,
        relation_kind: str,
        target_span: tuple[int, int, int, int],
    ) -> tuple[_Node, ...]:
        key = (owner_type, relation_kind)
        index = resolved_field_indexes.get(key)
        if index is None:
            index = {}
            for candidate in named_nodes_by_type.get(owner_type, ()):
                resolved = resolve_field(candidate, relation_kind)
                if resolved is not None:
                    index.setdefault(node_span(resolved), []).append(candidate)
            resolved_field_indexes[key] = index
        return tuple(index.get(target_span, ()))

    for query in queries:
        span_values = cast(list[int], query["span"])
        span = (span_values[0], span_values[1], span_values[2], span_values[3])
        relation = str(query["ast_relation"])
        owner_type, separator, relation_kind = relation.partition(".")
        exact_types = sorted(exact.get(span, set()))
        boundary_start_types = sorted(starts.get((span[0], span[1]), set()))
        boundary_end_types = sorted(ends.get((span[2], span[3]), set()))
        matching_owners: list[_Node] = []
        exact_targets = tuple(nodes_by_span.get(span, ()))
        if not separator:
            pass
        elif relation_kind == "self":
            matching_owners = [
                node
                for node in exact_targets
                if bool(getattr(node, "is_named", False))
                and str(node.type) == owner_type
            ]
        elif relation_kind == "scope" and span[0:2] == span[2:4]:
            boundary_nodes = {
                node_key(node): node
                for node in (
                    *named_starts.get(span[0:2], ()),
                    *named_ends.get(span[0:2], ()),
                )
                if str(node.type) == owner_type
            }
            matching_owners = list(boundary_nodes.values())
        else:
            candidate_owners: dict[tuple[int, int, str], _Node] = {}
            for target in exact_targets:
                current: _Node | None = target
                while current is not None:
                    if (
                        bool(getattr(current, "is_named", False))
                        and str(current.type) == owner_type
                    ):
                        candidate_owners[node_key(current)] = current
                    current = getattr(current, "parent", None)
            for node in exact_targets:
                if (
                    bool(getattr(node, "is_named", False))
                    and str(node.type) == owner_type
                ):
                    candidate_owners[node_key(node)] = node
            for node in field_owners(owner_type, relation_kind, span):
                candidate_owners[node_key(node)] = node
            for node in candidate_owners.values():
                if any(
                    synthetic_relation_matches(node, target, relation)
                    for target in exact_targets
                ):
                    matching_owners.append(node)
                    continue
                resolved = resolve_field(node, relation_kind)
                if resolved is not None and node_span(resolved) == span:
                    matching_owners.append(node)
                    continue
                if (
                    resolved is None
                    and (
                        relation in declared_node_relations
                        or declared_role_fallback(node, relation)
                    )
                    and node_span(node) == span
                ):
                    matching_owners.append(node)
        relation_match = bool(matching_owners)
        derived_surfaces: set[str] = set()
        if relation_match:
            if relation_kind == "scope":
                for owner in matching_owners:
                    rule = scope_rules_by_type.get(str(owner.type))
                    if rule is None:
                        continue
                    name_field = rule.name_field
                    name_node = (
                        resolve_field(owner, name_field)
                        if isinstance(name_field, str) and name_field
                        else None
                    )
                    derived_surfaces.add(
                        canonical_surface(name_node)
                        if name_node is not None
                        else str(owner.type)
                    )
            else:
                derived_surfaces.update(canonical_surface(node) for node in exact_targets)
        unique_surface = (
            next(iter(derived_surfaces)) if len(derived_surfaces) == 1 else None
        )
        claimed_surface = query.get("surface")
        claimed_scope = query.get("scope")
        derived_scopes = sorted(
            {derived_scope(owner, relation, span) for owner in matching_owners}
        )
        unique_scope = derived_scopes[0] if len(derived_scopes) == 1 else None
        witnesses.append(
            {
                "query_id": query["query_id"],
                "span": list(span),
                "ast_relation": relation,
                "exact_named_node_types": exact_types,
                "boundary_start_named_node_types": boundary_start_types,
                "boundary_end_named_node_types": boundary_end_types,
                "relation_span_match": relation_match,
                "claimed_surface": claimed_surface,
                "derived_surface": unique_surface,
                "surface_match": (
                    claimed_surface in derived_surfaces
                    if isinstance(claimed_surface, str)
                    else None
                ),
                "claimed_scope": claimed_scope,
                "derived_scope": unique_scope,
                "scope_match": (
                    claimed_scope in derived_scopes
                    if isinstance(claimed_scope, str)
                    else None
                ),
            }
        )
    return {
        "protocol": PROTOCOL,
        "sidecar_source_sha256": SIDECAR_SOURCE_SHA256,
        "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "language": language,
        "path": path,
        "dialect": dialect,
        "root_node_type": str(document.tree.root_node.type),
        "witnesses": witnesses,
    }


def main() -> int:
    serving = "--serve" in sys.argv[1:]
    if serving:
        if sys.argv[1:] != ["--serve"]:
            print("raw AST witness rejected arguments", file=sys.stderr)
            return 2
        while True:
            raw = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 2)
            if not raw:
                break
            if len(raw) > MAX_REQUEST_BYTES + 1:
                print(
                    "raw AST witness rejected input: ValueError",
                    file=sys.stderr,
                    flush=True,
                )
                return 2
            try:
                response = _run(_load_request(raw.rstrip(b"\r\n")))
            except Exception as exc:
                print(
                    f"raw AST witness rejected input: {type(exc).__name__}",
                    file=sys.stderr,
                    flush=True,
                )
                return 2
            encoded = (
                json.dumps(
                    response,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            sys.stdout.buffer.write(encoded)
            sys.stdout.buffer.flush()
        return 0
    try:
        response = _run(_load_request())
    except Exception as exc:
        print(f"raw AST witness rejected input: {type(exc).__name__}", file=sys.stderr)
        return 2
    encoded = (
        json.dumps(response, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    sys.stdout.buffer.write(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
