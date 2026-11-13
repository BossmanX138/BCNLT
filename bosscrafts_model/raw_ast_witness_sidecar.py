"""Isolated raw Tree-sitter witness sidecar; semantic extraction is not invoked."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = PROJECT_ROOT / "vendor" / "bcnlt_runtime_v10_learning_candidate"
sys.path.insert(0, str(RUNTIME_ROOT / "site-packages"))
sys.path.insert(0, str(RUNTIME_ROOT / "source"))
os.environ["BCNLT_PARSER_CACHE"] = str(RUNTIME_ROOT / "grammars")

PROTOCOL = "bosscrafts-raw-ast-witness-v1"
PARSER_RUNTIME_IDENTITY_SHA256 = (
    "d0a0a61605faf57270d9df4828a01c281ace1af095e87c86fa2a5812b1cd950a"
)
MAX_SOURCE_BYTES = 200 * 1024
MAX_QUERIES = 16_384
MAX_REQUEST_BYTES = MAX_SOURCE_BYTES * 6 + 4 * 1024 * 1024


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _load_request(raw: bytes | None = None) -> dict[str, object]:
    if raw is None:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        raise ValueError("request exceeds bounded transport")
    payload = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys
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
        if type(query) is not dict or set(query) not in (
            {"query_id", "span", "ast_relation"},
            {"query_id", "span", "ast_relation", "scope"},
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
            or type(relation) is not str
            or not relation
            or (
                "scope" in query
                and (type(query["scope"]) is not str or not query["scope"])
            )
        ):
            raise ValueError("query is invalid")
        seen.add(query_id)
    return payload


def _run(payload: dict[str, object]) -> dict[str, object]:
    from bcnlt.adapters import BCNLTMultilingualAdapter, LanguageCarousel
    from bcnlt.adapters.tree_sitter import TreeSitterSemanticEngine

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
    scope_rules_by_type: dict[str, object] = {}
    for rule in cartridge.spec.resolved_scope_rules:
        scope_rules_by_type.setdefault(rule.node_type, rule)
    adapter = BCNLTMultilingualAdapter(
        engine=TreeSitterSemanticEngine(), carousel=carousel
    )
    document = adapter.parse(
        source, language=language, path=path, dialect=dialect
    )

    exact: dict[tuple[int, int, int, int], set[str]] = {}
    starts: dict[tuple[int, int], set[str]] = {}
    ends: dict[tuple[int, int], set[str]] = {}
    named_nodes: list[object] = []
    stack = [document.tree.root_node]
    while stack:
        node = stack.pop()
        stack.extend(reversed(node.children))
        if not bool(getattr(node, "is_named", False)):
            continue
        named_nodes.append(node)
        start_row, start_col = node.start_point
        end_row, end_col = node.end_point
        span = (start_row + 1, start_col, end_row + 1, end_col)
        node_type = str(node.type)
        exact.setdefault(span, set()).add(node_type)
        starts.setdefault((span[0], span[1]), set()).add(node_type)
        ends.setdefault((span[2], span[3]), set()).add(node_type)

    def node_span(node: object) -> tuple[int, int, int, int]:
        start_row, start_col = node.start_point
        end_row, end_col = node.end_point
        return (start_row + 1, start_col, end_row + 1, end_col)

    def node_key(node: object) -> tuple[int, int, str]:
        return (int(node.start_byte), int(node.end_byte), str(node.type))

    def resolve_field(node: object, field_path: str) -> object | None:
        if (
            field_path == "name"
            and language in {"javascript", "typescript"}
            and str(node.type) == "arrow_function"
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
                    return binding
        for alternative in field_path.split("|"):
            current = node
            for segment in alternative.split("."):
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

    def derived_scope(
        owner: object,
        relation_kind: str,
        span: tuple[int, int, int, int],
    ) -> str:
        ancestry: list[object] = []
        current = owner
        while current is not None:
            ancestry.append(current)
            current = getattr(current, "parent", None)
        ancestry.reverse()
        components = ["module"]
        owner_key = node_key(owner)
        point = (span[0], span[1])
        for node in ancestry:
            rule = scope_rules_by_type.get(str(node.type))
            if rule is None:
                continue
            parent = getattr(node, "parent", None)
            if (
                language == "go"
                and str(node.type) == "block"
                and parent is not None
                and str(parent.type)
                in {"function_declaration", "method_declaration", "func_literal"}
            ):
                continue
            if node_key(node) == owner_key:
                if relation_kind != "scope":
                    continue
                node_end = node_span(node)[2:4]
                if point != node_end:
                    continue
            name_field = rule.name_field
            name_node = (
                resolve_field(node, name_field)
                if isinstance(name_field, str) and name_field
                else None
            )
            name = (
                source_bytes[name_node.start_byte : name_node.end_byte].decode(
                    "utf-8", errors="replace"
                )
                if name_node is not None
                else str(node.type)
            )
            components.append(name)
        return ".".join(components)

    def declared_role_fallback(node: object, relation: str) -> bool:
        requirements = declared_role_fallbacks.get(relation)
        if not requirements:
            return False
        parent = getattr(node, "parent", None)
        parent_type = str(parent.type) if parent is not None else None
        return None in requirements or parent_type in requirements

    witnesses: list[dict[str, object]] = []
    for query in payload["queries"]:
        span = tuple(query["span"])
        relation = str(query["ast_relation"])
        owner_type, separator, relation_kind = relation.partition(".")
        exact_types = sorted(exact.get(span, set()))
        boundary_start_types = sorted(starts.get((span[0], span[1]), set()))
        boundary_end_types = sorted(ends.get((span[2], span[3]), set()))
        matching_owners: list[object] = []
        if not separator:
            pass
        elif relation_kind == "self":
            matching_owners = [
                node
                for node in named_nodes
                if str(node.type) == owner_type and node_span(node) == span
            ]
        elif relation_kind == "scope" and span[0:2] == span[2:4]:
            matching_owners = [
                node
                for node in named_nodes
                if str(node.type) == owner_type
                and (node_span(node)[0:2] == span[0:2] or node_span(node)[2:4] == span[0:2])
            ]
        else:
            matching_owners = [
                node
                for node in named_nodes
                if str(node.type) == owner_type
                for resolved in (resolve_field(node, relation_kind),)
                if (
                    (resolved is not None and node_span(resolved) == span)
                    or (
                        resolved is None
                        and (
                            relation in declared_node_relations
                            or declared_role_fallback(node, relation)
                        )
                        and node_span(node) == span
                    )
                )
            ]
        relation_match = bool(matching_owners)
        claimed_scope = query.get("scope")
        derived_scopes = sorted(
            {
                derived_scope(owner, relation_kind, span)
                for owner in matching_owners
            }
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
