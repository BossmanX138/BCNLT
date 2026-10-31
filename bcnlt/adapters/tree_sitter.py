from __future__ import annotations

import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from bcnlt.cache import ProjectIndex
from bcnlt.core import ContextualUnit, SourceDocument, Span
from bcnlt.ir import CommonCodingCore, IRKind, IRNode
from bcnlt.parsers.bundle import MAX_SOURCE_BYTES, PARSE_TIMEOUT_MS, assert_clean_parse_tree
from bcnlt.preprocessing import SemanticPreprocessor
from bcnlt.semantic_schema import SEMANTIC_SCHEMA_VERSION
from bcnlt.uncertainty import KnowledgeBoundaryAssessment

from .carousel import LanguageCartridge
from .coverage import SemanticCoverage, SemanticExtraction
from .spec import IdentifierRoleRule, NodeRule, ScopeRule
from .symbols import LexicalSymbolTable


ParserProvider = Callable[[str], Any]


def _default_parser_provider(grammar_id: str) -> Any:
    try:
        from tree_sitter_language_pack import PackConfig, cache_dir, configure, get_parser
    except ImportError as exc:  # pragma: no cover - exercised by deployment preflight
        raise ValueError("Install the pinned BCNLT multilingual parser dependencies") from exc

    library_name = (
        f"tree_sitter_{grammar_id}.dll"
        if sys.platform == "win32"
        else f"libtree_sitter_{grammar_id}.so"
    )
    configured_cache = os.environ.get("BCNLT_PARSER_CACHE")
    candidates = [
        Path(configured_cache).expanduser() if configured_cache else None,
        Path.home() / ".cache" / "bcnlt-parser-offline-verify",
        Path.home() / ".cache" / "bcnlt-parser-freezer",
    ]
    active_library = Path(cache_dir()) / library_name
    if active_library.is_file():
        return get_parser(grammar_id)
    for candidate in candidates:
        if candidate is None:
            continue
        configure(PackConfig(cache_dir=str(candidate)))
        if (Path(cache_dir()) / library_name).is_file():
            return get_parser(grammar_id)
    return get_parser(grammar_id)


class TreeSitterSemanticEngine:
    """Shared deterministic semantic extractor for every language cartridge."""

    def __init__(
        self,
        *,
        parser_provider: ParserProvider | None = None,
        wildcard_memory_root: str | Path | None = None,
    ) -> None:
        self._parser_provider = parser_provider or _default_parser_provider
        self._wildcard_memory_root = (
            Path(wildcard_memory_root) if wildcard_memory_root is not None else None
        )
        self._core = CommonCodingCore()
        self._preprocessor = SemanticPreprocessor(memory_root=self._wildcard_memory_root)

    def parse(
        self,
        source: str,
        *,
        cartridge: LanguageCartridge,
        path: str,
        dialect: str | None,
    ) -> SourceDocument:
        source_bytes = source.encode("utf-8")
        if len(source_bytes) > MAX_SOURCE_BYTES:
            raise ValueError(f"Source exceeds the {MAX_SOURCE_BYTES}-byte parser limit")
        parser = self._parser_provider(cartridge.definition.grammar_id)
        started = time.perf_counter()
        tree = parser.parse(source_bytes)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if elapsed_ms > PARSE_TIMEOUT_MS:
            raise ValueError(f"Tree-sitter parse exceeded {PARSE_TIMEOUT_MS}ms")
        assert_clean_parse_tree(tree.root_node)
        return SourceDocument(
            path=path,
            source=source,
            tree=tree,
            index=ProjectIndex(),
            language=cartridge.definition.canonical_id,
            dialect=dialect,
        )

    def extract(
        self,
        document: SourceDocument,
        *,
        cartridge: LanguageCartridge,
    ) -> SemanticExtraction:
        if document.language != cartridge.definition.canonical_id:
            raise ValueError("Document language does not match its semantic cartridge")

        source_bytes = document.source.encode("utf-8")
        spec = cartridge.spec
        node_rules = _group_by_node_type(spec.resolved_node_rules)
        role_rules = _group_by_node_type(spec.resolved_identifier_roles)
        scope_rules = _group_by_node_type(spec.resolved_scope_rules)
        uncertainty_rules = _group_by_node_type(spec.resolved_uncertainty_rules)
        excluded = set(spec.resolved_excluded_node_types)
        symbols = LexicalSymbolTable(language=document.language, path=document.path)
        raw_units: list[ContextualUnit] = []
        named_nodes: set[tuple[int, int, str]] = set()
        mapped_nodes: set[tuple[int, int, str]] = set()
        excluded_nodes: set[tuple[int, int, str]] = set()
        unknown_nodes: set[tuple[int, int, str]] = set()
        definition_owned_nodes: set[tuple[int, int, str]] = set()
        total_named_nodes = 0
        scope_enters = 0
        scope_exits = 0

        def node_key(node: Any) -> tuple[int, int, str]:
            return (int(node.start_byte), int(node.end_byte), str(node.type))

        def text(node: Any) -> str:
            return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

        def span(node: Any) -> Span:
            start_row, start_col = node.start_point
            end_row, end_col = node.end_point
            return (start_row + 1, start_col, end_row + 1, end_col)

        def point_span(point: tuple[int, int]) -> Span:
            row, column = point
            return (row + 1, column, row + 1, column)

        def field(node: Any, name: str | None) -> Any | None:
            if not name:
                return None
            current = node
            for segment in name.split("."):
                if segment.startswith("#"):
                    try:
                        index = int(segment[1:])
                    except ValueError:
                        return None
                    named_children = [
                        child
                        for child in current.children
                        if bool(getattr(child, "is_named", False))
                    ]
                    if index < 0 or index >= len(named_children):
                        return None
                    current = named_children[index]
                else:
                    current = current.child_by_field_name(segment)
                if current is None:
                    return None
            return current

        def cover_named_descendants(node: Any) -> None:
            for child in node.children:
                if bool(getattr(child, "is_named", False)):
                    mapped_nodes.add(node_key(child))
                cover_named_descendants(child)

        def metadata_for(node_type: str) -> dict[str, object]:
            metadata: dict[str, object] = {}
            for uncertainty_rule in uncertainty_rules.get(node_type, ()):
                key = uncertainty_rule.metadata_key
                metadata[key] = True if key in {
                    "dynamic_call",
                    "dynamic_binding",
                    "macro_expansion",
                    "overload_unresolved",
                } else uncertainty_rule.signal
            return metadata

        def emit(
            *,
            surface: str,
            event: str,
            unit_span: Span,
            relation: str,
            symbol_id: str | None = None,
            defines: str | None = None,
            ir: IRNode | None = None,
            metadata: dict[str, object] | None = None,
            scope: str | None = None,
        ) -> None:
            raw_units.append(
                ContextualUnit(
                    surface=surface,
                    event=event,
                    scope=scope or symbols.scope,
                    span=unit_span,
                    symbol_id=symbol_id,
                    defines=defines,
                    ast_relation=relation,
                    ir=ir,
                    metadata=dict(metadata or {}),
                )
            )

        def walk(node: Any) -> None:
            nonlocal total_named_nodes, scope_enters, scope_exits
            key = node_key(node)
            is_named = bool(getattr(node, "is_named", False))
            if is_named:
                total_named_nodes += 1
                named_nodes.add(key)

            roles: dict[str, tuple[str, str | None, Any]] = {}
            for role_rule in role_rules.get(str(node.type), ()):
                child = field(node, role_rule.field_name)
                if child is None:
                    continue
                child_key = node_key(child)
                if bool(getattr(child, "is_named", False)):
                    mapped_nodes.add(child_key)
                name = text(child)
                child_span = span(child)
                if role_rule.role == "definition":
                    definition_owned_nodes.add(child_key)
                    symbol_id = symbols.define(name, child_span)
                    document.index.define(document.path, symbol_id, name, child_span)
                    roles[role_rule.field_name] = (name, symbol_id, child)
                elif role_rule.role == "reference":
                    symbol_id = symbols.resolve(name)
                    document.index.reference(document.path, symbol_id, name, child_span)
                    roles[role_rule.field_name] = (name, symbol_id, child)
                else:
                    raise ValueError(f"Unsupported identifier role: {role_rule.role}")

            rules = node_rules.get(str(node.type), ())
            matching_rules: list[tuple[NodeRule, Any | None, str]] = []
            for rule in rules:
                surface_node = field(node, rule.surface_field)
                surface = text(surface_node or node)
                if rule.surface_equals is not None and surface != rule.surface_equals:
                    continue
                if rule.surface_prefix is not None and not surface.startswith(rule.surface_prefix):
                    continue
                matching_rules.append((rule, surface_node, surface))
            conditional_matches = [
                item
                for item in matching_rules
                if item[0].surface_equals is not None or item[0].surface_prefix is not None
            ]
            if conditional_matches:
                matching_rules = conditional_matches
            if key in definition_owned_nodes:
                matching_rules = [
                    item for item in matching_rules if item[0].event != "REFERENCE"
                ]
            for rule, surface_node, surface in matching_rules:
                mapped_nodes.add(key)
                if rule.event == "LITERAL" and rule.ir_kind == IRKind.VALUE.value:
                    cover_named_descendants(node)
                if surface_node is not None and bool(
                    getattr(surface_node, "is_named", False)
                ):
                    # A named child selected as the semantic surface is already
                    # accounted for by its parent rule. Counting it again as an
                    # unknown node unfairly penalizes languages whose grammars
                    # wrap member, field, or property names in dedicated nodes.
                    mapped_nodes.add(node_key(surface_node))
                role = roles.get(rule.surface_field or "")
                symbol_id = role[1] if role else None
                defines = symbol_id if role and _role_for(role_rules, node.type, rule.surface_field) == "reference" else None
                event_span = span(surface_node or node)
                emit(
                    surface=surface,
                    event=rule.event,
                    unit_span=event_span,
                    relation=f"{node.type}.{rule.surface_field or 'self'}",
                    symbol_id=symbol_id,
                    defines=defines,
                    ir=self._ir_for(rule, surface, document.language, node.type),
                    metadata=metadata_for(str(node.type)),
                )

            for role_rule in role_rules.get(str(node.type), ()):
                if any(rule.surface_field == role_rule.field_name for rule in rules):
                    continue
                role = roles.get(role_rule.field_name)
                if role is None:
                    continue
                name, symbol_id, child = role
                event = "DEFINE" if role_rule.role == "definition" else "REFERENCE"
                emit(
                    surface=name,
                    event=event,
                    unit_span=span(child),
                    relation=f"{node.type}.{role_rule.field_name}",
                    symbol_id=symbol_id,
                    defines=symbol_id if role_rule.role == "reference" else None,
                    ir=self._core.variable(name),
                )

            active_scope: ScopeRule | None = None
            rules_for_scope = scope_rules.get(str(node.type), ())
            if rules_for_scope:
                active_scope = rules_for_scope[0]
                mapped_nodes.add(key)
                name_node = field(node, active_scope.name_field)
                scope_name = text(name_node) if name_node is not None else str(node.type)
                parent_scope = symbols.scope
                scope_id = symbols.enter_scope(active_scope.scope_kind, scope_name, int(node.start_byte))
                scope_enters += 1
                emit(
                    surface=scope_name,
                    event="SCOPE_ENTER",
                    unit_span=point_span(node.start_point),
                    relation=f"{node.type}.scope",
                    symbol_id=scope_id,
                    ir=self._core.extension(
                        f"{document.language}.scope_enter",
                        {"scope_kind": active_scope.scope_kind},
                    ),
                    scope=parent_scope,
                )

            if is_named and key not in mapped_nodes:
                if str(node.type) in excluded:
                    excluded_nodes.add(key)
                else:
                    unknown_nodes.add(key)
                    emit(
                        surface=text(node),
                        event="UNKNOWN_SEMANTIC",
                        unit_span=span(node),
                        relation=f"{node.type}.self",
                        ir=self._core.extension(
                            f"{document.language}.{node.type}",
                            {"node_type": str(node.type)},
                        ),
                        metadata={"unknown_node": str(node.type)},
                    )

            for child in node.children:
                walk(child)

            if active_scope is not None:
                scope_name = symbols.scope.rsplit(".", 1)[-1]
                current_scope = symbols.scope
                symbols.exit_scope()
                scope_exits += 1
                emit(
                    surface=scope_name,
                    event="SCOPE_EXIT",
                    unit_span=point_span(node.end_point),
                    relation=f"{node.type}.scope",
                    ir=self._core.extension(
                        f"{document.language}.scope_exit",
                        {"scope_kind": active_scope.scope_kind},
                    ),
                    scope=current_scope,
                )

        walk(document.tree.root_node)
        ordered = sorted(raw_units, key=lambda unit: unit.span)
        units: list[ContextualUnit] = []
        history: list[str] = []
        for unit in ordered:
            units.append(replace(unit, recent_history=tuple(history[-8:])))
            history.append(unit.surface)

        assessed_units, assessments = self._assess(document, cartridge, units)
        common_core_units = sum(
            1
            for unit in assessed_units
            if unit.ir is not None and unit.ir.kind is not IRKind.EXTENSION
        )
        unknown_units = sum(
            unit.event == "UNKNOWN_SEMANTIC"
            and unit.metadata.get("learned_definition_state") != "validated"
            for unit in assessed_units
        )
        links_valid = all(
            unit.defines is None or unit.defines in document.index.definitions
            for unit in assessed_units
        )
        coverage = SemanticCoverage(
            total_units=len(assessed_units),
            common_core_units=common_core_units,
            unknown_units=unknown_units,
            total_named_nodes=total_named_nodes,
            excluded_named_nodes=len(excluded_nodes.intersection(named_nodes)),
            mapped_named_nodes=len(mapped_nodes.intersection(named_nodes)),
            unknown_named_nodes=len(unknown_nodes.intersection(named_nodes)),
            scopes_balanced=scope_enters == scope_exits and symbols.depth == 1,
            links_valid=links_valid,
        )
        return SemanticExtraction(
            schema_version=SEMANTIC_SCHEMA_VERSION,
            language=document.language,
            dialect=document.dialect,
            document=document,
            units=tuple(assessed_units),
            coverage=coverage,
            assessments=tuple(assessments),
        )

    def _assess(
        self,
        document: SourceDocument,
        cartridge: LanguageCartridge,
        units: list[ContextualUnit],
    ) -> tuple[list[ContextualUnit], list[KnowledgeBoundaryAssessment]]:
        result = self._preprocessor.process(
            document,
            units,
            semantic_spec_sha256=cartridge.spec.canonical_hash,
        )
        return list(result.units), list(result.assessments)

    def _ir_for(
        self,
        rule: NodeRule,
        surface: str,
        language: str,
        node_type: str,
    ) -> IRNode:
        try:
            kind = IRKind(rule.ir_kind)
        except ValueError:
            return self._core.extension(
                f"{language}.{node_type}",
                {"declared_ir_kind": rule.ir_kind},
            )
        if kind is IRKind.EXTENSION:
            return self._core.extension(
                f"{language}.{node_type}",
                {"surface": surface},
            )
        return IRNode(kind=kind, name=surface)


def _group_by_node_type(rules: tuple[Any, ...]) -> dict[str, tuple[Any, ...]]:
    grouped: dict[str, list[Any]] = {}
    for rule in rules:
        grouped.setdefault(rule.node_type, []).append(rule)
    return {key: tuple(value) for key, value in grouped.items()}


def _role_for(
    rules: dict[str, tuple[IdentifierRoleRule, ...]],
    node_type: str,
    field_name: str | None,
) -> str | None:
    for rule in rules.get(str(node_type), ()):
        if rule.field_name == field_name:
            return rule.role
    return None
