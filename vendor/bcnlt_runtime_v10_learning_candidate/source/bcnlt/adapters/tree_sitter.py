from __future__ import annotations

import ctypes
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from bcnlt.cache import ProjectIndex
from bcnlt.core import ContextualUnit, SourceDocument, Span
from bcnlt.ir import CommonCodingCore, IRKind, IRNode
from bcnlt.parsers.bundle import MAX_SOURCE_BYTES, PARSE_TIMEOUT_MS, assert_clean_parse_tree
from bcnlt.preprocessing import SemanticPreprocessor
from bcnlt.semantic_record import normalize_semantic_text
from bcnlt.semantic_schema import SEMANTIC_SCHEMA_VERSION
from bcnlt.uncertainty import KnowledgeBoundaryAssessment

from .carousel import LanguageCartridge
from .coverage import SemanticCoverage, SemanticExtraction, is_unresolved_unknown
from .spec import IdentifierRoleRule, NodeRule, ScopeRule
from .symbols import LexicalSymbolTable


ParserProvider = Callable[[str], Any]
_OWNED_GRAMMARS = frozenset(
    {
        "bash", "c", "c_sharp", "cpp", "go", "java", "javascript",
        "kotlin", "php", "python", "ruby", "rust", "swift", "typescript",
    }
)
_LOADED_GRAMMAR_LIBRARIES: dict[str, object] = {}
MAX_SEMANTIC_SURFACE_CHARS = 4096
MAX_HISTORY_SURFACE_CHARS = 512
MAX_IR_NAME_CHARS = 8192


def _default_parser_provider(grammar_id: str) -> Any:
    try:
        from tree_sitter import Language, Parser
    except ImportError as exc:  # pragma: no cover - exercised by deployment preflight
        raise ValueError("Install the pinned BCNLT multilingual parser dependencies") from exc
    if grammar_id not in _OWNED_GRAMMARS:
        raise ValueError("grammar is not in the owned offline bundle")
    configured_cache = os.environ.get("BCNLT_PARSER_CACHE")
    if not configured_cache:
        raise ValueError("owned grammar directory is not configured")
    directory = Path(configured_cache).expanduser().resolve()
    library = (directory / f"tree_sitter_{grammar_id}.dll").resolve()
    if not library.is_relative_to(directory) or not library.is_file():
        raise ValueError("owned grammar missing; downloads and external fallbacks disabled")
    dll = ctypes.CDLL(str(library))
    _LOADED_GRAMMAR_LIBRARIES[grammar_id] = dll
    factory = getattr(dll, f"tree_sitter_{grammar_id}")
    factory.restype = ctypes.c_void_p
    factory.argtypes = []
    pointer = factory()
    if not pointer:
        raise ValueError("owned grammar returned a null language")
    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype = ctypes.py_object
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = capsule_new(pointer, b"tree_sitter.Language", None)
    return Parser(Language(capsule))


class TreeSitterSemanticEngine:
    """Shared deterministic semantic extractor for every language cartridge."""

    def __init__(
        self,
        *,
        parser_provider: ParserProvider | None = None,
        wildcard_memory_root: str | Path | None = None,
        wildcard_memory_read_only: bool = False,
    ) -> None:
        if not isinstance(wildcard_memory_read_only, bool):
            raise TypeError("wildcard_memory_read_only must be a boolean")
        if wildcard_memory_read_only and wildcard_memory_root is None:
            raise ValueError("read-only wildcard memory requires an explicit root")
        self._parser_provider = parser_provider or _default_parser_provider
        self._wildcard_memory_root = (
            Path(wildcard_memory_root) if wildcard_memory_root is not None else None
        )
        self._core = CommonCodingCore()
        self._preprocessor = SemanticPreprocessor(
            memory_root=self._wildcard_memory_root,
            memory_read_only=wildcard_memory_read_only,
        )

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
        pattern_owned_nodes: set[tuple[int, int, str]] = set()
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
            if name == "":
                return node
            if name is None:
                return None
            # A directly bound JS/TS anonymous function owns the variable's
            # lexical name. Never infer names for callbacks, properties, or
            # destructuring: those require a separate alias-analysis contract.
            if (name == "name" and document.language in {"javascript", "typescript"}
                    and str(node.type) == "arrow_function"
                    and node.child_by_field_name("name") is None):
                parent = getattr(node, "parent", None)
                if parent is not None and str(parent.type) == "variable_declarator":
                    value = parent.child_by_field_name("value")
                    binding = parent.child_by_field_name("name")
                    if (value is not None and node_key(value) == node_key(node)
                            and binding is not None and str(binding.type) == "identifier"):
                        return binding
            for alternative in name.split("|"):
                current = node
                for segment in alternative.split("."):
                    if segment.startswith("#"):
                        try:
                            index = int(segment[1:])
                        except ValueError:
                            current = None
                            break
                        named_children = [
                            child
                            for child in current.children
                            if bool(getattr(child, "is_named", False))
                        ]
                        if index < 0 or index >= len(named_children):
                            current = None
                            break
                        current = named_children[index]
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
            unit_metadata = dict(metadata or {})
            if len(surface) > MAX_SEMANTIC_SURFACE_CHARS:
                surface = surface[:MAX_SEMANTIC_SURFACE_CHARS]
                unit_metadata["surface_truncated"] = True
            if ir is not None and ir.name is not None and len(ir.name) > MAX_IR_NAME_CHARS:
                ir = replace(ir, name=ir.name[:MAX_IR_NAME_CHARS])
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
                    metadata=unit_metadata,
                )
            )

        def claim_pattern_subtree(node: Any) -> None:
            """Keep binding syntax out of reference/unknown extraction."""
            key = node_key(node)
            pattern_owned_nodes.add(key)
            definition_owned_nodes.add(key)
            if bool(getattr(node, "is_named", False)):
                mapped_nodes.add(key)
            for child in node.children:
                claim_pattern_subtree(child)

        def bind_javascript_pattern(
            binding: Any,
            *,
            relation: str,
            hoist_var: bool = False,
        ) -> None:
            """Define the local names owned by a JS/TS binding pattern.

            Property keys are not bindings. Default-value expressions are
            evaluated before the binding they initialize, while bindings that
            occur earlier in the pattern are already visible to later defaults.
            """
            binding_type = str(binding.type)
            key = node_key(binding)
            pattern_owned_nodes.add(key)
            definition_owned_nodes.add(key)
            if bool(getattr(binding, "is_named", False)):
                mapped_nodes.add(key)

            if binding_type in {"identifier", "shorthand_property_identifier_pattern"}:
                name = text(binding)
                binding_span = span(binding)
                binding_scope = symbols.scope
                if hoist_var:
                    symbol_id, binding_scope = symbols.define_in_enclosing(
                        name,
                        binding_span,
                        kinds=frozenset({"function", "module"}),
                    )
                else:
                    symbol_id = symbols.define(name, binding_span)
                document.index.define(
                    document.path,
                    symbol_id,
                    name,
                    binding_span,
                )
                emit(
                    surface=name,
                    event="DEFINE",
                    unit_span=binding_span,
                    relation=relation,
                    symbol_id=symbol_id,
                    ir=self._core.variable(name),
                    scope=binding_scope,
                )
                return

            if binding_type in {"assignment_pattern", "object_assignment_pattern"}:
                right = binding.child_by_field_name("right")
                if right is not None:
                    walk(right)
                left = binding.child_by_field_name("left")
                if left is not None:
                    bind_javascript_pattern(
                        left,
                        relation=relation,
                        hoist_var=hoist_var,
                    )
                return

            if binding_type == "pair_pattern":
                property_node = binding.child_by_field_name("key")
                value = binding.child_by_field_name("value")
                if property_node is not None:
                    if str(property_node.type) == "computed_property_name":
                        walk(property_node)
                    else:
                        claim_pattern_subtree(property_node)
                if value is not None:
                    bind_javascript_pattern(
                        value,
                        relation=relation,
                        hoist_var=hoist_var,
                    )
                return

            if binding_type == "rest_pattern":
                children = [
                    child
                    for child in binding.named_children
                    if str(child.type) not in {"comment"}
                ]
                if children:
                    bind_javascript_pattern(
                        children[-1],
                        relation=relation,
                        hoist_var=hoist_var,
                    )
                return

            if binding_type in {"object_pattern", "array_pattern"}:
                for child in binding.named_children:
                    child_type = str(child.type)
                    if child_type in {
                        "identifier",
                        "shorthand_property_identifier_pattern",
                        "assignment_pattern",
                        "object_assignment_pattern",
                        "pair_pattern",
                        "rest_pattern",
                        "object_pattern",
                        "array_pattern",
                    }:
                        bind_javascript_pattern(
                            child,
                            relation=relation,
                            hoist_var=hoist_var,
                        )
                    else:
                        claim_pattern_subtree(child)
                return

            # Unknown pattern wrappers must remain explicit unknowns rather
            # than being silently promoted to definitions.
            pattern_owned_nodes.discard(key)
            definition_owned_nodes.discard(key)
            mapped_nodes.discard(key)

        def walk(node: Any) -> None:
            nonlocal total_named_nodes, scope_enters, scope_exits
            key = node_key(node)
            is_named = bool(getattr(node, "is_named", False))
            if is_named:
                total_named_nodes += 1
                named_nodes.add(key)
                if int(node.start_byte) == int(node.end_byte):
                    # Some grammars expose zero-width named bookkeeping nodes
                    # (for example Bash regex and heredoc-content sentinels).
                    # They contain no source evidence and cannot form a valid
                    # semantic surface, so exclude them rather than inventing
                    # training content or rejecting the complete document.
                    excluded_nodes.add(key)
                    return
            if key in pattern_owned_nodes:
                return

            candidate_role_rules = role_rules.get(str(node.type), ())
            parent = getattr(node, "parent", None)
            parent_type = str(parent.type) if parent is not None else None
            contextual_role_rules = tuple(
                rule for rule in candidate_role_rules if rule.parent_type == parent_type
                and rule.parent_type is not None
            )
            node_role_rules = contextual_role_rules or tuple(
                rule for rule in candidate_role_rules if rule.parent_type is None
            )
            prewalked_child_keys: set[tuple[int, int, str]] = set()
            js_pattern_definition = False

            if document.language in {"javascript", "typescript"}:
                node_type = str(node.type)
                if node_type in {"import_clause", "named_imports"}:
                    mapped_nodes.add(key)
                if node_type == "import_clause":
                    for imported in node.named_children:
                        if str(imported.type) == "identifier":
                            bind_javascript_pattern(
                                imported,
                                relation="import_clause.binding",
                            )
                if node_type == "namespace_import":
                    local = next(
                        (
                            child
                            for child in node.named_children
                            if str(child.type) == "identifier"
                        ),
                        None,
                    )
                    if local is not None:
                        bind_javascript_pattern(
                            local,
                            relation="namespace_import.binding",
                        )
                        mapped_nodes.add(key)
                if node_type == "import_specifier":
                    imported = node.child_by_field_name("name")
                    alias = node.child_by_field_name("alias")
                    local = alias or imported
                    if alias is not None and imported is not None:
                        claim_pattern_subtree(imported)
                    if local is not None:
                        bind_javascript_pattern(
                            local,
                            relation="import_specifier.binding",
                        )
                        mapped_nodes.add(key)
                if (
                    parent_type == "formal_parameters"
                    and node_type
                    in {"object_pattern", "array_pattern", "rest_pattern", "assignment_pattern"}
                ):
                    bind_javascript_pattern(
                        node,
                        relation="formal_parameters.binding",
                    )
                    return

                if node_type == "variable_declarator":
                    binding = node.child_by_field_name("name")
                    if binding is not None and str(binding.type) in {
                        "object_pattern",
                        "array_pattern",
                    }:
                        initializer = node.child_by_field_name("value")
                        if initializer is not None:
                            walk(initializer)
                            prewalked_child_keys.add(node_key(initializer))
                        parent_text = text(parent).lstrip() if parent is not None else ""
                        bind_javascript_pattern(
                            binding,
                            relation="variable_declarator.binding",
                            hoist_var=parent_text.startswith("var "),
                        )
                        prewalked_child_keys.add(node_key(binding))
                        mapped_nodes.add(key)
                        node_role_rules = tuple(
                            rule for rule in node_role_rules if rule.role != "definition"
                        )
                        js_pattern_definition = True

                if node_type in {"required_parameter", "optional_parameter"}:
                    binding = node.child_by_field_name("pattern")
                    if binding is not None and str(binding.type) in {
                        "object_pattern",
                        "array_pattern",
                        "rest_pattern",
                        "assignment_pattern",
                    }:
                        default = node.child_by_field_name("value")
                        if default is not None:
                            walk(default)
                            prewalked_child_keys.add(node_key(default))
                        bind_javascript_pattern(
                            binding,
                            relation=f"{node_type}.binding",
                        )
                        prewalked_child_keys.add(node_key(binding))
                        mapped_nodes.add(key)
                        node_role_rules = tuple(
                            rule for rule in node_role_rules if rule.role != "definition"
                        )
                        js_pattern_definition = True
            roles: dict[str, tuple[str, str | None, Any, str, str]] = {}
            # Go short declarations bind each LHS identifier, not the
            # expression-list spelling "a, b". Existing same-frame names are
            # assignments; outer-frame names are newly shadowed. `_` never binds.
            go_short = document.language == "go" and str(node.type) == "short_var_declaration"
            go_right = node.child_by_field_name("right") if go_short else None
            if go_right is not None:
                # Initializers see the old environment, not newly declared names.
                walk(go_right)
            if go_short:
                left = node.child_by_field_name("left")
                if left is not None:
                    mapped_nodes.add(node_key(left))
                    for binding in left.named_children:
                        if str(binding.type) != "identifier":
                            continue
                        definition_owned_nodes.add(node_key(binding))
                        mapped_nodes.add(node_key(binding))
                        name = text(binding)
                        if name == "_":
                            continue
                        symbol_id = symbols.resolve_current(name)
                        event = "REFERENCE" if symbol_id is not None else "DEFINE"
                        if symbol_id is None:
                            symbol_id = symbols.define(name, span(binding))
                            document.index.define(document.path, symbol_id, name, span(binding))
                        else:
                            document.index.reference(document.path, symbol_id, name, span(binding))
                        emit(surface=name, event=event, unit_span=span(binding),
                             relation="short_var_declaration.binding", symbol_id=symbol_id,
                             defines=symbol_id if event == "REFERENCE" else None,
                             ir=self._core.variable(name))
                node_role_rules = tuple(r for r in node_role_rules if r.role != "definition")
            for role_rule in node_role_rules:
                child = field(node, role_rule.field_name)
                if child is None:
                    continue
                child_key = node_key(child)
                if role_rule.role == "reference" and child_key in definition_owned_nodes:
                    continue
                if bool(getattr(child, "is_named", False)):
                    mapped_nodes.add(child_key)
                name = text(child)
                child_span = span(child)
                if role_rule.role == "definition":
                    definition_owned_nodes.add(child_key)
                    binding_scope = symbols.scope
                    parent = getattr(node, "parent", None)
                    is_javascript_var = (
                        document.language in {"javascript", "typescript"}
                        and str(node.type) == "variable_declarator"
                        and parent is not None
                        and str(parent.type) == "variable_declaration"
                        and text(parent).lstrip().startswith("var ")
                    )
                    if is_javascript_var:
                        symbol_id, binding_scope = symbols.define_in_enclosing(
                            name,
                            child_span,
                            kinds=frozenset({"function", "module"}),
                        )
                    else:
                        symbol_id = symbols.define(name, child_span)
                    document.index.define(document.path, symbol_id, name, child_span)
                    roles[role_rule.field_name] = (
                        name,
                        symbol_id,
                        child,
                        binding_scope,
                        role_rule.role,
                    )
                elif role_rule.role == "reference":
                    symbol_id = symbols.resolve(name)
                    document.index.reference(document.path, symbol_id, name, child_span)
                    roles[role_rule.field_name] = (
                        name,
                        symbol_id,
                        child,
                        symbols.scope,
                        role_rule.role,
                    )
                else:
                    raise ValueError(f"Unsupported identifier role: {role_rule.role}")

            rules = node_rules.get(str(node.type), ())
            matching_rules: list[tuple[NodeRule, Any | None, str]] = []
            for rule in rules:
                if go_short and rule.event == "DEFINE":
                    continue
                if js_pattern_definition and rule.event == "DEFINE":
                    continue
                surface_node = field(node, rule.surface_field)
                if (
                    rule.event == "DEFINE"
                    and rule.surface_field is not None
                    and surface_node is None
                ):
                    # A field-qualified definition does not own the entire node
                    # when that field is absent (for example C's unnamed
                    # ``void`` parameter). Falling back invents a symbol. Other
                    # semantic events may legitimately describe the whole node.
                    continue
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
                if surface_node is not None:
                    # A field path can descend through grammar-only wrappers.  The
                    # semantic event owns that complete selection path; otherwise
                    # wrappers such as Swift/Kotlin ``navigation_suffix`` or C++
                    # declarators are emitted again as unknown semantics.  Nodes on
                    # the path may still emit their own explicit rules when walked.
                    owned_surface_node = surface_node
                    while owned_surface_node is not None and node_key(owned_surface_node) != key:
                        surface_key = node_key(owned_surface_node)
                        mapped_nodes.add(surface_key)
                        if rule.event == "MEMBER_ACCESS":
                            definition_owned_nodes.add(surface_key)
                        owned_surface_node = owned_surface_node.parent
                role = roles.get(rule.surface_field or "")
                symbol_id = role[1] if role else None
                defines = symbol_id if role and role[4] == "reference" else None
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
                    scope=(
                        role[3]
                        if role
                        and role[4] == "definition"
                        else None
                    ),
                )

            for role_rule in node_role_rules:
                if any(
                    (rule.surface_field or "") == role_rule.field_name
                    for rule, _, _ in matching_rules
                ):
                    continue
                role = roles.get(role_rule.field_name)
                if role is None:
                    continue
                name, symbol_id, child, binding_scope, _ = role
                event = "DEFINE" if role_rule.role == "definition" else "REFERENCE"
                relation_role = role_rule.field_name
                if not relation_role:
                    relation_role = (
                        "parameter"
                        if role_rule.parent_type is not None
                        and "parameter" in role_rule.parent_type
                        else role_rule.role
                    )
                emit(
                    surface=name,
                    event=event,
                    unit_span=span(child),
                    relation=f"{node.type}.{relation_role}",
                    symbol_id=symbol_id,
                    defines=symbol_id if role_rule.role == "reference" else None,
                    ir=self._core.variable(name),
                    scope=binding_scope if role_rule.role == "definition" else None,
                )

            active_scope: ScopeRule | None = None
            rules_for_scope = scope_rules.get(str(node.type), ())
            if (document.language == "go" and str(node.type) == "block"
                    and parent_type in {"function_declaration", "method_declaration", "func_literal"}):
                # Parameters and the outer function body share one Go scope.
                rules_for_scope = ()
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

            if (
                document.language in {"javascript", "typescript"}
                and str(node.type) == "catch_clause"
            ):
                parameter = node.child_by_field_name("parameter")
                if parameter is not None and str(parameter.type) in {
                    "identifier",
                    "object_pattern",
                    "array_pattern",
                }:
                    bind_javascript_pattern(
                        parameter,
                        relation="catch_clause.parameter",
                    )
                    prewalked_child_keys.add(node_key(parameter))
                    mapped_nodes.add(key)

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

            if document.language in {"javascript", "typescript"} and str(node.type) == "arrow_function":
                parameter = node.child_by_field_name("parameter")
                if parameter is not None and str(parameter.type) == "identifier":
                    definition_owned_nodes.add(node_key(parameter))
                    mapped_nodes.add(node_key(parameter))
                    name = text(parameter)
                    symbol_id = symbols.define(name, span(parameter))
                    document.index.define(document.path, symbol_id, name, span(parameter))
                    emit(surface=name, event="DEFINE", unit_span=span(parameter),
                         relation="arrow_function.parameter", symbol_id=symbol_id,
                         ir=self._core.variable(name))
                body = node.child_by_field_name("body")
                if body is not None and str(body.type) != "statement_block":
                    emit(surface=text(body), event="RETURN", unit_span=span(body),
                         relation="arrow_function.implicit_return",
                         ir=self._core.extension("javascript.arrow_return", {"syntactic_body": True}))

            # Rust function tails return their final expression without a
            # `return` keyword. Only a direct function-body tail qualifies;
            # semicolon statements and declarations must not become returns.
            if document.language == "rust" and str(node.type) == "block":
                owner = getattr(node, "parent", None)
                body = owner.child_by_field_name("body") if owner is not None else None
                if (owner is not None and str(owner.type) == "function_item"
                        and body is not None and node_key(body) == key):
                    children = [c for c in node.named_children
                                if str(c.type) not in {"line_comment", "block_comment"}]
                    tail = children[-1] if children else None
                    if tail is not None and str(tail.type) == "expression_statement":
                        tail = (None if any(str(c.type) == ";" for c in tail.children)
                                else next(iter(tail.named_children), None))
                    if (tail is not None and str(tail.type) != "return_expression"
                            and (str(tail.type).endswith(("_expression", "_literal"))
                                 or str(tail.type) in {"identifier", "block", "self"})):
                        emit(surface=text(tail), event="RETURN", unit_span=span(tail),
                             relation="function_item.implicit_tail",
                             ir=self._core.extension("rust.implicit_return", {"syntactic_tail": True}))

            for child in node.children:
                if go_right is not None and node_key(child) == node_key(go_right):
                    continue
                if node_key(child) in prewalked_child_keys:
                    continue
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
            history.append(
                normalize_semantic_text(unit.surface)[:MAX_HISTORY_SURFACE_CHARS]
            )

        assessed_units, assessments = self._assess(document, cartridge, units)
        common_core_units = sum(
            1
            for unit in assessed_units
            if unit.ir is not None and unit.ir.kind is not IRKind.EXTENSION
        )
        unknown_units = sum(is_unresolved_unknown(unit) for unit in assessed_units)
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
        if rule.field_name == (field_name or ""):
            return rule.role
    return None
