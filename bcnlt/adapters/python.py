from __future__ import annotations

import ast
from collections import deque
from collections.abc import Iterable
from typing import Any

from bcnlt.cache import ProjectIndex
from bcnlt.core import ContextualUnit, SourceDocument, Span
from bcnlt.events import SemanticEvent
from bcnlt.ir import IRKind, IRNode

from .base import LanguageAdapter


class PythonAdapter(LanguageAdapter):
    language = "python"

    def parse(self, source: str, path: str = "<memory>") -> SourceDocument:
        tree = ast.parse(source, filename=path)
        return SourceDocument(path=path, source=source, tree=tree, index=ProjectIndex())

    def to_units(self, document: SourceDocument) -> Iterable[ContextualUnit]:
        visitor = _PythonUnitVisitor(self, document)
        return visitor.collect()

    def render(self, node: IRNode) -> str:
        renderer = _PythonIRRenderer()
        code = renderer.render(node)
        ast.parse(code)
        compile(code, "<bcnlt-rendered>", "exec")
        return code


class _PythonUnitVisitor(ast.NodeVisitor):
    def __init__(self, adapter: PythonAdapter, document: SourceDocument) -> None:
        self.adapter = adapter
        self.document = document
        self.units: list[ContextualUnit] = []
        self.history: deque[str] = deque(maxlen=8)
        self.scopes: list[str] = ["module"]
        self.scope_kinds: list[str] = ["module"]
        self.scope_defs: list[dict[str, str]] = [{}]
        self.dynamic_symbols: set[str] = set()

    def collect(self) -> list[ContextualUnit]:
        self.visit(self.document.tree)
        return self.units

    def generic_visit(self, node: ast.AST) -> Any:
        if isinstance(node, (ast.stmt, ast.expr)):
            self._emit_unknown(node)
        return super().generic_visit(node)

    @property
    def scope(self) -> str:
        return ".".join(self.scopes)

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            symbol_id = self._define(name, node)
            self._emit(
                alias.name,
                "IMPORT",
                node,
                symbol_id=symbol_id,
                ast_relation="Import.names",
                ir=self.adapter.common_core.import_name(alias.name),
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        module = "." * node.level + (node.module or "")
        for alias in node.names:
            name = alias.asname or alias.name
            symbol_id = self._define(name, node)
            self._emit(
                f"{module}.{alias.name}".strip("."),
                "IMPORT",
                node,
                symbol_id=symbol_id,
                ast_relation="ImportFrom.names",
                ir=self.adapter.common_core.import_name(alias.name),
            )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._visit_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._visit_function(node, is_async=True)

    def _visit_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, *, is_async: bool
    ) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self._visit_argument_expressions(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        symbol_id = self._define(node.name, node)
        if is_async:
            self._emit(
                node.name,
                SemanticEvent.ASYNC,
                node,
                symbol_id=symbol_id,
                ast_relation="AsyncFunctionDef",
                ir=self.adapter.common_core.extension("python.async_function"),
            )
        self._emit(
            node.name,
            SemanticEvent.DEFINE,
            node,
            symbol_id=symbol_id,
            ast_relation=f"{node.__class__.__name__}.name",
            ir=self.adapter.common_core.function(node.name, [], []),
        )
        self._emit(
            node.name,
            SemanticEvent.SCOPE_ENTER,
            node,
            symbol_id=symbol_id,
            ast_relation=node.__class__.__name__,
        )
        self.scopes.append(node.name)
        self.scope_kinds.append("function")
        self.scope_defs.append({})
        try:
            self._define_arguments(node.args)
            for stmt in node.body:
                self.visit(stmt)
        finally:
            self.scope_defs.pop()
            self.scope_kinds.pop()
            self.scopes.pop()
            self._emit(
                node.name,
                SemanticEvent.SCOPE_EXIT,
                node,
                symbol_id=symbol_id,
                ast_relation=node.__class__.__name__,
            )

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        symbol_id = self._define(node.name, node)
        self._emit(
            node.name,
            SemanticEvent.CLASS,
            node,
            symbol_id=symbol_id,
            ast_relation="ClassDef.name",
            ir=self.adapter.common_core.extension("python.class", {"name": node.name}),
        )
        self._emit(
            node.name,
            SemanticEvent.SCOPE_ENTER,
            node,
            symbol_id=symbol_id,
            ast_relation="ClassDef",
        )
        self.scopes.append(node.name)
        self.scope_kinds.append("class")
        self.scope_defs.append({})
        try:
            for stmt in node.body:
                self.visit(stmt)
        finally:
            self.scope_defs.pop()
            self.scope_kinds.pop()
            self.scopes.pop()
            self._emit(
                node.name,
                SemanticEvent.SCOPE_EXIT,
                node,
                symbol_id=symbol_id,
                ast_relation="ClassDef",
            )

    def visit_Lambda(self, node: ast.Lambda) -> Any:
        self._visit_argument_expressions(node.args)
        name = self._anonymous_scope_name("lambda", node)
        self._emit(name, SemanticEvent.SCOPE_ENTER, node, ast_relation="Lambda")
        self.scopes.append(name)
        self.scope_kinds.append("lambda")
        self.scope_defs.append({})
        try:
            self._define_arguments(node.args)
            self.visit(node.body)
        finally:
            self.scope_defs.pop()
            self.scope_kinds.pop()
            self.scopes.pop()
            self._emit(name, SemanticEvent.SCOPE_EXIT, node, ast_relation="Lambda")

    def visit_Expr(self, node: ast.Expr) -> Any:
        self.visit(node.value)

    def visit_Assign(self, node: ast.Assign) -> Any:
        self.visit(node.value)
        for target in node.targets:
            self._define_target(
                target,
                "Assign.targets",
                dynamic_binding=self._expr_is_dynamic_call(node.value),
            )
        self._emit(
            self._surface(node),
            "ASSIGN",
            node,
            ast_relation="Assign",
            ir=self.adapter.common_core.assignment(self._target_name(node.targets[0])),
        )

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        if node.value:
            self.visit(node.value)
        self._define_target(node.target, "AnnAssign.target")
        self._emit(
            self._surface(node),
            "ASSIGN",
            node,
            ast_relation="AnnAssign",
            ir=self.adapter.common_core.assignment(self._target_name(node.target)),
        )

    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        self.visit(node.target)
        self._emit(self._operator_name(node.op), "OPERATOR", node, ast_relation="AugAssign.op")
        self.visit(node.value)
        self._emit(
            self._surface(node),
            "ASSIGN",
            node,
            ast_relation="AugAssign",
            ir=self.adapter.common_core.assignment(self._target_name(node.target)),
        )

    def visit_For(self, node: ast.For) -> Any:
        self.visit(node.iter)
        self._define_target(node.target, "For.target")
        self._emit(
            self._target_name(node.target),
            "LOOP",
            node,
            ast_relation="For",
            ir=self.adapter.common_core.loop(self._target_name(node.target)),
        )
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> Any:
        self._emit(
            "for",
            SemanticEvent.ASYNC,
            node,
            ast_relation="AsyncFor",
            ir=self.adapter.common_core.extension("python.async_for"),
        )
        self.visit(node.iter)
        self._define_target(node.target, "AsyncFor.target")
        self._emit(
            self._target_name(node.target),
            SemanticEvent.LOOP,
            node,
            ast_relation="AsyncFor",
            ir=self.adapter.common_core.loop(self._target_name(node.target)),
        )
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_While(self, node: ast.While) -> Any:
        self.visit(node.test)
        self._emit("while", "LOOP", node, ast_relation="While", ir=self.adapter.common_core.loop())
        for stmt in node.body:
            self.visit(stmt)

    def visit_If(self, node: ast.If) -> Any:
        self.visit(node.test)
        self._emit("if", "BRANCH", node, ast_relation="If", ir=self.adapter.common_core.branch())
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_Return(self, node: ast.Return) -> Any:
        if node.value:
            self.visit(node.value)
        self._emit("return", "RETURN", node, ast_relation="Return", ir=self.adapter.common_core.return_value())

    def visit_With(self, node: ast.With) -> Any:
        self._visit_with(node, is_async=False)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> Any:
        self._visit_with(node, is_async=True)

    def _visit_with(self, node: ast.With | ast.AsyncWith, *, is_async: bool) -> None:
        surface = "async with" if is_async else "with"
        if is_async:
            self._emit(surface, SemanticEvent.ASYNC, node, ast_relation="AsyncWith")
        self._emit(
            surface,
            SemanticEvent.CONTEXT_ENTER,
            node,
            ast_relation=node.__class__.__name__,
            ir=self.adapter.common_core.extension("python.context_manager"),
        )
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._define_target(item.optional_vars, f"{node.__class__.__name__}.optional_vars")
        for stmt in node.body:
            self.visit(stmt)
        self._emit(
            surface,
            SemanticEvent.CONTEXT_EXIT,
            node,
            ast_relation=node.__class__.__name__,
            ir=self.adapter.common_core.extension("python.context_manager"),
        )

    def visit_Try(self, node: ast.Try) -> Any:
        self._visit_try(node)

    def visit_TryStar(self, node: ast.TryStar) -> Any:
        self._visit_try(node)

    def _visit_try(self, node: ast.Try | ast.TryStar) -> None:
        self._emit(
            "try",
            SemanticEvent.TRY,
            node,
            ast_relation=node.__class__.__name__,
            ir=self.adapter.common_core.extension("python.try"),
        )
        for stmt in node.body:
            self.visit(stmt)
        for handler in node.handlers:
            self._emit(
                self._surface(handler.type) if handler.type is not None else "except",
                SemanticEvent.EXCEPT,
                handler,
                ast_relation="ExceptHandler",
                ir=self.adapter.common_core.extension("python.except"),
            )
            if handler.type is not None:
                self.visit(handler.type)
            if handler.name:
                symbol_id = self._define(handler.name, handler)
                self._emit(
                    handler.name,
                    SemanticEvent.DEFINE,
                    handler,
                    symbol_id=symbol_id,
                    ast_relation="ExceptHandler.name",
                    ir=self.adapter.common_core.variable(handler.name),
                )
            for stmt in handler.body:
                self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)
        if node.finalbody:
            self._emit(
                "finally",
                SemanticEvent.FINALLY,
                node,
                ast_relation=f"{node.__class__.__name__}.finalbody",
                ir=self.adapter.common_core.extension("python.finally"),
            )
            for stmt in node.finalbody:
                self.visit(stmt)

    def visit_Raise(self, node: ast.Raise) -> Any:
        self._emit(
            "raise",
            SemanticEvent.RAISE,
            node,
            ast_relation="Raise",
            ir=IRNode(IRKind.ERROR, name="raise"),
        )
        if node.exc is not None:
            self.visit(node.exc)
        if node.cause is not None:
            self.visit(node.cause)

    def visit_Yield(self, node: ast.Yield) -> Any:
        self._emit(
            "yield",
            SemanticEvent.YIELD,
            node,
            ast_relation="Yield",
            ir=self.adapter.common_core.extension("python.yield"),
        )
        if node.value is not None:
            self.visit(node.value)

    def visit_YieldFrom(self, node: ast.YieldFrom) -> Any:
        self._emit(
            "yield from",
            SemanticEvent.YIELD,
            node,
            ast_relation="YieldFrom",
            ir=self.adapter.common_core.extension("python.yield_from"),
        )
        self.visit(node.value)

    def visit_Await(self, node: ast.Await) -> Any:
        self._emit(
            "await",
            SemanticEvent.AWAIT,
            node,
            ast_relation="Await",
            ir=self.adapter.common_core.extension("python.await"),
        )
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> Any:
        self.visit(node.func)
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)
        name = self._call_name(node.func)
        definition = self._resolve(name) if isinstance(node.func, ast.Name) else None
        metadata = {
            "dynamic_call": name in {"getattr", "eval", "exec", "__import__"},
            "dynamic_binding": definition in self.dynamic_symbols if definition else False,
        }
        self._emit(
            name,
            "CALL",
            node,
            defines=definition,
            ast_relation="Call.func",
            ir=self.adapter.common_core.call(name),
            metadata=metadata,
        )

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        self.visit(node.value)
        self._emit(
            node.attr,
            "MEMBER_ACCESS",
            node,
            ast_relation="Attribute.attr",
            ir=self.adapter.common_core.member_access(node.attr),
        )

    def visit_Name(self, node: ast.Name) -> Any:
        if isinstance(node.ctx, ast.Store):
            self._define_target(node, "Name.store")
            return
        definition = self._resolve(node.id)
        self.document.index.reference(self.document.path, definition, node.id, self._span(node))
        metadata = {"dynamic_binding": definition in self.dynamic_symbols} if definition else {}
        self._emit(
            node.id,
            "REFERENCE",
            node,
            defines=definition,
            ast_relation="Name.load",
            ir=self.adapter.common_core.variable(node.id),
            metadata=metadata,
        )

    def visit_Constant(self, node: ast.Constant) -> Any:
        self._emit(
            repr(node.value),
            "LITERAL",
            node,
            ast_relation="Constant",
            ir=self.adapter.common_core.literal(node.value),
        )

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        self.visit(node.left)
        self._emit(
            self._operator_name(node.op),
            "OPERATOR",
            node,
            ast_relation="BinOp.op",
            ir=self.adapter.common_core.operator(self._operator_name(node.op)),
        )
        self.visit(node.right)

    def visit_Compare(self, node: ast.Compare) -> Any:
        self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            self._emit(
                self._operator_name(op),
                "OPERATOR",
                node,
                ast_relation="Compare.ops",
                ir=IRNode(IRKind.COMPARISON, name=self._operator_name(op)),
            )
            self.visit(comparator)

    def visit_List(self, node: ast.List) -> Any:
        for elt in node.elts:
            self.visit(elt)
        self._emit("list", "LITERAL", node, ast_relation="List", ir=IRNode(IRKind.DATA_STRUCTURE, name="list"))

    def visit_Dict(self, node: ast.Dict) -> Any:
        for key in node.keys:
            if key:
                self.visit(key)
        for value in node.values:
            self.visit(value)
        self._emit("dict", "LITERAL", node, ast_relation="Dict", ir=IRNode(IRKind.DATA_STRUCTURE, name="dict"))

    def visit_ListComp(self, node: ast.ListComp) -> Any:
        self._visit_comprehension(node, (node.elt,))

    def visit_SetComp(self, node: ast.SetComp) -> Any:
        self._visit_comprehension(node, (node.elt,))

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> Any:
        self._visit_comprehension(node, (node.elt,))

    def visit_DictComp(self, node: ast.DictComp) -> Any:
        self._visit_comprehension(node, (node.key, node.value))

    def _visit_comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp,
        result_nodes: tuple[ast.AST, ...],
    ) -> None:
        name = self._anonymous_scope_name("comprehension", node)
        self._emit(
            self._surface(node),
            SemanticEvent.COMPREHENSION,
            node,
            ast_relation=node.__class__.__name__,
            ir=self.adapter.common_core.extension(f"python.{node.__class__.__name__.lower()}"),
        )
        self._emit(name, SemanticEvent.SCOPE_ENTER, node, ast_relation=node.__class__.__name__)
        self.scopes.append(name)
        self.scope_kinds.append("comprehension")
        self.scope_defs.append({})
        try:
            for generator in node.generators:
                self.visit(generator.iter)
                self._define_target(generator.target, "comprehension.target")
                for condition in generator.ifs:
                    self.visit(condition)
            for result in result_nodes:
                self.visit(result)
        finally:
            self.scope_defs.pop()
            self.scope_kinds.pop()
            self.scopes.pop()
            self._emit(name, SemanticEvent.SCOPE_EXIT, node, ast_relation=node.__class__.__name__)

    def visit_Match(self, node: ast.Match) -> Any:
        self.visit(node.subject)
        self._emit(
            "match",
            SemanticEvent.MATCH,
            node,
            ast_relation="Match",
            ir=self.adapter.common_core.extension("python.match", {"surface": self._surface(node)}),
        )
        for case in node.cases:
            self._visit_pattern_expressions(case.pattern)
            for name, capture_node in self._pattern_captures(case.pattern):
                symbol_id = self._define(name, capture_node)
                self._emit(
                    name,
                    SemanticEvent.DEFINE,
                    capture_node,
                    symbol_id=symbol_id,
                    ast_relation=f"{capture_node.__class__.__name__}.name",
                    ir=self.adapter.common_core.variable(name),
                )
            if case.guard is not None:
                self.visit(case.guard)
            for stmt in case.body:
                self.visit(stmt)

    def visit_Delete(self, node: ast.Delete) -> Any:
        self._emit_unknown(node)
        for target in node.targets:
            self.visit(target)

    def _define_target(
        self, target: ast.AST, relation: str, dynamic_binding: bool = False
    ) -> str | None:
        if isinstance(target, ast.Name):
            symbol_id = self._define(target.id, target)
            if dynamic_binding:
                self.dynamic_symbols.add(symbol_id)
            self._emit(
                target.id,
                "DEFINE",
                target,
                symbol_id=symbol_id,
                ast_relation=relation,
                ir=self.adapter.common_core.variable(target.id),
                metadata={"dynamic_binding": dynamic_binding} if dynamic_binding else None,
            )
            return symbol_id
        if isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._define_target(elt, relation, dynamic_binding=dynamic_binding)
            return None
        self.visit(target)
        return None

    def _define(self, name: str, node: ast.AST) -> str:
        symbol_id = f"{self.scope}.{name}"
        self.scope_defs[-1][name] = symbol_id
        self.document.index.define(self.document.path, symbol_id, name, self._span(node))
        return symbol_id

    def _resolve(self, name: str) -> str | None:
        inside_function_scope = False
        for definitions, kind in reversed(list(zip(self.scope_defs, self.scope_kinds))):
            if kind in {"function", "lambda", "comprehension"}:
                inside_function_scope = True
            if kind == "class" and inside_function_scope:
                continue
            if name in definitions:
                return definitions[name]
        return None

    def _visit_argument_expressions(self, arguments: ast.arguments) -> None:
        all_arguments = [
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ]
        for argument in all_arguments:
            if argument.annotation is not None:
                self.visit(argument.annotation)
        for argument in (arguments.vararg, arguments.kwarg):
            if argument is not None and argument.annotation is not None:
                self.visit(argument.annotation)
        for default in [
            *arguments.defaults,
            *(item for item in arguments.kw_defaults if item is not None),
        ]:
            self.visit(default)

    def _define_arguments(self, arguments: ast.arguments) -> None:
        positional = [*arguments.posonlyargs, *arguments.args]
        keyword_only = list(arguments.kwonlyargs)
        for arg in [*positional, *keyword_only]:
            param_id = self._define(arg.arg, arg)
            self._emit(
                arg.arg,
                SemanticEvent.DEFINE,
                arg,
                symbol_id=param_id,
                ast_relation="arguments.args",
                ir=self.adapter.common_core.parameter(arg.arg),
            )
        for arg in (arguments.vararg, arguments.kwarg):
            if arg is not None:
                param_id = self._define(arg.arg, arg)
                self._emit(
                    arg.arg,
                    SemanticEvent.DEFINE,
                    arg,
                    symbol_id=param_id,
                    ast_relation="arguments.vararg" if arg is arguments.vararg else "arguments.kwarg",
                    ir=self.adapter.common_core.parameter(arg.arg),
                )

    def _visit_pattern_expressions(self, pattern: ast.pattern) -> None:
        if isinstance(pattern, ast.MatchValue):
            self.visit(pattern.value)
        elif isinstance(pattern, ast.MatchSequence):
            for child in pattern.patterns:
                self._visit_pattern_expressions(child)
        elif isinstance(pattern, ast.MatchMapping):
            for key in pattern.keys:
                self.visit(key)
            for child in pattern.patterns:
                self._visit_pattern_expressions(child)
        elif isinstance(pattern, ast.MatchClass):
            self.visit(pattern.cls)
            for child in [*pattern.patterns, *pattern.kwd_patterns]:
                self._visit_pattern_expressions(child)
        elif isinstance(pattern, ast.MatchAs) and pattern.pattern is not None:
            self._visit_pattern_expressions(pattern.pattern)
        elif isinstance(pattern, ast.MatchOr):
            for child in pattern.patterns:
                self._visit_pattern_expressions(child)

    def _pattern_captures(self, pattern: ast.pattern) -> list[tuple[str, ast.pattern]]:
        captures: list[tuple[str, ast.pattern]] = []

        def collect(current: ast.pattern) -> None:
            if isinstance(current, ast.MatchStar) and current.name is not None:
                captures.append((current.name, current))
            elif isinstance(current, ast.MatchAs):
                if current.pattern is not None:
                    collect(current.pattern)
                if current.name is not None:
                    captures.append((current.name, current))
            elif isinstance(current, ast.MatchSequence):
                for child in current.patterns:
                    collect(child)
            elif isinstance(current, ast.MatchMapping):
                for child in current.patterns:
                    collect(child)
                if current.rest is not None:
                    captures.append((current.rest, current))
            elif isinstance(current, ast.MatchClass):
                for child in [*current.patterns, *current.kwd_patterns]:
                    collect(child)
            elif isinstance(current, ast.MatchOr):
                for child in current.patterns:
                    collect(child)

        collect(pattern)
        unique: list[tuple[str, ast.pattern]] = []
        seen: set[str] = set()
        for name, capture_node in captures:
            if name not in seen:
                unique.append((name, capture_node))
                seen.add(name)
        return unique

    def _emit_unknown(self, node: ast.AST) -> None:
        node_name = node.__class__.__name__
        self._emit(
            self._surface(node),
            SemanticEvent.UNKNOWN_SEMANTIC,
            node,
            ast_relation=node_name,
            ir=self.adapter.common_core.extension("python.unknown", {"node": node_name}),
            metadata={"unknown_node": node_name},
        )

    def _anonymous_scope_name(self, prefix: str, node: ast.AST) -> str:
        return f"{prefix}@{getattr(node, 'lineno', 1)}:{getattr(node, 'col_offset', 0)}"

    def _emit(
        self,
        surface: str,
        event: str | SemanticEvent,
        node: ast.AST,
        *,
        symbol_id: str | None = None,
        defines: str | None = None,
        ast_relation: str | None = None,
        ir: IRNode | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event_value = SemanticEvent(event).value
        unit = ContextualUnit(
            surface=surface,
            event=event_value,
            scope=self.scope,
            symbol_id=symbol_id,
            defines=defines,
            ast_relation=ast_relation,
            span=self._span(node),
            ir=ir,
            recent_history=tuple(self.history),
            metadata=metadata or {},
        )
        self.units.append(unit)
        self.history.append(f"{event_value}:{surface}")

    def _span(self, node: ast.AST) -> Span:
        return (
            getattr(node, "lineno", 1),
            getattr(node, "col_offset", 0),
            getattr(node, "end_lineno", getattr(node, "lineno", 1)),
            getattr(node, "end_col_offset", getattr(node, "col_offset", 0)),
        )

    def _surface(self, node: ast.AST) -> str:
        return ast.get_source_segment(self.document.source, node) or node.__class__.__name__

    def _target_name(self, target: ast.AST) -> str:
        if isinstance(target, ast.Name):
            return target.id
        if isinstance(target, ast.Attribute):
            return target.attr
        return self._surface(target)

    def _call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return self._surface(node)

    def _operator_name(self, op: ast.AST) -> str:
        return op.__class__.__name__

    def _expr_is_dynamic_call(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "eval", "exec", "__import__"}
        )


class _PythonIRRenderer:
    def render(self, node: IRNode) -> str:
        lines = self._render_node(node, indent=0)
        return "\n".join(lines) + "\n"

    def _render_node(self, node: IRNode, indent: int) -> list[str]:
        prefix = " " * indent
        if node.kind == IRKind.MODULE:
            lines: list[str] = []
            for child in node.children:
                lines.extend(self._render_node(child, indent))
            return lines
        if node.kind == IRKind.ASSIGNMENT:
            value = self._expr(node.children[0]) if node.children else "None"
            return [f"{prefix}{node.name} = {value}"]
        if node.kind == IRKind.FUNCTION:
            parameter_count = int(node.metadata.get("parameter_count", 0))
            params = [child.name or "arg" for child in node.children[:parameter_count]]
            body_nodes = node.children[parameter_count:]
            lines = [f"{prefix}def {node.name}({', '.join(params)}):"]
            if not body_nodes:
                lines.append(f"{prefix}    pass")
            for child in body_nodes:
                lines.extend(self._render_node(child, indent + 4))
            return lines
        if node.kind == IRKind.RETURN:
            value = self._expr(node.children[0]) if node.children else "None"
            return [f"{prefix}return {value}"]
        raise ValueError(f"Cannot render IR kind to Python in v0.1: {node.kind}")

    def _expr(self, node: IRNode) -> str:
        if node.kind == IRKind.VALUE:
            return repr(node.value)
        if node.kind == IRKind.VARIABLE:
            return node.name or "_"
        if node.kind == IRKind.CALL:
            args = ", ".join(self._expr(child) for child in node.children)
            return f"{node.name}({args})"
        raise ValueError(f"Cannot render IR expression in v0.1: {node.kind}")
