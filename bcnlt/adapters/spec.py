from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class NodeRule:
    node_type: str
    event: str
    ir_kind: str


@dataclass(frozen=True, slots=True)
class IdentifierRoleRule:
    node_type: str
    field_name: str
    role: str


@dataclass(frozen=True, slots=True)
class ScopeRule:
    node_type: str
    scope_kind: str
    name_field: str | None = None


@dataclass(frozen=True, slots=True)
class UncertaintyRule:
    node_type: str
    signal: str
    metadata_key: str


@dataclass(frozen=True, slots=True)
class LanguageSemanticSpec:
    semantic_spec_version: str
    inherited: tuple["LanguageSemanticSpec", ...] = ()
    node_rules: tuple[NodeRule, ...] = ()
    identifier_roles: tuple[IdentifierRoleRule, ...] = ()
    scope_rules: tuple[ScopeRule, ...] = ()
    uncertainty_rules: tuple[UncertaintyRule, ...] = ()

    @property
    def resolved_node_rules(self) -> tuple[NodeRule, ...]:
        return self._resolved("node_rules")

    @property
    def resolved_identifier_roles(self) -> tuple[IdentifierRoleRule, ...]:
        return self._resolved("identifier_roles")

    @property
    def resolved_scope_rules(self) -> tuple[ScopeRule, ...]:
        return self._resolved("scope_rules")

    @property
    def resolved_uncertainty_rules(self) -> tuple[UncertaintyRule, ...]:
        return self._resolved("uncertainty_rules")

    def _resolved(self, field: str):
        inherited = tuple(
            rule
            for parent in self.inherited
            for rule in getattr(parent, f"resolved_{field}")
        )
        return (*inherited, *getattr(self, field))

    @property
    def canonical_hash(self) -> str:
        payload = {
            "semantic_spec_version": self.semantic_spec_version,
            "node_rules": [asdict(item) for item in self.resolved_node_rules],
            "identifier_roles": [asdict(item) for item in self.resolved_identifier_roles],
            "scope_rules": [asdict(item) for item in self.resolved_scope_rules],
            "uncertainty_rules": [asdict(item) for item in self.resolved_uncertainty_rules],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
