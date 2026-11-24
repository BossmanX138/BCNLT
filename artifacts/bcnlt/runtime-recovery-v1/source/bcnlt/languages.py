from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LanguageDefinition:
    canonical_id: str
    grammar_id: str
    accepted_extensions: tuple[str, ...]
    deferred_extensions: tuple[str, ...] = ()
    dialects: tuple[str, ...] = ()
    semantic_spec_version: str = "bcnlt-language-spec-v1"

    def __post_init__(self) -> None:
        if not self.canonical_id or not self.grammar_id:
            raise ValueError("Language definitions require canonical and grammar ids")
        if any(not item.startswith(".") for item in (*self.accepted_extensions, *self.deferred_extensions)):
            raise ValueError("Language extensions must begin with a dot")
        if set(self.accepted_extensions).intersection(self.deferred_extensions):
            raise ValueError("Accepted and deferred extensions must not overlap")


LANGUAGE_DEFINITIONS = (
    LanguageDefinition("python", "python", (".py", ".pyi"), (".pyx",), ("python3",)),
    LanguageDefinition("javascript", "javascript", (".js", ".mjs", ".cjs"), (".jsx",), ("ecmascript",)),
    LanguageDefinition("typescript", "typescript", (".ts", ".mts", ".cts"), (".tsx",), ("typescript",)),
    LanguageDefinition("c", "c", (".c",), (".h",), ("c11", "c17", "c23")),
    LanguageDefinition("cpp", "cpp", (".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx"), (".h",), ("c++17", "c++20", "c++23")),
    LanguageDefinition("c_sharp", "c_sharp", (".cs",), (), ("csharp",)),
    LanguageDefinition("java", "java", (".java",), (), ("java",)),
    LanguageDefinition("go", "go", (".go",), (), ("go",)),
    LanguageDefinition("rust", "rust", (".rs",), (), ("rust",)),
    LanguageDefinition("php", "php", (".php", ".php3", ".php4", ".php5", ".phtml"), (), ("php",)),
    LanguageDefinition("ruby", "ruby", (".rb", ".rake", ".gemspec"), (), ("ruby",)),
    LanguageDefinition("kotlin", "kotlin", (".kt", ".kts"), (), ("kotlin",)),
    LanguageDefinition("swift", "swift", (".swift",), (), ("swift",)),
    LanguageDefinition("bash", "bash", (".sh", ".bash"), (".zsh",), ("bash", "posix_sh")),
)
CANONICAL_LANGUAGE_IDS = tuple(item.canonical_id for item in LANGUAGE_DEFINITIONS)
LANGUAGE_BY_ID = {item.canonical_id: item for item in LANGUAGE_DEFINITIONS}
LANGUAGE_ALIASES = {
    "csharp": "c_sharp",
    "cs": "c_sharp",
    "c++": "cpp",
    "shell": "bash",
    "sh": "bash",
    "js": "javascript",
    "ts": "typescript",
}


def resolve_language_id(value: str) -> str:
    if value in LANGUAGE_BY_ID:
        return value
    try:
        return LANGUAGE_ALIASES[value]
    except KeyError as exc:
        raise ValueError(f"Unknown BCNLT language id or alias: {value}") from exc
