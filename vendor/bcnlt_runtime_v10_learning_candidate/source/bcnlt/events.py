from __future__ import annotations

from enum import Enum


class SemanticEvent(str, Enum):
    IMPORT = "IMPORT"
    DEFINE = "DEFINE"
    REFERENCE = "REFERENCE"
    ASSIGN = "ASSIGN"
    CALL = "CALL"
    RETURN = "RETURN"
    BRANCH = "BRANCH"
    LOOP = "LOOP"
    MEMBER_ACCESS = "MEMBER_ACCESS"
    OPERATOR = "OPERATOR"
    SCOPE_ENTER = "SCOPE_ENTER"
    SCOPE_EXIT = "SCOPE_EXIT"
    LITERAL = "LITERAL"
    UNKNOWN_SEMANTIC = "UNKNOWN_SEMANTIC"
    CLASS = "CLASS"
    ASYNC = "ASYNC"
    CONTEXT_ENTER = "CONTEXT_ENTER"
    CONTEXT_EXIT = "CONTEXT_EXIT"
    TRY = "TRY"
    EXCEPT = "EXCEPT"
    FINALLY = "FINALLY"
    RAISE = "RAISE"
    COMPREHENSION = "COMPREHENSION"
    YIELD = "YIELD"
    AWAIT = "AWAIT"
    MATCH = "MATCH"


CURRENT_EVENT_VALUES = tuple(event.value for event in SemanticEvent)
