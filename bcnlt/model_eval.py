from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

from .adapters.python import PythonAdapter
from .core import ContextualUnit
from .meaning_eval import MeaningProbe


class ContextAnswerModel(Protocol):
    name: str

    def answer(self, question: str, context: str) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class ModelAnswerResult:
    probe: MeaningProbe
    standard_answer: str
    bcnlt_answer: str
    expected_line: int
    standard_correct: bool
    bcnlt_correct: bool
    standard_context: str
    bcnlt_context: str

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["probe"] = self.probe.to_dict()
        return data


@dataclass(frozen=True)
class ModelEvalReport:
    path: str
    model_name: str
    probe_count: int
    standard_correct_count: int
    bcnlt_correct_count: int
    standard_accuracy: float
    bcnlt_accuracy: float
    results: tuple[ModelAnswerResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "model_name": self.model_name,
            "probe_count": self.probe_count,
            "standard_correct_count": self.standard_correct_count,
            "bcnlt_correct_count": self.bcnlt_correct_count,
            "standard_accuracy": self.standard_accuracy,
            "bcnlt_accuracy": self.bcnlt_accuracy,
            "results": [result.to_dict() for result in self.results],
        }


class OracleDefinitionLineModel:
    """Local deterministic stand-in for an LLM answering only from context."""

    name = "oracle-definition-line"

    def answer(self, question: str, context: str) -> str:
        symbol = _symbol_from_question(question)
        if not symbol:
            return "unknown"
        for line in context.splitlines():
            line_number = _line_number(line)
            if line_number is None:
                continue
            if _bcnlt_line_defines_symbol(line, symbol) or _python_line_defines_symbol(line, symbol):
                return f"line {line_number}"
        return "unknown"


Transport = Callable[[str, dict[str, str], dict[str, object], float], dict[str, object]]


class OpenAICompatibleChatModel:
    """OpenAI-compatible chat-completions context answer model."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        transport: Transport | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport or _urlopen_transport
        self.name = f"openai-compatible:{model}"

    def answer(self, question: str, context: str) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer only from the provided context. "
                        "For definition-line questions, answer in the exact form 'line N'. "
                        "If the context does not contain the answer, answer 'unknown'."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nContext:\n{context}",
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self.transport(
            f"{self.base_url}/chat/completions",
            headers,
            payload,
            self.timeout,
        )
        return _extract_chat_content(response)


class OllamaChatModel:
    """Ollama local chat model using /api/chat."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
        max_output_tokens: int = 64,
        transport: Transport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.transport = transport or _urlopen_transport
        self.name = f"ollama:{model}"

    def answer(self, question: str, context: str) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0, "num_predict": self.max_output_tokens},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Answer only from the provided context. "
                        "For definition-line questions, answer in the exact form 'line N'. "
                        "If the context does not contain the answer, answer 'unknown'."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {question}\n\nContext:\n{context}",
                },
            ],
        }
        response = self.transport(
            f"{self.base_url}/api/chat",
            {"Content-Type": "application/json"},
            payload,
            self.timeout,
        )
        return _extract_ollama_content(response)


def evaluate_model_source(
    source: str,
    path: str = "<memory>",
    model: ContextAnswerModel | None = None,
    max_probes: int = 50,
    standard_context_lines: int = 1,
    bcnlt_context_units: int = 2,
) -> ModelEvalReport:
    model = model or OracleDefinitionLineModel()
    adapter = PythonAdapter()
    document = adapter.parse(source, path=path)
    units = list(adapter.to_units(document))
    probes = _generate_definition_probes(units, max_probes=max_probes)
    results = tuple(
        _evaluate_answer_probe(
            source,
            units,
            probe,
            model,
            standard_context_lines=standard_context_lines,
            bcnlt_context_units=bcnlt_context_units,
        )
        for probe in probes
    )
    standard_correct = sum(1 for result in results if result.standard_correct)
    bcnlt_correct = sum(1 for result in results if result.bcnlt_correct)
    probe_count = len(results)
    return ModelEvalReport(
        path=path,
        model_name=model.name,
        probe_count=probe_count,
        standard_correct_count=standard_correct,
        bcnlt_correct_count=bcnlt_correct,
        standard_accuracy=(standard_correct / probe_count) if probe_count else 0.0,
        bcnlt_accuracy=(bcnlt_correct / probe_count) if probe_count else 0.0,
        results=results,
    )


def evaluate_model_file(
    path: str | Path,
    max_probes: int = 50,
    model: ContextAnswerModel | None = None,
) -> ModelEvalReport:
    file_path = Path(path)
    source = file_path.read_text(encoding="utf-8-sig")
    return evaluate_model_source(source, path=str(file_path), max_probes=max_probes, model=model)


def _generate_definition_probes(
    units: list[ContextualUnit],
    max_probes: int,
) -> list[MeaningProbe]:
    probes: list[MeaningProbe] = []
    for unit in units:
        if unit.event != "DEFINE" or not unit.symbol_id:
            continue
        if unit.ast_relation == "arguments.args":
            continue
        probes.append(
            MeaningProbe(
                question=f"What line defines `{unit.surface}` in scope `{unit.scope}`?",
                surface=unit.surface,
                symbol_id=unit.symbol_id,
                target_line=unit.span[0],
                target_event=unit.event,
                scope=unit.scope,
            )
        )
        if len(probes) >= max_probes:
            break
    return probes


def _evaluate_answer_probe(
    source: str,
    units: list[ContextualUnit],
    probe: MeaningProbe,
    model: ContextAnswerModel,
    standard_context_lines: int,
    bcnlt_context_units: int,
) -> ModelAnswerResult:
    standard_context = _standard_context(source, probe, standard_context_lines)
    bcnlt_context = _bcnlt_context(units, probe, bcnlt_context_units)
    standard_answer = model.answer(probe.question, standard_context)
    bcnlt_answer = model.answer(probe.question, bcnlt_context)
    return ModelAnswerResult(
        probe=probe,
        standard_answer=standard_answer,
        bcnlt_answer=bcnlt_answer,
        expected_line=probe.target_line,
        standard_correct=_answer_matches_line(standard_answer, probe.target_line),
        bcnlt_correct=_answer_matches_line(bcnlt_answer, probe.target_line),
        standard_context=standard_context,
        bcnlt_context=bcnlt_context,
    )


def _standard_context(source: str, probe: MeaningProbe, context_lines: int) -> str:
    lines = source.splitlines()
    match = None
    for index, line in enumerate(lines, start=1):
        if probe.surface in line:
            match = index
            break
    if match is None:
        return ""
    start = max(1, match - context_lines)
    end = min(len(lines), match + context_lines)
    return "\n".join(f"L{line_no}: {lines[line_no - 1]}" for line_no in range(start, end + 1))


def _bcnlt_context(
    units: list[ContextualUnit],
    probe: MeaningProbe,
    context_units: int,
) -> str:
    for index, unit in enumerate(units):
        if (
            unit.event == probe.target_event
            and unit.symbol_id == probe.symbol_id
            and unit.span[0] == probe.target_line
        ):
            start = max(0, index - context_units)
            end = min(len(units), index + context_units + 1)
            return "\n".join(_format_unit_context(item) for item in units[start:end])
    return ""


def _format_unit_context(unit: ContextualUnit) -> str:
    symbol = f" symbol={unit.symbol_id}" if unit.symbol_id else ""
    defines = f" defines={unit.defines}" if unit.defines else ""
    return f"L{unit.span[0]}: {unit.event} {unit.surface} scope={unit.scope}{symbol}{defines}"


def _symbol_from_question(question: str) -> str | None:
    match = re.search(r"`([^`]+)`", question)
    return match.group(1) if match else None


def _line_number(line: str) -> int | None:
    match = re.match(r"L(\d+):", line)
    return int(match.group(1)) if match else None


def _bcnlt_line_defines_symbol(line: str, symbol: str) -> bool:
    return f"DEFINE {symbol} " in line or line.endswith(f"DEFINE {symbol}")


def _python_line_defines_symbol(line: str, symbol: str) -> bool:
    escaped = re.escape(symbol)
    patterns = [
        rf"\bdef\s+{escaped}\s*\(",
        rf"\bclass\s+{escaped}\b",
        rf"\b{escaped}\s*=",
    ]
    return any(re.search(pattern, line) for pattern in patterns)


def _answer_matches_line(answer: str, line: int) -> bool:
    explicit = {
        int(match)
        for match in re.findall(r"(?i)(?:\bline\s*|\bL)(\d+)\b", answer)
    }
    if explicit:
        return explicit == {line}
    bare = re.fullmatch(r"\s*(\d+)\s*[.!]?\s*", answer)
    return bare is not None and int(bare.group(1)) == line


def _urlopen_transport(
    url: str,
    headers: dict[str, str],
    payload: dict[str, object],
    timeout: float,
) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _extract_chat_content(response: dict[str, object]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return "unknown"
    first = choices[0]
    if not isinstance(first, dict):
        return "unknown"
    message = first.get("message")
    if not isinstance(message, dict):
        return "unknown"
    content = message.get("content")
    return content.strip() if isinstance(content, str) else "unknown"


def _extract_ollama_content(response: dict[str, object]) -> str:
    message = response.get("message")
    if not isinstance(message, dict):
        return "unknown"
    content = message.get("content")
    return content.strip() if isinstance(content, str) else "unknown"
