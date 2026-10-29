from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bcnlt.adaptive_vocab.runner import (
    ExperimentInputs,
    plan_experiment,
    run_experiment,
)


def _load_spec(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != "bcnlt-adaptive-run-spec-v1":
        raise ValueError("unsupported BCNLT adaptive run specification")
    return raw


def _expand(value: str, *, repository: Path, python: str) -> str:
    return value.replace("{repo}", str(repository)).replace("{python}", python)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the resumable BCNLT v5 adaptive experiment")
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-training", action="store_true")
    args = parser.parse_args(argv)
    try:
        spec = _load_spec(args.spec.resolve(strict=True))
        repository = Path(__file__).resolve().parents[1]
        raw_commands = spec.get("phases")
        if not isinstance(raw_commands, dict):
            raise ValueError("run specification requires a phases object")
        commands = []
        phase_inputs = []
        for name, raw in raw_commands.items():
            if not isinstance(name, str) or not isinstance(raw, dict):
                raise ValueError("invalid phase specification")
            command = raw.get("command")
            inputs = raw.get("inputs", [])
            if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
                raise ValueError(f"phase {name} requires an argument-list command")
            if not isinstance(inputs, list) or not all(isinstance(item, str) for item in inputs):
                raise ValueError(f"phase {name} requires an input path list")
            commands.append(
                (name, tuple(_expand(item, repository=repository, python=sys.executable) for item in command))
            )
            phase_inputs.append(
                (
                    name,
                    tuple(
                        Path(_expand(item, repository=repository, python=sys.executable)).resolve(strict=True)
                        for item in inputs
                    ),
                )
            )
        inputs = ExperimentInputs(
            output_root=Path(str(spec["output_root"])),
            protected_paths=tuple(Path(str(item)) for item in spec["protected_paths"]),
            config_path=Path(str(spec["config_path"])),
            code_revision=str(spec["code_revision"]),
            phase_commands=tuple(commands),
            phase_inputs=tuple(phase_inputs),
            legacy_corpus=bool(spec.get("legacy_corpus", False)),
            start_training=args.start_training,
        )
        phases = plan_experiment(inputs)
        selection_value = spec.get("tokenizer_selection_path")
        selection_path = Path(str(selection_value)) if selection_value else None
        result = run_experiment(
            phases,
            resume=args.resume,
            start_training=args.start_training,
            tokenizer_selection_path=selection_path,
        )
        root = Path(str(spec["output_root"]))
        root.mkdir(parents=True, exist_ok=True)
        (root / "experiment-run.json").write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return result.return_code
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
