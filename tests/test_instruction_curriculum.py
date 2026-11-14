import gzip
import hashlib
import json
import sqlite3
import tempfile
import unittest
import zlib
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from bosscrafts_model.bcnlt_source_pipeline import (
    CURRENT_RUNTIME_SPEC_SHA256,
    PARSER_RUNTIME_IDENTITY_SHA256,
)
from bosscrafts_model.instruction_curriculum import (
    PythonInstructionCandidate,
    _EncodedInstructionCandidate,
    _exclude_unsafe_prompt_groups,
    build_instruction_curriculum,
    extract_python_instruction_candidates,
)
from bosscrafts_model.resumable_task_packing import (
    TASK_DATABASE_SCHEMA,
    _canonical_json,
    _encode_u16,
    audit_task_database,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding


class InstructionCurriculumExtractionTests(unittest.TestCase):
    def _extract(self, source: str):
        return extract_python_instruction_candidates(
            document_id="a" * 64,
            relative_path="sample.py",
            repository_identity="fixture/repository",
            license_spdx="MIT",
            source_sha256="b" * 64,
            source=source,
        )

    def test_extracts_signature_docstring_prompt_and_body(self):
        rows = self._extract(
            "from math import sqrt\n\n"
            "def hypotenuse(a: float, b: float) -> float:\n"
            '    """Return the hypotenuse for two perpendicular sides."""\n'
            "    return sqrt(a * a + b * b)\n"
        )
        self.assertEqual(len(rows), 1)
        self.assertIn("from math import sqrt", rows[0].prompt)
        self.assertIn("def hypotenuse", rows[0].prompt)
        self.assertNotIn("return sqrt", rows[0].prompt)
        self.assertEqual(rows[0].target, "    return sqrt(a * a + b * b)\n")

    def test_dedents_methods_to_standalone_functions(self):
        rows = self._extract(
            "class Counter:\n"
            "    def increment(self, value: int) -> int:\n"
            '        """Increase a supplied integer by exactly one."""\n'
            "        return value + 1\n"
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].prompt.startswith("def increment"))
        self.assertEqual(rows[0].target, "    return value + 1\n")

    def test_rejects_functions_without_meaningful_docstrings_or_bodies(self):
        rows = self._extract(
            "def no_doc(value):\n"
            "    return value\n\n"
            "def too_short(value):\n"
            '    """short"""\n'
            "    return value\n\n"
            "def empty(value):\n"
            '    """Documented but deliberately has no implementation."""\n'
            "    pass\n"
        )
        self.assertEqual(rows, [])

    def test_rejects_targets_that_depend_on_omitted_module_definitions(self):
        rows = self._extract(
            "OFFSET = 7\n\n"
            "def helper(value):\n"
            "    return value * 2\n\n"
            "def depends_on_constant(value):\n"
            '    """Return a value adjusted by the module-level offset."""\n'
            "    return value + OFFSET\n\n"
            "def depends_on_helper(value):\n"
            '    """Return a value transformed by a module-level helper."""\n'
            "    return helper(value)\n"
        )
        self.assertEqual(rows, [])

    def test_rejects_nested_function_candidate_that_needs_closure_state(self):
        rows = self._extract(
            "def make_multiplier(factor):\n"
            '    """Build and return a multiplier using the supplied factor."""\n'
            "    def multiply(value):\n"
            '        """Multiply a supplied value by the captured outer factor."""\n'
            "        return value * factor\n"
            "    return multiply\n"
        )
        self.assertEqual(len(rows), 1)
        self.assertIn("def make_multiplier", rows[0].prompt)


class InstructionCurriculumPromptIsolationTests(unittest.TestCase):
    @staticmethod
    def _row(
        number: int,
        *,
        split: str,
        prompt: str,
        target: str,
    ) -> _EncodedInstructionCandidate:
        candidate = PythonInstructionCandidate(
            document_id=f"document-{number}",
            record_index=number,
            relative_path=f"fixture-{number}.py",
            repository_identity="fixture/repository",
            license_spdx="MIT",
            source_sha256=f"{number:064x}",
            prompt=prompt,
            target=target,
            docstring="A meaningful fixture docstring with enough content.",
        )
        return _EncodedInstructionCandidate.from_candidate(
            example_id=f"{number:064x}",
            candidate=candidate,
            input_ids=(1, 2, 3, 4),
            loss_start_index=2,
            split=split,
        )

    def test_excludes_conflicting_and_cross_split_prompt_groups(self):
        rows = {
            "train": [
                self._row(1, split="train", prompt="same", target="one"),
                self._row(3, split="train", prompt="cross", target="shared"),
                self._row(5, split="train", prompt="safe", target="answer"),
            ],
            "validation": [
                self._row(2, split="validation", prompt="same", target="two"),
            ],
            "test": [
                self._row(4, split="test", prompt="cross", target="shared"),
            ],
        }
        rejection_counts: Counter[str] = Counter()

        filtered = _exclude_unsafe_prompt_groups(rows, rejection_counts)

        self.assertEqual(
            [item.example_id for item in filtered["train"]],
            [f"{5:064x}"],
        )
        self.assertEqual(filtered["validation"], [])
        self.assertEqual(filtered["test"], [])
        self.assertEqual(rejection_counts["conflicting_prompt"], 2)
        self.assertEqual(rejection_counts["cross_split_prompt"], 2)

    def test_deduplicates_identical_prompts_within_one_split_deterministically(self):
        rows = {
            "train": [
                self._row(9, split="train", prompt="same", target="answer"),
                self._row(7, split="train", prompt="same", target="answer"),
            ],
            "validation": [],
            "test": [],
        }
        rejection_counts: Counter[str] = Counter()

        filtered = _exclude_unsafe_prompt_groups(rows, rejection_counts)

        self.assertEqual(
            [item.example_id for item in filtered["train"]],
            [f"{7:064x}"],
        )
        self.assertEqual(rejection_counts["duplicate_prompt"], 1)


class InstructionCurriculumMaterializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binding = NativeTokenizerBinding.load()

    @staticmethod
    def _source(name: str, increment: int) -> str:
        return (
            f"def {name}(value: int) -> int:\n"
            f'    """Return the supplied integer after adding exactly {increment}."""\n'
            f"    return value + {increment}\n"
        )

    def _write_fixture(self, root: Path):
        split_by_document = {
            "train-unique": "train",
            "validation-unique": "validation",
            "test-unique": "test",
            "train-copy": "train",
            "test-copy": "test",
        }
        sources = {
            "train-unique": self._source("train_only", 11),
            "validation-unique": self._source("validation_only", 12),
            "test-unique": self._source("test_only", 13),
            "train-copy": self._source("copied_function", 14),
            "test-copy": self._source("copied_function", 14),
        }
        catalog_path = root / "catalog.sqlite"
        catalog = sqlite3.connect(catalog_path)
        try:
            catalog.execute(
                """CREATE TABLE documents(
                       document_id TEXT PRIMARY KEY,canonical_language TEXT NOT NULL,
                       relative_path TEXT NOT NULL,repository_identity TEXT NOT NULL,
                       license_spdx TEXT NOT NULL,content_sha256 TEXT NOT NULL,
                       content TEXT NOT NULL)"""
            )
            catalog.executemany(
                "INSERT INTO documents VALUES(?,?,?,?,?,?,?)",
                [
                    (
                        document_id,
                        "python",
                        f"{document_id}.py",
                        "fixture/repository",
                        "MIT",
                        hashlib.sha256(source.encode()).hexdigest(),
                        source,
                    )
                    for document_id, source in sources.items()
                ],
            )
            catalog.commit()
        finally:
            catalog.close()
        catalog_sha256 = hashlib.sha256(catalog_path.read_bytes()).hexdigest()

        admission_path = root / "admission.sqlite"
        admission = sqlite3.connect(admission_path)
        try:
            admission.execute(
                "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
            )
            admission.execute(
                "CREATE TABLE documents(document_id TEXT PRIMARY KEY,status TEXT,language TEXT)"
            )
            admission.executemany(
                "INSERT INTO metadata VALUES(?,?)",
                [
                    ("catalog_exhausted", "true"),
                    ("binding_sha256", self.binding.binding_sha256),
                    ("tokenizer_sha256", self.binding.tokenizer_sha256),
                    ("vocabulary_size", str(self.binding.vocab_size)),
                    ("parser_runtime_identity_sha256", PARSER_RUNTIME_IDENTITY_SHA256),
                    (
                        "semantic_spec_sha256",
                        _canonical_json(CURRENT_RUNTIME_SPEC_SHA256),
                    ),
                    ("source_catalog_sha256", catalog_sha256),
                ],
            )
            admission.executemany(
                "INSERT INTO documents VALUES(?,'admitted','python')",
                [(document_id,) for document_id in sources],
            )
            admission.commit()
        finally:
            admission.close()

        split_path = root / "splits.sqlite"
        split_database = sqlite3.connect(split_path)
        try:
            split_database.execute(
                "CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)"
            )
            split_database.execute(
                """CREATE TABLE tasks(
                       example_id TEXT PRIMARY KEY,document_id TEXT NOT NULL,
                       language TEXT NOT NULL,family TEXT NOT NULL,
                       record_index INTEGER NOT NULL,split TEXT NOT NULL,
                       input_ids_zlib BLOB NOT NULL,token_count INTEGER NOT NULL,
                       loss_start_index INTEGER NOT NULL,
                       semantic_spec_sha256 TEXT NOT NULL,
                       parser_runtime_identity_sha256 TEXT NOT NULL,
                       source_sha256 TEXT NOT NULL)"""
            )
            split_database.executemany(
                "INSERT INTO metadata VALUES(?,?)",
                sorted(
                    {
                        "schema_version": TASK_DATABASE_SCHEMA,
                        "packing_complete": "true",
                        "binding_sha256": self.binding.binding_sha256,
                        "tokenizer_sha256": self.binding.tokenizer_sha256,
                        "vocabulary_size": str(self.binding.vocab_size),
                        "parser_runtime_identity_sha256": PARSER_RUNTIME_IDENTITY_SHA256,
                        "maximum_sequence_tokens": "256",
                        "source_catalog_sha256": catalog_sha256,
                    }.items()
                ),
            )
            for index, (document_id, split) in enumerate(split_by_document.items()):
                input_ids = [100 + index, 200 + index, 300 + index]
                identity = {
                    "document_id": document_id,
                    "family": "semantic_to_code",
                    "record_index": 0,
                    "input_ids_sha256": hashlib.sha256(
                        _canonical_json(input_ids).encode()
                    ).hexdigest(),
                }
                example_id = hashlib.sha256(
                    _canonical_json(identity).encode()
                ).hexdigest()
                split_database.execute(
                    "INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        example_id,
                        document_id,
                        "python",
                        "semantic_to_code",
                        0,
                        split,
                        zlib.compress(_encode_u16(input_ids), 6),
                        len(input_ids),
                        1,
                        "a" * 64,
                        PARSER_RUNTIME_IDENTITY_SHA256,
                        hashlib.sha256(sources[document_id].encode()).hexdigest(),
                    ),
                )
            split_database.commit()
        finally:
            split_database.close()

        humaneval_path = root / "HumanEval.jsonl.gz"
        with gzip.open(humaneval_path, "wt", encoding="utf-8", newline="\n") as handle:
            for index in range(164):
                handle.write(
                    json.dumps(
                        {
                            "task_id": f"HumanEval/{index}",
                            "prompt": f"def benchmark_{index}(value):\n    pass\n",
                            "canonical_solution": f"return value * {index + 1000}\n",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        return catalog_path, admission_path, split_path, humaneval_path

    def test_builds_audited_output_and_removes_cross_split_prompt_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, admission, splits, humaneval = self._write_fixture(root)
            output = root / "instruction.sqlite"

            result = build_instruction_curriculum(
                self.binding,
                source_catalog=catalog,
                admission_database=admission,
                split_database=splits,
                humaneval_dataset=humaneval,
                output_database=output,
                quotas={"train": 1, "validation": 1, "test": 1},
                max_sequence=256,
            )

            self.assertEqual(result.example_count, 3)
            self.assertEqual(
                result.split_counts,
                {"test": 1, "train": 1, "validation": 1},
            )
            self.assertEqual(result.rejection_counts["cross_split_prompt"], 2)
            connection = sqlite3.connect(output)
            try:
                documents = {
                    row[0]
                    for row in connection.execute("SELECT document_id FROM tasks")
                }
            finally:
                connection.close()
            self.assertEqual(
                documents,
                {"train-unique", "validation-unique", "test-unique"},
            )
            self.assertFalse(output.with_name(f".{output.name}.tmp").exists())

    def test_rejects_split_database_from_another_source_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, admission, splits, humaneval = self._write_fixture(root)
            connection = sqlite3.connect(splits)
            try:
                connection.execute(
                    "UPDATE metadata SET value=? WHERE key='source_catalog_sha256'",
                    ("f" * 64,),
                )
                connection.commit()
            finally:
                connection.close()

            with self.assertRaisesRegex(
                ValueError, "split database source catalog lineage mismatch"
            ):
                build_instruction_curriculum(
                    self.binding,
                    source_catalog=catalog,
                    admission_database=admission,
                    split_database=splits,
                    humaneval_dataset=humaneval,
                    output_database=root / "instruction.sqlite",
                    quotas={"train": 1, "validation": 1, "test": 1},
                    max_sequence=256,
                )

    def test_failed_final_audit_does_not_publish_or_leave_temporary_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog, admission, splits, humaneval = self._write_fixture(root)
            output = root / "instruction.sqlite"

            def audit_split_then_fail(binding, database_path):
                if Path(database_path).resolve() == splits.resolve():
                    return audit_task_database(binding, database_path)
                raise ValueError("deliberate final audit failure")

            with (
                patch(
                    "bosscrafts_model.instruction_curriculum.audit_task_database",
                    side_effect=audit_split_then_fail,
                ),
                self.assertRaisesRegex(ValueError, "deliberate final audit failure"),
            ):
                build_instruction_curriculum(
                    self.binding,
                    source_catalog=catalog,
                    admission_database=admission,
                    split_database=splits,
                    humaneval_dataset=humaneval,
                    output_database=output,
                    quotas={"train": 1, "validation": 1, "test": 1},
                    max_sequence=256,
                )

            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(f".{output.name}.tmp").exists())


if __name__ == "__main__":
    unittest.main()
