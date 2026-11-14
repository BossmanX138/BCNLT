from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_project_codenet_source import audit_dataset  # noqa: E402

FIELDS = [
    "submission_id",
    "problem_id",
    "user_id",
    "date",
    "language",
    "original_language",
    "filename_ext",
    "status",
    "cpu_time",
    "memory",
    "code_size",
    "accuracy",
]


class ProjectCodeNetAuditTests(unittest.TestCase):
    def _fixture(self, root: Path, *, include_source: bool) -> Path:
        dataset = root / "Project_CodeNet"
        (dataset / "metadata").mkdir(parents=True)
        (dataset / "problem_descriptions").mkdir()
        (dataset / "data" / "p00001" / "JavaScript").mkdir(parents=True)
        with (dataset / "metadata" / "problem_list.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "name", "dataset"])
            writer.writeheader()
            writer.writerow({"id": "p00001", "name": "Example", "dataset": "AIZU"})
        rows = [
            {
                "submission_id": "s000000001",
                "problem_id": "p00001",
                "user_id": "u000000001",
                "date": "1",
                "language": "JavaScript",
                "original_language": "JavaScript",
                "filename_ext": "js",
                "status": "Accepted",
                "cpu_time": "1",
                "memory": "2",
                "code_size": "17",
                "accuracy": "1/1",
            },
            {
                "submission_id": "s000000002",
                "problem_id": "p00001",
                "user_id": "u000000002",
                "date": "2",
                "language": "JavaScript",
                "original_language": "JavaScript",
                "filename_ext": "js",
                "status": "Wrong Answer",
                "cpu_time": "1",
                "memory": "2",
                "code_size": "18",
                "accuracy": "0/1",
            },
        ]
        with (dataset / "metadata" / "p00001.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        (dataset / "problem_descriptions" / "p00001.html").write_text(
            "<p>Example</p>", encoding="utf-8"
        )
        if include_source:
            (dataset / "data" / "p00001" / "JavaScript" / "s000000001.js").write_text(
                "console.log(1);\n", encoding="utf-8"
            )
        return dataset

    @staticmethod
    def _change_first_metadata_row(dataset: Path, key: str, value: str) -> None:
        path = dataset / "metadata" / "p00001.csv"
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0][key] = value
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_counts_and_verifies_accepted_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._fixture(Path(temporary), include_source=True)
            report = audit_dataset(dataset, verify_accepted_files=True)
        self.assertTrue(report["verification"]["integrity_passed"])
        self.assertEqual(report["counts"]["metadata_rows"], 2)
        self.assertEqual(report["counts"]["accepted_rows"], 1)
        self.assertEqual(report["accepted_language_counts"], {"JavaScript": 1})

    def test_missing_accepted_source_fails_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._fixture(Path(temporary), include_source=False)
            report = audit_dataset(dataset, verify_accepted_files=True)
        self.assertFalse(report["verification"]["integrity_passed"])
        self.assertEqual(
            len(report["integrity_findings"]["missing_accepted_files_first_1000"]),
            1,
        )

    def test_invalid_accepted_metadata_fails_integrity(self) -> None:
        for key, value in (("code_size", "invalid"), ("submission_id", "../escape")):
            with self.subTest(key=key):
                with tempfile.TemporaryDirectory() as temporary:
                    dataset = self._fixture(Path(temporary), include_source=True)
                    self._change_first_metadata_row(dataset, key, value)
                    report = audit_dataset(dataset, verify_accepted_files=False)
                self.assertFalse(report["verification"]["integrity_passed"])
                self.assertTrue(
                    report["integrity_findings"]["invalid_accepted_metadata_first_1000"]
                )


if __name__ == "__main__":
    unittest.main()
