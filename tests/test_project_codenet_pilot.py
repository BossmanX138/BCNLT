from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bosscrafts_model.project_codenet_pilot import build_pilot  # noqa: E402


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


class _NoOverlap:
    sha256 = "b" * 64

    def overlaps(self, prompt: str, code: str) -> bool:
        return False


class ProjectCodeNetPilotTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        dataset = root / "Project_CodeNet"
        for problem in ("p00001", "p00002"):
            (dataset / "data" / problem / "JavaScript").mkdir(parents=True)
        (dataset / "metadata").mkdir()
        (dataset / "problem_descriptions").mkdir()
        description = (
            "<html><style>ignored</style><body><h1>Sum two values</h1>"
            "<p>Read two integers from standard input and print their sum. "
            "The input contains exactly two decimal integers.</p></body></html>"
        )
        rows_by_problem = {}
        for index, problem in enumerate(("p00001", "p00002"), start=1):
            submission = f"s{index:09d}"
            source = dataset / "data" / problem / "JavaScript" / f"{submission}.js"
            source.write_text(
                f"const n = require('fs').readFileSync(0,'utf8').trim().split(/\\s+/).map(Number);\nconsole.log(n[0]+n[1]+{index - 1});\n",
                encoding="utf-8",
            )
            (dataset / "problem_descriptions" / f"{problem}.html").write_text(
                description, encoding="utf-8"
            )
            rows_by_problem[problem] = [
                {
                    "submission_id": submission,
                    "problem_id": problem,
                    "user_id": "u000000001",
                    "date": "1",
                    "language": "JavaScript",
                    "original_language": "JavaScript",
                    "filename_ext": "js",
                    "status": "Accepted",
                    "cpu_time": "1",
                    "memory": "1",
                    "code_size": str(source.stat().st_size),
                    "accuracy": "1/1",
                }
            ]
        for problem, rows in rows_by_problem.items():
            with (dataset / "metadata" / f"{problem}.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
        return dataset

    def test_balanced_deterministic_pilot_and_problem_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._fixture(Path(temporary))
            kwargs = dict(
                languages={"JavaScript"},
                per_language_limit=2,
                seed="test",
                minimum_code_bytes=1,
                maximum_code_bytes=10000,
                archive_sha256="a" * 64,
                tooling_commit="deadbeef",
                overlap_filter=_NoOverlap(),
            )
            first, report = build_pilot(dataset, **kwargs)
            second, _ = build_pilot(dataset, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertEqual(report["language_counts"], {"JavaScript": 2})
        self.assertTrue(all(row["dataset_license"] == "CDLA-Permissive-2.0" for row in first))
        self.assertTrue(all(row["split"] in {"train", "validation", "test"} for row in first))


if __name__ == "__main__":
    unittest.main()
