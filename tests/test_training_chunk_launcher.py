from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from bosscrafts_model.balanced_curriculum import (
    CURRICULUM_SCHEMA,
    SPLITS,
    TASK_FAMILIES,
)
from bosscrafts_model.bcnlt_source_pipeline import CURRENT_RUNTIME_SPEC_SHA256
from bosscrafts_model.resumable_task_packing import TASK_DATABASE_SCHEMA


class TrainingChunkLauncherTests(unittest.TestCase):
    def _database(self, path: Path, *, balanced: bool, complete: bool = True) -> None:
        connection = sqlite3.connect(path)
        try:
            connection.execute(
                "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            metadata = {
                "schema_version": TASK_DATABASE_SCHEMA,
                "packing_complete": "true",
            }
            if balanced:
                metadata["curriculum_schema_version"] = CURRICULUM_SCHEMA
                metadata["maximum_sequence_tokens"] = "2048"
            else:
                metadata["gate_only"] = "true"
            connection.executemany(
                "INSERT INTO metadata VALUES(?,?)", sorted(metadata.items())
            )
            if balanced:
                connection.execute(
                    "CREATE TABLE tasks (split TEXT, language TEXT, family TEXT)"
                )
                quotas = {"train": 2048, "validation": 128, "test": 128}
                languages = (
                    tuple(sorted(CURRENT_RUNTIME_SPEC_SHA256))
                    if complete
                    else ("c_sharp",)
                )
                connection.executemany(
                    "INSERT INTO tasks VALUES(?,?,?)",
                    (
                        (split, language, family)
                        for split in SPLITS
                        for language in languages
                        for family in TASK_FAMILIES
                        for _ in range(quotas[split])
                    ),
                )
            connection.commit()
        finally:
            connection.close()

    def test_dry_run_freezes_one_safe_full_model_chunk_and_resume(self):
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "run_bosscrafts_training_chunk.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "tasks.sqlite"
            checkpoint = root / "checkpoint.pt"
            self._database(database, balanced=True)
            command = [
                sys.executable,
                str(script),
                "--database",
                str(database),
                "--checkpoint",
                str(checkpoint),
                "--dry-run",
            ]
            first = subprocess.run(
                command,
                cwd=project_root,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stderr)
            payload = json.loads(first.stdout)
            self.assertEqual(payload["gradient_accumulation"], 8)
            self.assertEqual(payload["maximum_optimizer_steps_this_chunk"], 100)
            self.assertEqual(payload["epochs"], 1)
            self.assertTrue(payload["full_model"])
            self.assertFalse(payload["resume"])
            expected_status = str(checkpoint.with_name("training-progress.json").resolve())
            self.assertEqual(payload["status_output"], expected_status)
            status_flag_index = payload["command"].index("--status-output")
            self.assertEqual(payload["command"][status_flag_index + 1], expected_status)
            checkpoint.touch()
            resumed = subprocess.run(
                command,
                cwd=project_root,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertTrue(json.loads(resumed.stdout)["resume"])

    def test_refuses_gate_or_noncurriculum_database(self):
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "run_bosscrafts_training_chunk.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "tasks.sqlite"
            self._database(database, balanced=False)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--database",
                    str(database),
                    "--checkpoint",
                    str(root / "checkpoint.pt"),
                    "--dry-run",
                ],
                cwd=project_root,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("balanced curriculum", completed.stderr)

    def test_refuses_partial_balanced_curriculum(self):
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "run_bosscrafts_training_chunk.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            database = root / "tasks.sqlite"
            self._database(database, balanced=True, complete=False)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--database",
                    str(database),
                    "--checkpoint",
                    str(root / "checkpoint.pt"),
                    "--dry-run",
                ],
                cwd=project_root,
                text=True,
                encoding="utf-8",
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("exact 14-language", completed.stderr)


if __name__ == "__main__":
    unittest.main()
