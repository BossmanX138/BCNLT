from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys
import unittest

from bosscrafts_model.current_corpus import SourceCandidate
from bosscrafts_model.current_task_packing import CurrentCorpusTaskPacker
from bosscrafts_model.heldout_source_admission import select_heldout_candidates


def _candidate(document_id: str, language: str) -> SourceCandidate:
    source = "value = 1\n"
    return SourceCandidate(
        document_id=document_id,
        language=language,
        source=source,
        relative_path=f"{document_id}.txt",
        content_sha256=hashlib.sha256(source.encode()).hexdigest(),
        license_spdx="MIT",
        attribution="fixture-author",
        ownership="third_party",
        source_provider="fixture",
        source_name="fixture/heldout-admission",
        source_revision="fixture-v1",
        repository_identity="fixture/repository",
        upstream_url="https://example.invalid/fixture",
    )


def _ids(language: str, split_kind: str, count: int) -> list[str]:
    found: list[str] = []
    for index in range(100_000):
        document_id = f"{language}-{index}"
        split = CurrentCorpusTaskPacker._split(document_id)
        is_heldout = split in {"validation", "test"}
        if is_heldout == (split_kind == "heldout"):
            found.append(document_id)
            if len(found) == count:
                return found
    raise AssertionError("unable to make split fixture")


class HeldoutSourceAdmissionTests(unittest.TestCase):
    def test_selects_only_heldout_candidates_with_equal_language_quota(self):
        candidates = []
        for language in ("python", "rust"):
            candidates.extend(_candidate(value, language) for value in _ids(language, "train", 3))
            candidates.extend(_candidate(value, language) for value in _ids(language, "heldout", 3))
        selected = select_heldout_candidates(
            candidates, languages={"python", "rust"}, per_language=2
        )
        self.assertEqual(len(selected), 4)
        self.assertEqual(
            {language: sum(item.language == language for item in selected) for language in ("python", "rust")},
            {"python": 2, "rust": 2},
        )
        self.assertTrue(
            all(CurrentCorpusTaskPacker._split(item.document_id) in {"validation", "test"} for item in selected)
        )

    def test_raises_when_a_language_cannot_fill_the_heldout_quota(self):
        candidates = [_candidate(value, "python") for value in _ids("python", "heldout", 1)]
        with self.assertRaisesRegex(ValueError, "held-out candidate quota"):
            select_heldout_candidates(candidates, languages={"python", "rust"}, per_language=1)

    def test_heldout_admission_script_runs_from_project_root(self):
        project_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "build_heldout_semantic_admission.py"), "--help"],
            cwd=project_root,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--candidates-per-language", completed.stdout)


if __name__ == "__main__":
    unittest.main()
