from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from bosscrafts_model.balanced_binary_shards import (
    BalancedBinaryShardWriter,
    decode_u16_tokens,
    encode_u16_tokens,
    iter_binary_examples,
)
from bosscrafts_model.bcnlt_source_pipeline import BCNLTSourcePipeline
from bosscrafts_model.current_corpus import CurrentCorpusAdmission, SourceCandidate
from bosscrafts_model.current_task_packing import CurrentCorpusTaskPacker
from bosscrafts_model.tokenizer import NativeTokenizerBinding


def _candidate(document_id: str, source: str) -> SourceCandidate:
    return SourceCandidate(
        document_id=document_id,
        language="python",
        source=source,
        relative_path=f"{document_id}.py",
        content_sha256=hashlib.sha256(source.encode()).hexdigest(),
        license_spdx="MIT",
        attribution="fixture-author",
        ownership="third_party",
        source_provider="fixture",
        source_name="fixture/binary-shards",
        source_revision="fixture-v1",
        repository_identity="fixture/repository",
        upstream_url="https://example.invalid/fixture",
    )


class BalancedBinaryShardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binding = NativeTokenizerBinding.load()
        cls.pipeline = BCNLTSourcePipeline(cls.binding)

    def _packed_fixture(self, admission_dir: str, tasks_dir: str) -> None:
        candidates = (
            _candidate("binary-a", "x = 1\ny = x + 2\nprint(y)\n"),
            _candidate("binary-b", "def add(a, b):\n    return a + b\nprint(add(2, 3))\n"),
        )
        CurrentCorpusAdmission(self.pipeline).materialize(candidates, admission_dir)
        CurrentCorpusTaskPacker(self.binding).materialize(admission_dir, tasks_dir)

    def test_uint16_codec_preserves_full_frozen_vocabulary_range(self):
        encoded = encode_u16_tokens([0, 1, 65_535])
        self.assertEqual(encoded, b"\x00\x00\x01\x00\xff\xff")
        self.assertEqual(decode_u16_tokens(encoded), [0, 1, 65_535])
        with self.assertRaisesRegex(ValueError, "uint16"):
            encode_u16_tokens([65_536])

    def test_shards_round_trip_examples_and_equalize_sampling_mass(self):
        with (
            tempfile.TemporaryDirectory() as admission_dir,
            tempfile.TemporaryDirectory() as tasks_dir,
            tempfile.TemporaryDirectory() as output_dir,
        ):
            self._packed_fixture(admission_dir, tasks_dir)
            result = BalancedBinaryShardWriter(
                self.binding, max_tokens_per_shard=2048
            ).materialize(tasks_dir, output_dir)

            source_rows = [
                json.loads(line)
                for line in (Path(tasks_dir) / "examples.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            binary_rows = list(iter_binary_examples(output_dir))
            self.assertEqual(result.example_count, len(source_rows))
            self.assertGreater(result.shard_count, 1)
            self.assertEqual(
                [(row["example_id"], row["input_ids"], row["loss_start_index"]) for row in binary_rows],
                [(row["example_id"], row["input_ids"], row["loss_start_index"]) for row in source_rows],
            )
            manifest = json.loads(
                (Path(output_dir) / "manifest.json").read_text(encoding="utf-8")
            )
            for shard in manifest["shards"]:
                self.assertLessEqual(shard["token_count"], 2048)
            masses_by_split: dict[str, list[float]] = {}
            for key, mass in manifest["weighted_cell_mass"].items():
                split = key.split("|", 1)[0]
                masses_by_split.setdefault(split, []).append(mass)
            for masses in masses_by_split.values():
                self.assertLess(max(masses) - min(masses), 1e-5)
            self.assertFalse(manifest["training_ready"])

    def test_binary_materialization_is_deterministic_and_rejects_tampering(self):
        with (
            tempfile.TemporaryDirectory() as admission_dir,
            tempfile.TemporaryDirectory() as tasks_dir,
            tempfile.TemporaryDirectory() as first_dir,
            tempfile.TemporaryDirectory() as second_dir,
        ):
            self._packed_fixture(admission_dir, tasks_dir)
            first = BalancedBinaryShardWriter(self.binding).materialize(tasks_dir, first_dir)
            second = BalancedBinaryShardWriter(self.binding).materialize(tasks_dir, second_dir)
            self.assertEqual(first.manifest_sha256, second.manifest_sha256)
            self.assertEqual(first.shard_hashes, second.shard_hashes)

            examples_path = Path(tasks_dir) / "examples.jsonl"
            examples_path.write_text(
                examples_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "packed examples SHA-256"):
                BalancedBinaryShardWriter(self.binding).materialize(tasks_dir, first_dir)

    def test_binary_shard_script_runs_from_the_project_root(self):
        project_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "shard_current_65k_tasks.py"), "--help"],
            cwd=project_root,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--max-tokens-per-shard", completed.stdout)

    def test_existing_binary_generation_is_never_overwritten(self):
        with (
            tempfile.TemporaryDirectory() as admission_dir,
            tempfile.TemporaryDirectory() as tasks_dir,
            tempfile.TemporaryDirectory() as output_dir,
        ):
            self._packed_fixture(admission_dir, tasks_dir)
            root = Path(output_dir)
            writer = BalancedBinaryShardWriter(self.binding)
            writer.materialize(tasks_dir, root)
            original = {
                path.name: path.read_bytes()
                for path in root.iterdir()
                if path.is_file()
            }

            with self.assertRaisesRegex(ValueError, "already contains"):
                writer.materialize(tasks_dir, root)

            self.assertEqual(
                {
                    path.name: path.read_bytes()
                    for path in root.iterdir()
                    if path.is_file()
                },
                original,
            )

    def test_encoder_failure_leaves_no_published_binary_files(self):
        with (
            tempfile.TemporaryDirectory() as admission_dir,
            tempfile.TemporaryDirectory() as tasks_dir,
            tempfile.TemporaryDirectory() as output_dir,
        ):
            self._packed_fixture(admission_dir, tasks_dir)
            root = Path(output_dir)
            from bosscrafts_model import balanced_binary_shards as module

            real_encode = module.encode_u16_tokens
            encode_calls = 0

            def fail_second_encode(token_ids: list[int]) -> bytes:
                nonlocal encode_calls
                encode_calls += 1
                if encode_calls == 2:
                    raise RuntimeError("deliberate encoder failure")
                return real_encode(token_ids)

            with (
                mock.patch.object(
                    module,
                    "encode_u16_tokens",
                    side_effect=fail_second_encode,
                ),
                self.assertRaisesRegex(RuntimeError, "deliberate encoder failure"),
            ):
                BalancedBinaryShardWriter(self.binding).materialize(tasks_dir, root)

            self.assertEqual(list(root.iterdir()), [])

    def test_reader_rejects_manifest_shard_path_escape(self):
        with (
            tempfile.TemporaryDirectory() as admission_dir,
            tempfile.TemporaryDirectory() as tasks_dir,
            tempfile.TemporaryDirectory() as directory,
        ):
            self._packed_fixture(admission_dir, tasks_dir)
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            BalancedBinaryShardWriter(self.binding).materialize(tasks_dir, output)
            manifest_path = output / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.pop("manifest_sha256")
            first_shard = manifest["shards"][0]
            original_token = output / first_shard["token_file"]
            escaped_token = root / "outside.u16"
            escaped_token.write_bytes(original_token.read_bytes())
            first_shard["token_file"] = "../outside.u16"
            canonical_manifest = json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            manifest["manifest_sha256"] = hashlib.sha256(
                canonical_manifest.encode("utf-8")
            ).hexdigest()
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
                newline="\n",
            )

            with self.assertRaisesRegex(ValueError, "file inventory"):
                list(iter_binary_examples(output))


if __name__ == "__main__":
    unittest.main()
