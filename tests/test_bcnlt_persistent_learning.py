from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from bosscrafts_model.bcnlt_source_pipeline import BCNLTSourcePipeline
from bosscrafts_model.tokenizer import NativeTokenizerBinding


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNKNOWN_CPP = "int f() { return JNI_ERR; }\n"


class PersistentLearningPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binding = NativeTokenizerBinding.load()
        cls.temp_root = PROJECT_ROOT / ".tmp-tests"
        cls.temp_root.mkdir(exist_ok=True)

    def test_ephemeral_is_the_safe_default(self) -> None:
        pipeline = BCNLTSourcePipeline(self.binding)
        prepared = pipeline.prepare_semantic(
            UNKNOWN_CPP,
            language="cpp",
            path="ephemeral.cpp",
        )
        self.assertEqual(pipeline.learning_mode, "ephemeral")
        self.assertIsNone(pipeline.learned_memory_root)
        self.assertEqual(prepared.learning_mode, "ephemeral")
        self.assertIsNone(prepared.learned_memory_sha256)
        self.assertEqual(prepared.learned_definition_count, 0)
        self.assertEqual(prepared.extended_core_entry_count, 0)

    def test_observe_persists_and_read_only_reuses_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.temp_root) as directory:
            memory_root = Path(directory)
            observer = BCNLTSourcePipeline(
                self.binding,
                learning_mode="observe",
                learned_memory_root=memory_root,
            )
            observed = observer.prepare_semantic(
                UNKNOWN_CPP,
                language="cpp",
                path="observe.cpp",
            )
            memory_path = memory_root / "cpp.jsonl"
            self.assertTrue(memory_path.is_file())
            persisted = memory_path.read_bytes()
            persisted_sha256 = hashlib.sha256(persisted).hexdigest()
            self.assertEqual(observed.learning_mode, "observe")
            self.assertEqual(observed.learned_memory_sha256, persisted_sha256)
            self.assertGreaterEqual(observed.learned_definition_count, 1)
            self.assertEqual(observed.extended_core_entry_count, 0)

            reader = BCNLTSourcePipeline(
                self.binding,
                learning_mode="read_only",
                learned_memory_root=memory_root,
            )
            retrieved = reader.prepare_semantic(
                UNKNOWN_CPP,
                language="cpp",
                path="read-only.cpp",
            )
            self.assertEqual(retrieved.learning_mode, "read_only")
            self.assertEqual(retrieved.learned_memory_sha256, persisted_sha256)
            self.assertEqual(
                retrieved.learned_definition_count,
                observed.learned_definition_count,
            )
            self.assertEqual(memory_path.read_bytes(), persisted)

    def test_persistent_modes_require_a_non_root_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit learned-memory root"):
            BCNLTSourcePipeline(self.binding, learning_mode="observe")
        with self.assertRaisesRegex(ValueError, "filesystem root"):
            BCNLTSourcePipeline(
                self.binding,
                learning_mode="read_only",
                learned_memory_root=Path(Path.cwd().anchor),
            )
        with self.assertRaisesRegex(ValueError, "must not receive"):
            BCNLTSourcePipeline(
                self.binding,
                learning_mode="ephemeral",
                learned_memory_root=PROJECT_ROOT / "forbidden",
            )


if __name__ == "__main__":
    unittest.main()
