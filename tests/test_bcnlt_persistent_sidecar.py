from __future__ import annotations

import json
import unittest
from concurrent.futures import ThreadPoolExecutor

from bosscrafts_model.bcnlt_source_pipeline import (
    BCNLTPreprocessingError,
    BCNLTSourcePipeline,
)
from bosscrafts_model.tokenizer import NativeTokenizerBinding

SAMPLES = {
    "bash": "f() { local x=$1; return $((x + 1)); }\n",
    "c": "int f(int x) { return x + 1; }\n",
    "c_sharp": "class C { static int F(int x) { return x + 1; } }\n",
    "cpp": "int f(int x) { return x + 1; }\n",
    "go": "package main\nfunc f(x int) int { return x + 1 }\n",
    "java": "class C { static int f(int x) { return x + 1; } }\n",
    "javascript": "function f(x) { return x + 1; }\n",
    "kotlin": "fun f(x: Int): Int { return x + 1 }\n",
    "php": "<?php function f($x) { return $x + 1; }\n",
    "python": "def f(x):\n    return x + 1\n",
    "ruby": "def f(x)\n  x + 1\nend\n",
    "rust": "fn f(x: i32) -> i32 { x + 1 }\n",
    "swift": "func f(_ x: Int) -> Int { return x + 1 }\n",
    "typescript": "function f(x: number): number { return x + 1; }\n",
}


class PersistentSidecarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.binding = NativeTokenizerBinding.load()

    def test_persistent_transport_is_byte_equivalent_for_all_languages(self) -> None:
        reference = BCNLTSourcePipeline(self.binding)
        persistent = BCNLTSourcePipeline(self.binding, persistent_sidecars=True)
        self.addCleanup(persistent.close)
        for language, source in SAMPLES.items():
            with self.subTest(language=language):
                expected = reference.prepare_semantic(
                    source,
                    language=language,
                    path=f"equivalence.{language}",
                )
                actual = persistent.prepare_semantic(
                    source,
                    language=language,
                    path=f"equivalence.{language}",
                )
                self.assertEqual(actual, expected)

    def test_worker_is_reused_and_recovers_after_a_rejected_request(self) -> None:
        pipeline = BCNLTSourcePipeline(self.binding, persistent_sidecars=True)
        self.addCleanup(pipeline.close)
        first = pipeline.prepare_semantic(
            SAMPLES["python"], language="python", path="first.py"
        )
        first_pids = pipeline.persistent_worker_pids
        self.assertEqual(len(first_pids), 1)
        with self.assertRaises(BCNLTPreprocessingError):
            pipeline.prepare_semantic(
                "function {",
                language="javascript",
                path="broken.js",
            )
        second = pipeline.prepare_semantic(
            SAMPLES["python"], language="python", path="first.py"
        )
        self.assertEqual(pipeline.persistent_worker_pids, first_pids)
        self.assertEqual(first.canonical_jsonl, second.canonical_jsonl)

    def test_threaded_calls_use_a_bounded_worker_set_and_preserve_order(self) -> None:
        pipeline = BCNLTSourcePipeline(self.binding, persistent_sidecars=True)
        self.addCleanup(pipeline.close)
        cases = list(SAMPLES.items()) * 2

        def prepare(index_and_case: tuple[int, tuple[str, str]]) -> tuple[int, str, str]:
            index, (language, source) = index_and_case
            prepared = pipeline.prepare_semantic(
                source,
                language=language,
                path=f"threaded-{index}.{language}",
            )
            return index, language, prepared.canonical_sha256 or ""

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(prepare, enumerate(cases)))

        self.assertEqual([item[0] for item in results], list(range(len(cases))))
        self.assertTrue(all(item[2] for item in results))
        self.assertGreaterEqual(len(pipeline.persistent_worker_pids), 1)
        self.assertLessEqual(len(pipeline.persistent_worker_pids), 4)

    def test_close_terminates_all_owned_workers(self) -> None:
        pipeline = BCNLTSourcePipeline(self.binding, persistent_sidecars=True)
        pipeline.prepare_semantic(
            SAMPLES["python"], language="python", path="close.py"
        )
        self.assertTrue(pipeline.persistent_worker_pids)
        pipeline.close()
        self.assertEqual(pipeline.persistent_worker_pids, ())

    def test_zero_width_bash_grammar_sentinels_are_not_semantic_records(self) -> None:
        pipeline = BCNLTSourcePipeline(self.binding, persistent_sidecars=True)
        self.addCleanup(pipeline.close)
        source = (
            "if [[ -z \"${GP// }\" ]]; then\n"
            "  cat <<EOF\n"
            "value\n"
            "EOF\n"
            "fi\n"
        )
        prepared = pipeline.prepare_semantic(
            source,
            language="bash",
            path="zero-width.sh",
        )
        records = [json.loads(line) for line in prepared.canonical_jsonl.splitlines()]
        self.assertTrue(records)
        self.assertTrue(all(record["surface"] for record in records))
        self.assertFalse(
            any(
                record["span"][0:2] == record["span"][2:4]
                and record["event"] == "UNKNOWN_SEMANTIC"
                for record in records
            )
        )


if __name__ == "__main__":
    unittest.main()
