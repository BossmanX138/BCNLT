from __future__ import annotations

import unittest

from bosscrafts_model.bcnlt_source_pipeline import BCNLTSourcePipeline
from bosscrafts_model.tokenizer import NativeTokenizerBinding


class InferenceProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binding = NativeTokenizerBinding.load()
        cls.pipeline = BCNLTSourcePipeline(cls.binding)

    def test_code_prompt_matches_the_training_task_framing(self):
        from bosscrafts_model.inference_protocol import build_inference_prompt

        ids = self.binding.structural_token_ids
        prompt = build_inference_prompt(
            self.binding,
            task="code_to_semantic",
            prompt="x = 1",
        )
        self.assertEqual(
            prompt,
            [
                ids["<|bcnlt:task:code_to_semantic|>"],
                ids["<|bcnlt:stream:begin|>"],
                ids["<|bcnlt:mode:raw_code|>"],
                *self.binding.encode("x = 1"),
                ids["<|bcnlt:stream:end|>"],
            ],
        )

    def test_semantic_tasks_require_a_complete_semantic_stream(self):
        from bosscrafts_model.inference_protocol import build_inference_prompt

        with self.assertRaisesRegex(ValueError, "semantic stream"):
            build_inference_prompt(
                self.binding,
                task="semantic_to_code",
                prompt="plain text",
            )
        prepared = self.pipeline.prepare_semantic(
            "def add(left, right):\n    return left + right\n",
            language="python",
            path="valid.py",
        )
        semantic = self.binding.decode(prepared.token_ids)
        prompt = build_inference_prompt(
            self.binding,
            task="semantic_to_code",
            prompt=semantic,
        )
        self.assertEqual(
            prompt[0],
            self.binding.structural_token_ids["<|bcnlt:task:semantic_to_code|>"],
        )

    def test_reserved_protocol_text_is_escaped_inside_raw_code(self):
        from bosscrafts_model.inference_protocol import build_inference_prompt

        source = "print('<|bcnlt:task:end|>')"
        prompt = build_inference_prompt(
            self.binding,
            task="code_to_semantic",
            prompt=source,
        )
        payload = prompt[3:-1]
        self.assertEqual(self.binding.decode(payload), source)
        self.assertTrue(all(token_id > 76 for token_id in payload))

    def test_semantic_prompt_ids_preserve_literal_control_spelling(self):
        from bosscrafts_model.inference_protocol import build_inference_prompt

        prepared = self.pipeline.prepare_semantic(
            "print('<|bcnlt:record:end|>')\n",
            language="python",
            path="literal-control.py",
        )
        prompt = build_inference_prompt(
            self.binding,
            task="semantic_to_code",
            prompt=prepared.token_ids,
        )
        self.assertEqual(
            prompt[0],
            self.binding.structural_token_ids["<|bcnlt:task:semantic_to_code|>"],
        )
        self.assertEqual(tuple(prompt[1:]), prepared.token_ids)

    def test_instruction_prompt_uses_mixed_instruction_raw_stream(self):
        from bosscrafts_model.inference_protocol import build_inference_prompt

        ids = self.binding.structural_token_ids
        prompt = build_inference_prompt(
            self.binding,
            task="instruction_to_code",
            prompt="def add(a, b):\n    \"\"\"Return their sum.\"\"\"\n",
        )
        self.assertEqual(prompt[0], ids["<|bcnlt:task:mixed_instruction|>"])
        self.assertEqual(prompt[1], ids["<|bcnlt:stream:begin|>"])
        self.assertEqual(prompt[2], ids["<|bcnlt:mode:raw_code|>"])
        self.assertEqual(prompt[-1], ids["<|bcnlt:stream:end|>"])

    def test_instruction_response_extracts_only_complete_code_payload(self):
        from bosscrafts_model.inference_protocol import response_payload_token_ids

        ids = self.binding.structural_token_ids
        code = self.binding.encode("    return a + b\n")
        generated = [
            ids["<|bcnlt:stream:begin|>"],
            ids["<|bcnlt:mode:raw_code|>"],
            *code,
            ids["<|bcnlt:stream:end|>"],
            ids["<|bcnlt:task:end|>"],
        ]
        self.assertEqual(
            response_payload_token_ids(
                self.binding, "instruction_to_code", generated
            ),
            code,
        )

    def test_response_completion_requires_the_expected_stream_and_task_end(self):
        from bosscrafts_model.inference_protocol import response_protocol_complete

        ids = self.binding.structural_token_ids
        prepared = self.pipeline.prepare_semantic(
            "def add(left, right):\n    return left + right\n",
            language="python",
            path="valid.py",
        )
        valid = [*prepared.token_ids, ids["<|bcnlt:task:end|>"]]
        self.assertTrue(
            response_protocol_complete(
                self.binding, task="code_to_semantic", generated_ids=valid
            )
        )
        self.assertFalse(
            response_protocol_complete(
                self.binding,
                task="code_to_semantic",
                generated_ids=[ids["<|bcnlt:stream:begin|>"]] * 8,
            )
        )

    def test_selects_a_bounded_heldout_case_from_exact_packed_framing(self):
        from bosscrafts_model.inference_protocol import select_heldout_inference_case

        examples = [
            {
                "example_id": "a" * 64,
                "family": "code_to_semantic",
                "input_ids": [1, 2, 3, 4, 5, 6],
                "loss_start_index": 2,
            },
            {
                "example_id": "b" * 64,
                "family": "semantic_to_code",
                "input_ids": [7, 8, 9, 10],
                "loss_start_index": 2,
            },
            {
                "example_id": "c" * 64,
                "family": "code_to_semantic",
                "input_ids": [11, 12, 13, 14, 15],
                "loss_start_index": 3,
            },
        ]
        case = select_heldout_inference_case(
            examples,
            family="code_to_semantic",
            max_new_tokens=3,
            maximum_sequence=16,
        )
        self.assertEqual(case.example_id, "c" * 64)
        self.assertEqual(case.prompt_ids, (11, 12, 13))
        self.assertEqual(case.expected_ids, (14, 15))

        with self.assertRaisesRegex(ValueError, "eligible held-out"):
            select_heldout_inference_case(
                examples,
                family="semantic_continuation",
                max_new_tokens=3,
                maximum_sequence=16,
            )


if __name__ == "__main__":
    unittest.main()
