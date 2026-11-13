from __future__ import annotations

import unittest

import torch

from bosscrafts_model.config import ModelConfig
from bosscrafts_model.model import EngineeringModel


class InferenceTests(unittest.TestCase):
    @staticmethod
    def _uncached_generate(model, prompt, max_new_tokens):
        generated = prompt.clone()
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits, _ = model(generated)
                generated = torch.cat(
                    (generated, logits[:, -1].argmax(-1).unsqueeze(1)), dim=1
                )
        return generated

    def test_cached_generation_matches_uncached_greedy_reference(self):
        torch.manual_seed(79)
        model = EngineeringModel(ModelConfig.tiny(max_sequence=10)).eval()
        prompt = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

        expected = self._uncached_generate(model, prompt, 4)
        actual = model.generate(prompt, max_new_tokens=4)

        self.assertTrue(torch.equal(actual, expected))

    def test_target_only_projection_matches_full_causal_logits(self):
        torch.manual_seed(81)
        model = EngineeringModel(ModelConfig.tiny(max_sequence=10)).eval()
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)

        with torch.inference_mode():
            full_logits, _ = model(ids)
            target_logits = model.forward_target_logits(ids, 3)

        torch.testing.assert_close(target_logits, full_logits[:, 2:-1])

    def test_greedy_generation_is_deterministic_bounded_and_preserves_prompt(self):
        torch.manual_seed(83)
        model = EngineeringModel(ModelConfig.tiny(max_sequence=8)).eval()
        generate = getattr(model, "generate", None)
        self.assertIsNotNone(generate, "model requires an inference generation surface")
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

        first = generate(prompt, max_new_tokens=3)
        second = generate(prompt, max_new_tokens=3)

        self.assertTrue(torch.equal(first, second))
        self.assertEqual(tuple(first.shape), (1, 6))
        self.assertTrue(torch.equal(first[:, :3], prompt))
        self.assertTrue(bool(((first >= 0) & (first < model.config.vocab_size)).all()))
        with self.assertRaisesRegex(ValueError, "maximum sequence"):
            generate(prompt, max_new_tokens=6)

    def test_generation_stops_after_predicted_eos_token(self):
        torch.manual_seed(89)
        model = EngineeringModel(ModelConfig.tiny(max_sequence=8)).eval()
        generate = getattr(model, "generate", None)
        self.assertIsNotNone(generate)
        prompt = torch.tensor([[4, 5]], dtype=torch.long)
        with torch.inference_mode():
            eos_token = int(model(prompt)[0][:, -1].argmax(-1).item())

        generated = generate(prompt, max_new_tokens=4, eos_token_id=eos_token)

        self.assertEqual(tuple(generated.shape), (1, 3))
        self.assertEqual(int(generated[0, -1]), eos_token)


if __name__ == "__main__":
    unittest.main()
