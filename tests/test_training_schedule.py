from __future__ import annotations

import math
import unittest


class TrainingScheduleTests(unittest.TestCase):
    def test_warmup_and_cosine_decay_reach_the_declared_endpoints(self):
        from bosscrafts_model.training_schedule import cosine_learning_rate

        values = [
            cosine_learning_rate(
                step,
                total_steps=10,
                peak_learning_rate=0.001,
                warmup_steps=2,
                minimum_ratio=0.1,
            )
            for step in range(10)
        ]
        self.assertAlmostEqual(values[0], 0.0005)
        self.assertAlmostEqual(values[1], 0.001)
        self.assertAlmostEqual(values[-1], 0.0001)
        self.assertTrue(all(math.isfinite(value) and value > 0 for value in values))
        self.assertTrue(
            all(left >= right for left, right in zip(values[1:], values[2:]))
        )

    def test_schedule_rejects_steps_outside_the_frozen_plan(self):
        from bosscrafts_model.training_schedule import cosine_learning_rate

        with self.assertRaisesRegex(ValueError, "step_index"):
            cosine_learning_rate(
                3,
                total_steps=3,
                peak_learning_rate=0.001,
                warmup_steps=0,
                minimum_ratio=0.1,
            )


if __name__ == "__main__":
    unittest.main()
