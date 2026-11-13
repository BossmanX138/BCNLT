import unittest

from bosscrafts_model.instruction_curriculum import (
    extract_python_instruction_candidates,
)


class InstructionCurriculumExtractionTests(unittest.TestCase):
    def _extract(self, source: str):
        return extract_python_instruction_candidates(
            document_id="a" * 64,
            relative_path="sample.py",
            repository_identity="fixture/repository",
            license_spdx="MIT",
            source_sha256="b" * 64,
            source=source,
        )

    def test_extracts_signature_docstring_prompt_and_body(self):
        rows = self._extract(
            'from math import sqrt\n\n'
            'def hypotenuse(a: float, b: float) -> float:\n'
            '    """Return the hypotenuse for two perpendicular sides."""\n'
            '    return sqrt(a * a + b * b)\n'
        )
        self.assertEqual(len(rows), 1)
        self.assertIn("from math import sqrt", rows[0].prompt)
        self.assertIn("def hypotenuse", rows[0].prompt)
        self.assertNotIn("return sqrt", rows[0].prompt)
        self.assertEqual(rows[0].target, "    return sqrt(a * a + b * b)\n")

    def test_dedents_methods_to_standalone_functions(self):
        rows = self._extract(
            'class Counter:\n'
            '    def increment(self, value: int) -> int:\n'
            '        """Increase a supplied integer by exactly one."""\n'
            '        return value + 1\n'
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0].prompt.startswith("def increment"))
        self.assertEqual(rows[0].target, "    return value + 1\n")

    def test_rejects_functions_without_meaningful_docstrings_or_bodies(self):
        rows = self._extract(
            'def no_doc(value):\n'
            '    return value\n\n'
            'def too_short(value):\n'
            '    """short"""\n'
            '    return value\n\n'
            'def empty(value):\n'
            '    """Documented but deliberately has no implementation."""\n'
            '    pass\n'
        )
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
