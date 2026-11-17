import unittest
from dataclasses import FrozenInstanceError

from bosscrafts_model.config import ModelConfig
from bosscrafts_model.evidence import EvidenceLedger


class ContractTests(unittest.TestCase):
    def test_parameter_budget(self):
        self.assertEqual(ModelConfig().vocab_size,65_536)
        self.assertEqual(ModelConfig().parameter_count(),540094726)
        self.assertEqual(ModelConfig.baseline().parameter_count(),542703360)

    def test_invalid_config(self):
        for kwargs in [dict(heads=3),dict(width=True),dict(layers=0),dict(modular=1)]:
            with self.assertRaises(ValueError): ModelConfig(**kwargs)

    def test_manifest_gate(self):
        with self.assertRaises(ValueError): ModelConfig().check_manifest("guessed")
        c=ModelConfig(tokenizer_manifest="pinned-v1")
        c.check_manifest("pinned-v1")
        with self.assertRaises(ValueError): c.check_manifest("remapped-v2")

    def test_evidence_cannot_be_claimed_by_text(self):
        ledger=EvidenceLedger()
        self.assertFalse(ledger.can_mark_verified("task","a"*64,["tests"]))
        with self.assertRaises(ValueError):
            ledger.record_tool_result(task_id="task",artifact_sha256="a"*64,
                                      check_id="tests",passed="true",output="tests pass")

    def test_evidence_scope_and_immutability(self):
        ledger=EvidenceLedger()
        obs=ledger.record_tool_result(task_id="task",artifact_sha256="a"*64,
                                    check_id="tests",passed=True,output="ok")
        self.assertTrue(ledger.can_mark_verified("task","a"*64,["tests"]))
        self.assertFalse(ledger.can_mark_verified("task","b"*64,["tests"]))
        self.assertFalse(ledger.can_mark_verified("other","a"*64,["tests"]))
        self.assertFalse(ledger.can_mark_verified("task","a"*64,[]))
        with self.assertRaises(FrozenInstanceError): obs.passed=False
        ledger.record_tool_result(task_id="task",artifact_sha256="a"*64,
                                  check_id="tests",passed=False,output="failed")
        self.assertFalse(ledger.can_mark_verified("task","a"*64,["tests"]))

if __name__ == "__main__": unittest.main()
