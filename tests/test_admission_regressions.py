import unittest

from bosscrafts_model.config import ModelConfig
from bosscrafts_model.evidence import EvidenceLedger


class AdmissionRegressionTests(unittest.TestCase):
    def test_missing_manifest_fails_closed(self):
        for missing in (None,""," ","UNBOUND",False):
            with self.assertRaises(ValueError):
                ModelConfig(tokenizer_manifest=missing).check_manifest(missing)

    def test_invalid_required_checks_fail_closed(self):
        ledger=EvidenceLedger()
        ledger.record_tool_result(task_id="task",artifact_sha256="a"*64,
                                  check_id="tests",passed=True,output="ok")
        for checks in (iter([]),iter(["tests"]),"tests",None,[],(),[""],[" "],[True]):
            self.assertFalse(ledger.can_mark_verified("task","a"*64,checks))
        self.assertTrue(ledger.can_mark_verified("task","a"*64,("tests",)))
