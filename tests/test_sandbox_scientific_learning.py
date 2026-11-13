from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_import import select_bcnlt_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCE = (
    PROJECT_ROOT
    / "vendor"
    / "bcnlt_runtime_v10_learning_candidate"
    / "source"
)
select_bcnlt_runtime(RUNTIME_SOURCE)

from bcnlt.core import ContextualUnit
from bcnlt.scientific_learning import (
    CodingExperimentFixture,
    HypothesisProjection,
    SandboxBackedProjectionVerifier,
    ScientificExperimentEvidence,
    VerificationStatus,
)
from bcnlt.wildcard import DefinitionState, ProvisionalMeaning, WildcardMemory

from bosscrafts_model.scientific_sandbox import DockerScientificSandbox


class SandboxScientificLearningTests(unittest.TestCase):
    @staticmethod
    def _unit(scope: str, line: int) -> ContextualUnit:
        return ContextualUnit(
            surface="mystery_binary",
            event="UNKNOWN_SEMANTIC",
            scope=scope,
            span=(line, 0, line, 14),
            ast_relation="binary_expression.operator",
            recent_history=("operand", "operator"),
        )

    @staticmethod
    def _meaning() -> ProvisionalMeaning:
        return ProvisionalMeaning(
            category="operator",
            hypotheses=("binary operator",),
            evidence=("grammar structure",),
            confidence=0.6,
            target_event="OPERATOR",
            target_ir_kind="operator",
            evidence_kinds=("grammar_structure",),
        )

    def test_passed_evidence_requires_real_execution_and_negative_control(self) -> None:
        fixture = CodingExperimentFixture(
            experiment_id="sandbox-addition-1",
            fixture_id="addition-vector",
            fixture_version=1,
            domain="coding",
            document_identity="doc-python-addition",
            oracle_id="human-addition-vector",
            oracle_kind="human_authored",
            oracle_version="1",
            operation="binary_operator",
            inputs={"left": 2, "operator": "+", "right": 3},
            expected_output=5,
        )
        candidate = HypothesisProjection(
            signature="python:binary-expression:add",
            revision=1,
            target_event="OPERATOR",
            target_ir_kind="operator",
            language="python",
            semantic_spec_sha256="a" * 64,
        )
        control = HypothesisProjection(
            signature=candidate.signature,
            revision=candidate.revision,
            target_event="RETURN",
            target_ir_kind="return",
            language=candidate.language,
            semantic_spec_sha256=candidate.semantic_spec_sha256,
        )
        source = (
            "import json, sys\n"
            "payload = json.load(sys.stdin)\n"
            "print(json.dumps(payload['left'] + payload['right']))\n"
        )

        evidence = SandboxBackedProjectionVerifier(
            DockerScientificSandbox.load_default()
        ).verify(
            candidate,
            fixture,
            source=source,
            context_fingerprint="b" * 64,
            negative_control=control,
        )

        self.assertEqual(evidence.status, VerificationStatus.PASSED)
        self.assertEqual(evidence.schema_version, 3)
        self.assertTrue(evidence.intended_passed)
        self.assertTrue(evidence.negative_control_rejected)
        self.assertIsNotNone(evidence.sandbox_execution)
        execution = evidence.sandbox_execution
        assert execution is not None
        self.assertEqual(execution.status, "passed")
        self.assertEqual(execution.actual_output, 5)
        self.assertEqual(execution.network_mode, "none")
        self.assertTrue(execution.read_only_rootfs)
        self.assertEqual(execution.mount_count, 0)
        self.assertEqual(
            ScientificExperimentEvidence.from_dict(evidence.to_dict()),
            evidence,
        )

    def test_wildcard_memory_refuses_the_unattested_default_verifier(self) -> None:
        unit = self._unit("module.first", 1)
        memory = WildcardMemory(
            language="python",
            semantic_spec_sha256="c" * 64,
            minimum_reproductions=2,
        )
        record = memory.remember(
            unit,
            self._meaning(),
            document_identity="document-one",
        )
        fixture = CodingExperimentFixture(
            experiment_id="default-verifier-rejected",
            fixture_id="addition-default",
            fixture_version=1,
            domain="coding",
            document_identity="document-one",
            oracle_id="human-default-vector",
            oracle_kind="human_authored",
            oracle_version="1",
            operation="binary_operator",
            inputs={"left": 1, "operator": "+", "right": 2},
            expected_output=3,
        )
        control = HypothesisProjection(
            signature=record.signature,
            revision=record.hypothesis_revision,
            target_event="RETURN",
            target_ir_kind="return",
            language=record.language,
            semantic_spec_sha256=record.semantic_spec_sha256,
        )

        with self.assertRaisesRegex(RuntimeError, "sandbox-backed verifier"):
            memory.run_scientific_experiment(
                unit,
                source="print(3)\n",
                fixture=fixture,
                negative_control=control,
            )

    def test_two_sandbox_reproductions_promote_and_reload_attested_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "wildcard.jsonl"
            memory = WildcardMemory(
                ledger,
                language="python",
                semantic_spec_sha256="d" * 64,
                minimum_reproductions=2,
            )
            first = self._unit("module.first", 1)
            second = self._unit("module.second", 20)
            record = memory.remember(
                first,
                self._meaning(),
                document_identity="document-one",
            )
            record = memory.remember(
                second,
                self._meaning(),
                document_identity="document-two",
            )
            verifier = SandboxBackedProjectionVerifier(
                DockerScientificSandbox.load_default()
            )
            control = HypothesisProjection(
                signature=record.signature,
                revision=record.hypothesis_revision,
                target_event="RETURN",
                target_ir_kind="return",
                language=record.language,
                semantic_spec_sha256=record.semantic_spec_sha256,
            )
            source_template = (
                "import json, sys\n"
                "payload = json.load(sys.stdin)\n"
                "print(json.dumps(payload['left'] + payload['right']))\n"
            )
            for index, (unit, document, left, right) in enumerate(
                (
                    (first, "document-one", 2, 3),
                    (second, "document-two", 7, 4),
                ),
                start=1,
            ):
                record = memory.run_scientific_experiment(
                    unit,
                    source=f"# independent source {index}\n{source_template}",
                    fixture=CodingExperimentFixture(
                        experiment_id=f"sandbox-reproduction-{index}",
                        fixture_id=f"addition-vector-{index}",
                        fixture_version=1,
                        domain="coding",
                        document_identity=document,
                        oracle_id=f"human-vector-{index}",
                        oracle_kind="human_authored",
                        oracle_version="1",
                        operation="binary_operator",
                        inputs={"left": left, "operator": "+", "right": right},
                        expected_output=left + right,
                    ),
                    negative_control=control,
                    verifier=verifier,
                )

            self.assertEqual(record.definition_state, DefinitionState.VALIDATED)
            self.assertEqual(record.validated_by, "bcnlt-sandbox-backed-projection:1")
            self.assertEqual(len(record.scientific_experiments), 2)
            self.assertTrue(
                all(item.sandbox_execution is not None for item in record.scientific_experiments)
            )
            reloaded = WildcardMemory(
                ledger,
                language="python",
                semantic_spec_sha256="d" * 64,
                minimum_reproductions=2,
                read_only=True,
            ).lookup(first)
            self.assertIsNotNone(reloaded)
            assert reloaded is not None
            self.assertEqual(reloaded.definition_state, DefinitionState.VALIDATED)
            self.assertEqual(reloaded.scientific_experiments, record.scientific_experiments)


if __name__ == "__main__":
    unittest.main()
