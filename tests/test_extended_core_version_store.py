from __future__ import annotations

import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

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
from bcnlt.extended_core_registry import (
    ExtendedCorePromotionCandidate,
    ExtendedCoreVersionStore,
)
from bcnlt.scientific_learning import (
    CodingExperimentFixture,
    HypothesisProjection,
    SandboxBackedProjectionVerifier,
)
from bcnlt.tokenizer_manifest import BaseTokenizerIdentity, TokenizerManifest
from bcnlt.wildcard import ProvisionalMeaning, WildcardMemory

from bosscrafts_model.scientific_sandbox import DockerScientificSandbox


class ExtendedCoreVersionStoreTests(unittest.TestCase):
    @staticmethod
    def _v1_manifest() -> TokenizerManifest:
        identity = BaseTokenizerIdentity.from_files(
            name="test-base",
            revision="immutable-test-revision",
            files={"tokenizer.json": b"immutable-test-tokenizer"},
        )
        return TokenizerManifest(
            schema_version="bcnlt-tokenizer-v1",
            canonicalization_version="bcnlt-canonical-record-v1",
            base_tokenizer=identity,
            base_tokenizer_sha256=identity.sha256,
            base_vocab_size=32,
            base_vocabulary_sha256="a" * 64,
            structural_inventory_sha256="b" * 64,
            added_tokens=(),
            manifest_sha256="c" * 64,
        )

    @staticmethod
    def _unit(surface: str, relation: str, scope: str, line: int) -> ContextualUnit:
        return ContextualUnit(
            surface=surface,
            event="UNKNOWN_SEMANTIC",
            scope=scope,
            span=(line, 0, line, len(surface)),
            ast_relation=relation,
            recent_history=("operand", "operator", scope),
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

    def _attest(
        self,
        memory: WildcardMemory,
        *,
        surface: str,
        relation: str,
        seed: int,
    ) -> None:
        first = self._unit(surface, relation, f"module.{seed}.first", seed)
        second = self._unit(surface, relation, f"module.{seed}.second", seed + 20)
        record = memory.remember(
            first,
            self._meaning(),
            document_identity=f"document-{seed}-one",
        )
        memory.remember(
            second,
            self._meaning(),
            document_identity=f"document-{seed}-two",
        )
        control = HypothesisProjection(
            signature=record.signature,
            revision=record.hypothesis_revision,
            target_event="RETURN",
            target_ir_kind="return",
            language=record.language,
            semantic_spec_sha256=record.semantic_spec_sha256,
        )
        verifier = SandboxBackedProjectionVerifier(
            DockerScientificSandbox.load_default()
        )
        source = (
            "import json, sys\n"
            "payload = json.load(sys.stdin)\n"
            "print(json.dumps(payload['left'] + payload['right']))\n"
        )
        for index, unit in enumerate((first, second), start=1):
            left = seed + index
            right = seed + index + 1
            memory.run_scientific_experiment(
                unit,
                source=f"# independent source {seed}-{index}\n{source}",
                fixture=CodingExperimentFixture(
                    experiment_id=f"registry-{seed}-{index}",
                    fixture_id=f"registry-vector-{seed}-{index}",
                    fixture_version=1,
                    domain="coding",
                    document_identity=f"document-{seed}-{'one' if index == 1 else 'two'}",
                    oracle_id=f"human-registry-vector-{seed}-{index}",
                    oracle_kind="human_authored",
                    oracle_version="1",
                    operation="binary_operator",
                    inputs={"left": left, "operator": "+", "right": right},
                    expected_output=left + right,
                ),
                negative_control=control,
                verifier=verifier,
            )

    def test_candidate_rejects_memory_without_sandbox_reproductions(self) -> None:
        memory = WildcardMemory(
            language="python",
            semantic_spec_sha256="d" * 64,
            minimum_reproductions=2,
        )
        memory.remember(
            self._unit("unverified", "unknown.operator", "module", 1),
            self._meaning(),
            document_identity="unverified-document",
        )

        with self.assertRaisesRegex(ValueError, "sandbox-attested"):
            ExtendedCorePromotionCandidate.from_memory(memory)

    @pytest.mark.requires_docker
    def test_versions_validate_activate_and_rollback_without_mutation(self) -> None:
        memory = WildcardMemory(
            language="python",
            semantic_spec_sha256="e" * 64,
            minimum_reproductions=2,
        )
        self._attest(
            memory,
            surface="mystery_binary_one",
            relation="binary_expression.operator.one",
            seed=10,
        )

        with tempfile.TemporaryDirectory() as directory:
            store = ExtendedCoreVersionStore(Path(directory), self._v1_manifest())
            first = store.stage(ExtendedCorePromotionCandidate.from_memory(memory))
            self.assertEqual(first.state, "staged")
            self.assertEqual(first.version_number, 1)
            self.assertIsNone(first.parent_version_sha256)
            first_receipt = store.validate(
                first.version_sha256,
                validation_report_sha256="1" * 64,
            )
            self.assertEqual(first_receipt.state, "validated")
            active_first = store.activate(first.version_sha256)
            self.assertEqual(active_first.state, "active")
            self.assertEqual(active_first.version_sha256, first.version_sha256)
            first_vocab = store.load_vocab_manifest(first.version_sha256)
            first_files = store.version_file_hashes(first.version_sha256)

            self._attest(
                memory,
                surface="mystery_binary_two",
                relation="binary_expression.operator.two",
                seed=40,
            )
            second = store.stage(ExtendedCorePromotionCandidate.from_memory(memory))
            self.assertEqual(second.version_number, 2)
            self.assertEqual(second.parent_version_sha256, first.version_sha256)
            store.validate(
                second.version_sha256,
                validation_report_sha256="2" * 64,
            )
            store.activate(second.version_sha256)
            second_vocab = store.load_vocab_manifest(second.version_sha256)

            first_ids = {
                token.extended_core_id: token.id for token in first_vocab.tokens
            }
            second_ids = {
                token.extended_core_id: token.id for token in second_vocab.tokens
            }
            self.assertEqual(
                first_ids,
                {key: second_ids[key] for key in first_ids},
            )
            self.assertGreater(len(second_vocab.tokens), len(first_vocab.tokens))

            rolled_back = store.rollback()
            self.assertEqual(rolled_back.action, "rollback")
            self.assertEqual(rolled_back.version_sha256, first.version_sha256)
            self.assertEqual(store.active().version_sha256, first.version_sha256)
            self.assertEqual(
                store.version_file_hashes(first.version_sha256),
                first_files,
            )
            self.assertTrue(store.version_path(first.version_sha256).is_dir())
            self.assertTrue(store.version_path(second.version_sha256).is_dir())
            store.verify_all()

    @pytest.mark.requires_docker
    def test_concurrent_activation_has_one_event_and_one_winner(self) -> None:
        memory = WildcardMemory(
            language="python",
            semantic_spec_sha256="f" * 64,
            minimum_reproductions=2,
        )
        self._attest(
            memory,
            surface="concurrent_binary",
            relation="binary_expression.operator.concurrent",
            seed=70,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = ExtendedCoreVersionStore(root, self._v1_manifest())
            version = first.stage(ExtendedCorePromotionCandidate.from_memory(memory))
            first.validate(version.version_sha256, validation_report_sha256="3" * 64)
            second = ExtendedCoreVersionStore(root, self._v1_manifest())
            barrier = Barrier(2)

            def activate(store: ExtendedCoreVersionStore) -> str:
                barrier.wait(timeout=5)
                try:
                    return store.activate(version.version_sha256).pointer_sha256
                except ValueError as exc:
                    return f"error:{exc}"

            with ThreadPoolExecutor(max_workers=2) as executor:
                results = tuple(executor.map(activate, (first, second)))

            assert sum(not item.startswith("error:") for item in results) == 1
            assert sum("activation parent" in item for item in results) == 1
            assert len(tuple(first.events.glob("*.json"))) == 1
            assert first.active().sequence == 1
            first.verify_all()


if __name__ == "__main__":
    unittest.main()
