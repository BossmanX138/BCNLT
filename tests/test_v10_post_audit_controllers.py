import unittest

from scripts import audit_v10_task_loss_masks_v2_code_audit as loss_mask_controller
from scripts import (
    materialize_v10_evaluation_eligibility_v2_code_audit as evaluation_eligibility_controller,
)
from scripts import (
    materialize_v10_training_eligibility_v2_code_audit as eligibility_controller,
)
from scripts import pack_v10_audited_training_tasks_v2_code_audit as packing_controller
from scripts import (
    pack_v10_frozen_evaluation_tasks_v2_code_audit as evaluation_packing_controller,
)
from scripts import (
    run_v10_frozen_evaluation_admission_v2_code_audit as evaluation_admission_controller,
)
from scripts import (
    run_v10_frozen_evaluation_semantic_audit_v2_code_audit as evaluation_audit_controller,
)
from scripts import run_v10_training_semantic_audit_v2_code_audit as audit_controller


class V10PostAuditControllerTests(unittest.TestCase):
    def test_audit_identity_inventory_is_shared_exactly(self):
        self.assertEqual(
            eligibility_controller.AUDIT_HOST_CODE_FILES,
            audit_controller.AUDIT_HOST_CODE_FILES,
        )
        self.assertEqual(
            eligibility_controller.audit_host_code_identity(),
            audit_controller._audit_host_code_identity(),
        )

    def test_v2_paths_form_one_isolated_pipeline(self):
        self.assertEqual(
            eligibility_controller.DEFAULT_ADMISSION,
            audit_controller.DEFAULT_ADMISSION,
        )
        self.assertEqual(
            eligibility_controller.DEFAULT_AUDIT,
            audit_controller.DEFAULT_AUDIT,
        )
        self.assertEqual(
            packing_controller.DEFAULT_ADMISSION,
            audit_controller.DEFAULT_ADMISSION,
        )
        self.assertEqual(
            packing_controller.DEFAULT_ELIGIBILITY,
            eligibility_controller.DEFAULT_OUTPUT,
        )
        self.assertEqual(
            loss_mask_controller.DEFAULT_TRAIN,
            packing_controller.DEFAULT_DATABASE,
        )
        for path in (
            audit_controller.DEFAULT_AUDIT,
            eligibility_controller.DEFAULT_OUTPUT,
            packing_controller.DEFAULT_DATABASE,
            loss_mask_controller.DEFAULT_OUTPUT,
        ):
            self.assertIn("v2-code-audit", path.as_posix())

    def test_frozen_evaluation_tasks_remain_preserved_inputs(self):
        expected_root = evaluation_packing_controller.PROJECT_ROOT / "artifacts" / "corpora"
        self.assertEqual(
            loss_mask_controller.DEFAULT_VALIDATION,
            expected_root
            / "bcnlt-v10-frozen-validation-tasks-v2-code-audit"
            / "tasks.sqlite",
        )
        self.assertEqual(
            loss_mask_controller.DEFAULT_TEST,
            expected_root
            / "bcnlt-v10-frozen-test-tasks-v2-code-audit"
            / "tasks.sqlite",
        )

    def test_evaluation_pipeline_uses_post_audit_freeze(self):
        self.assertEqual(
            evaluation_admission_controller.DEFAULT_FREEZE,
            audit_controller.DEFAULT_FREEZE,
        )
        self.assertEqual(
            evaluation_admission_controller.FREEZE_MANIFEST_SHA256,
            audit_controller.SOURCE_FREEZE_MANIFEST_SHA256,
        )
        self.assertEqual(
            evaluation_audit_controller.DEFAULT_FREEZE,
            audit_controller.DEFAULT_FREEZE,
        )
        self.assertEqual(
            evaluation_audit_controller.QUALIFICATION_FILE_SHA256,
            audit_controller.QUALIFICATION_FILE_SHA256,
        )
        self.assertIs(
            evaluation_eligibility_controller._audit_inventory,
            evaluation_audit_controller._audit_inventory,
        )


if __name__ == "__main__":
    unittest.main()
