from __future__ import annotations

import unittest

import run_two_phase


core = run_two_phase.core


class TwoPhaseBindingTests(unittest.TestCase):
    def test_phase_a_checkpoint_is_the_restored_objective_run(self):
        checkpoint = core.warmstart_path(490)
        self.assertEqual(checkpoint.name, "best_model.pt")
        self.assertEqual(checkpoint.parent.name, "seed_490")
        self.assertEqual(checkpoint.parent.parent.name, "level4_confirmation")
        self.assertTrue(checkpoint.exists())

    def test_phase_a_passes_stricter_high_gate(self):
        audit = core.warmstart_gate_audit(490)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["checkpoint_epoch"], 60)
        self.assertGreaterEqual(
            min(audit["trailing_validation_high_norm_mean"]), 0.9975
        )

    def test_artifacts_are_isolated(self):
        self.assertEqual(core.OUTPUT_ROOT.parent, run_two_phase.THIS_DIR)
        self.assertNotIn("round2_high_frozen", str(core.OUTPUT_ROOT))


if __name__ == "__main__":
    unittest.main()
