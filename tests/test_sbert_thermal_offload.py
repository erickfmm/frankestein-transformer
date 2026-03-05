import json
import os
import tempfile
import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import Mock, patch

SBERT_TEST_DEPS_AVAILABLE = all(
    find_spec(module_name) is not None
    for module_name in ("torch", "datasets", "sentence_transformers")
)

if SBERT_TEST_DEPS_AVAILABLE:
    from torch.utils.data import DataLoader
    from src.sbert.train_sbert import SBERTTrainer, _ThermalGuardedDataLoader


@unittest.skipUnless(
    SBERT_TEST_DEPS_AVAILABLE,
    "torch + datasets + sentence_transformers are required for SBERT thermal tests",
)
class SbertThermalOffloadTests(unittest.TestCase):
    def test_critical_temperature_invokes_offload_callback(self):
        dataloader = DataLoader(
            [1, 2],
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )
        guard = SimpleNamespace(is_active=True)
        callback = Mock(side_effect=["thermal_offload_cpu_mode", "thermal_cpu_training_1s"])
        wrapped = _ThermalGuardedDataLoader(
            dataloader=dataloader,
            guard=guard,
            context_prefix="unit.sbert",
            on_thermal_check=callback,
        )

        _ = list(wrapped)

        self.assertEqual(callback.call_count, 2)
        callback.assert_any_call(0)
        callback.assert_any_call(1)

    def test_noncritical_pause_uses_wait_path(self):
        dataloader = DataLoader(
            [1],
            batch_size=1,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )
        pause_result = SimpleNamespace(paused=True, repair_action="thermal_pause_5s", temp_c=80.0)
        guard = SimpleNamespace(
            is_active=True,
            critical_threshold_c=95.0,
            pause_threshold_c=90.0,
            read_temperature_c=Mock(return_value=91.0),
            wait_until_safe=Mock(return_value=pause_result),
        )
        callback = Mock(return_value="should_not_run")
        wrapped = _ThermalGuardedDataLoader(
            dataloader=dataloader,
            guard=guard,
            context_prefix="unit.sbert",
            on_critical_temperature=callback,
        )

        _ = list(wrapped)

        callback.assert_not_called()
        guard.wait_until_safe.assert_called_once()

    def test_thermal_artifacts_include_model_and_resume_copy_with_pruning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoints_root = os.path.join(tmpdir, "checkpoints", "checkpoint-10")
            os.makedirs(checkpoints_root, exist_ok=True)
            with open(
                os.path.join(checkpoints_root, "trainer_state.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump({"global_step": 10}, handle)
            with open(os.path.join(checkpoints_root, "weights.bin"), "w", encoding="utf-8") as handle:
                handle.write("weights")

            trainer = SBERTTrainer.__new__(SBERTTrainer)
            trainer.output_dir = tmpdir
            trainer.checkpoint_save_total_limit = 2
            trainer.thermal_emergency_dir = os.path.join(tmpdir, "thermal_emergency")
            trainer._switch_on_thermal = True
            trainer._force_cpu_only_after_gpu_error = False
            os.makedirs(trainer.thermal_emergency_dir, exist_ok=True)

            def _mock_save(path):
                os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, "model.txt"), "w", encoding="utf-8") as handle:
                    handle.write("snapshot")

            trainer.model = SimpleNamespace(save=_mock_save, to=Mock())

            first_snapshot = SBERTTrainer._save_thermal_model_snapshot(trainer, batch_idx=0, temp_c=95.0)
            resume_copy = SBERTTrainer._save_thermal_resume_artifact_if_available(trainer, batch_idx=0)
            second_snapshot = SBERTTrainer._save_thermal_model_snapshot(trainer, batch_idx=1, temp_c=96.0)

            self.assertTrue(first_snapshot.startswith(trainer.thermal_emergency_dir))
            self.assertIsNotNone(resume_copy)
            self.assertTrue(os.path.isdir(resume_copy))
            self.assertTrue(os.path.isdir(second_snapshot))
            self.assertTrue(os.path.exists(os.path.join(resume_copy, "weights.bin")))

            entries = os.listdir(trainer.thermal_emergency_dir)
            self.assertLessEqual(len(entries), 2)

    def test_sbert_offload_trains_on_cpu_until_resume_then_reloads_gpu(self):
        trainer = SBERTTrainer.__new__(SBERTTrainer)
        trainer._switch_on_thermal = True
        trainer._thermal_offload_active = False
        trainer._thermal_offload_started_monotonic = None
        trainer._thermal_last_poll_monotonic = 0.0
        trainer._thermal_last_model_snapshot = None
        trainer._thermal_last_resume_artifact = None
        trainer._force_cpu_only_after_gpu_error = False
        trainer.device = "cuda"
        trainer.model = SimpleNamespace(to=Mock(), save=Mock())
        trainer._save_thermal_model_snapshot = Mock(return_value="/tmp/model_snapshot")
        trainer._save_thermal_resume_artifact_if_available = Mock(return_value="/tmp/resume_artifact")
        trainer.gpu_temp_guard = SimpleNamespace(
            resume_threshold_c=80.0,
            poll_interval_seconds=0.01,
            read_temperature_c=Mock(return_value=79.0),
            is_active=True,
        )

        with patch("src.sbert.train_sbert.time.perf_counter", return_value=100.0):
            action = SBERTTrainer._offload_model_for_critical_thermal_event(
                trainer,
                batch_idx=0,
                temp_c=96.0,
            )

        self.assertEqual(action, "thermal_offload_cpu_mode")
        trainer.model.to.assert_called_with("cpu")
        self.assertTrue(trainer._thermal_offload_active)

        with patch("src.sbert.train_sbert.time.perf_counter", return_value=110.0):
            resume_action = SBERTTrainer._monitor_thermal_offload_and_maybe_reload(
                trainer,
                batch_idx=1,
            )

        self.assertEqual(resume_action, "thermal_onload_gpu_10s")
        self.assertFalse(trainer._thermal_offload_active)

    def test_gpu_error_forces_cpu_only_mode(self):
        trainer = SBERTTrainer.__new__(SBERTTrainer)
        trainer._switch_on_thermal = True
        trainer._force_cpu_only_after_gpu_error = False
        trainer._thermal_offload_active = False
        trainer._thermal_offload_started_monotonic = None
        trainer._thermal_last_poll_monotonic = 0.0
        trainer.device = "cuda"
        trainer.model = SimpleNamespace(to=Mock())
        trainer._save_thermal_model_snapshot = Mock(return_value="/tmp/model_snapshot")
        trainer._save_thermal_resume_artifact_if_available = Mock(return_value="/tmp/resume_artifact")

        handled = SBERTTrainer._switch_to_cpu_only_after_gpu_error(
            trainer,
            RuntimeError("CUDA error: device-side assert triggered"),
        )

        self.assertTrue(handled)
        self.assertTrue(trainer._force_cpu_only_after_gpu_error)
        trainer.model.to.assert_called_with("cpu")


if __name__ == "__main__":
    unittest.main()
