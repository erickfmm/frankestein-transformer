"""Unit tests for the CLI argument parser (no torch required)."""
import unittest
from argparse import ArgumentError
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.cli import build_parser, main


class BuildParserStructureTests(unittest.TestCase):
    def setUp(self):
        self.parser = build_parser()

    def test_parser_has_subcommands(self):
        # Verify that the parser accepts known subcommands without error.
        self.assertIsNotNone(self.parser)

    def test_train_subcommand_defaults(self):
        args = self.parser.parse_args(["train"])
        self.assertEqual(args.command, "train")
        self.assertEqual(args.device, "auto")
        self.assertIsNone(args.gpu_temp_guard)
        self.assertIsNone(args.batch_size)
        self.assertIsNone(args.model_mode)

    def test_train_subcommand_device_choices(self):
        for device in ("auto", "cpu", "cuda", "mps"):
            args = self.parser.parse_args(["train", "--device", device])
            self.assertEqual(args.device, device)

    def test_train_gpu_temp_guard_flag(self):
        args = self.parser.parse_args(["train", "--gpu-temp-guard"])
        self.assertTrue(args.gpu_temp_guard)

    def test_train_no_gpu_temp_guard_flag(self):
        args = self.parser.parse_args(["train", "--no-gpu-temp-guard"])
        self.assertFalse(args.gpu_temp_guard)

    def test_train_gpu_temp_guard_flags_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["train", "--gpu-temp-guard", "--no-gpu-temp-guard"])

    def test_train_switch_on_thermal_flag(self):
        args = self.parser.parse_args(["train", "--switch-on-thermal"])
        self.assertTrue(args.switch_on_thermal)

    def test_train_no_switch_on_thermal_flag(self):
        args = self.parser.parse_args(["train", "--no-switch-on-thermal"])
        self.assertFalse(args.switch_on_thermal)

    def test_train_switch_on_thermal_flags_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args(["train", "--switch-on-thermal", "--no-switch-on-thermal"])

    def test_train_gpu_temp_thresholds(self):
        args = self.parser.parse_args([
            "train",
            "--gpu-temp-pause-threshold-c", "91.5",
            "--gpu-temp-resume-threshold-c", "80.0",
            "--gpu-temp-critical-threshold-c", "95.0",
            "--gpu-temp-poll-interval-seconds", "15.0",
        ])
        self.assertAlmostEqual(args.gpu_temp_pause_threshold_c, 91.5)
        self.assertAlmostEqual(args.gpu_temp_resume_threshold_c, 80.0)
        self.assertAlmostEqual(args.gpu_temp_critical_threshold_c, 95.0)
        self.assertAlmostEqual(args.gpu_temp_poll_interval_seconds, 15.0)

    def test_train_model_mode_choices(self):
        for mode in ("frankenstein", "mini", "frankesteindecoder"):
            args = self.parser.parse_args(["train", "--model-mode", mode])
            self.assertEqual(args.model_mode, mode)

    def test_deploy_requires_checkpoint_and_output(self):
        args = self.parser.parse_args([
            "deploy",
            "--checkpoint", "/tmp/ckpt.pt",
            "--output", "/tmp/out",
        ])
        self.assertEqual(args.command, "deploy")
        self.assertEqual(args.checkpoint, "/tmp/ckpt.pt")

    def test_deploy_format_default(self):
        args = self.parser.parse_args([
            "deploy",
            "--checkpoint", "/tmp/ckpt.pt",
            "--output", "/tmp/out",
        ])
        self.assertEqual(args.format, "quantized")

    def test_deploy_format_choices(self):
        for fmt in ("quantized", "standard"):
            args = self.parser.parse_args([
                "deploy",
                "--checkpoint", "/tmp/ckpt.pt",
                "--output", "/tmp/out",
                "--format", fmt,
            ])
            self.assertEqual(args.format, fmt)

    def test_quantize_subcommand(self):
        args = self.parser.parse_args([
            "quantize",
            "--checkpoint", "/tmp/ckpt.pt",
            "--output", "/tmp/out",
        ])
        self.assertEqual(args.command, "quantize")

    def test_infer_subcommand_defaults(self):
        args = self.parser.parse_args(["infer", "--model", "/tmp/model"])
        self.assertEqual(args.command, "infer")
        self.assertIsNone(args.text)
        self.assertIsNone(args.input)
        self.assertEqual(args.batch_size, 8)
        self.assertFalse(args.fp16)
        self.assertFalse(args.benchmark)

    def test_sbert_train_defaults(self):
        args = self.parser.parse_args(["sbert-train"])
        self.assertEqual(args.command, "sbert-train")
        self.assertIsNone(args.base_model)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.pooling_mode, "mean")
        self.assertFalse(args.no_amp)

    def test_sbert_train_pooling_mode_choices(self):
        for mode in ("mean", "cls", "max"):
            args = self.parser.parse_args(["sbert-train", "--pooling_mode", mode])
            self.assertEqual(args.pooling_mode, mode)

    def test_sbert_infer_requires_model_path_and_mode(self):
        args = self.parser.parse_args([
            "sbert-infer",
            "--model_path", "/tmp/sbert",
            "--mode", "similarity",
        ])
        self.assertEqual(args.command, "sbert-infer")
        self.assertEqual(args.mode, "similarity")

    def test_sbert_infer_mode_choices(self):
        for mode in ("similarity", "search", "cluster", "encode"):
            args = self.parser.parse_args([
                "sbert-infer",
                "--model_path", "/tmp/sbert",
                "--mode", mode,
            ])
            self.assertEqual(args.mode, mode)

    def test_web_server_defaults(self):
        args = self.parser.parse_args(["web-server"])
        self.assertEqual(args.command, "web-server")
        self.assertEqual(args.server_port, 8501)
        self.assertEqual(args.server_address, "localhost")
        self.assertFalse(args.server_headless)
        self.assertFalse(args.development_mode)

    def test_missing_required_subcommand_exits(self):
        with self.assertRaises(SystemExit):
            self.parser.parse_args([])


class MainDispatchTests(unittest.TestCase):
    def test_train_dispatches_to_training_main(self):
        mocked = Mock(return_value=0)
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.training.main": fake_mod}):
            rc = main(["train"])
        self.assertEqual(rc, 0)
        mocked.assert_called_once()

    def test_deploy_dispatches_to_deploy_main(self):
        mocked = Mock(return_value=0)
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.deploy.deploy": fake_mod}):
            rc = main(["deploy", "--checkpoint", "/tmp/ckpt.pt", "--output", "/tmp/out"])
        self.assertEqual(rc, 0)
        mocked.assert_called_once()

    def test_quantize_dispatches_to_deploy_main_with_quantized_format(self):
        mocked = Mock(return_value=0)
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.deploy.deploy": fake_mod}):
            rc = main(["quantize", "--checkpoint", "/tmp/ckpt.pt", "--output", "/tmp/out"])
        self.assertEqual(rc, 0)
        forwarded = mocked.call_args[0][0]
        self.assertIn("quantized", forwarded)

    def test_infer_dispatches_to_infer_main(self):
        mocked = Mock(return_value=0)
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.deploy.inference": fake_mod}):
            rc = main(["infer", "--model", "/tmp/model"])
        self.assertEqual(rc, 0)
        mocked.assert_called_once()

    def test_return_code_zero_int(self):
        mocked = Mock(return_value=None)  # returns None, should convert to 0
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.training.main": fake_mod}):
            rc = main(["train"])
        self.assertEqual(rc, 0)

    def test_return_code_nonzero_propagated(self):
        mocked = Mock(return_value=1)
        fake_mod = SimpleNamespace(main=mocked)
        with patch.dict("sys.modules", {"src.training.main": fake_mod}):
            rc = main(["train"])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
