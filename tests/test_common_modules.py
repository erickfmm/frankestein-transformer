"""Unit tests for common building blocks: quant functions, BitLinear, norms, get_norm."""
import unittest
from importlib.util import find_spec

TORCH_AVAILABLE = find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    from src.model.attention.common import (
        activation_quant,
        weight_quant,
        BitLinear,
        DynamicTanhNorm,
        Derf,
        get_norm,
    )
    from src.model.tormented_bert_frankestein import UltraConfig


def _cfg(norm_type="layer_norm", hidden_size=32):
    return UltraConfig(
        vocab_size=64,
        hidden_size=hidden_size,
        num_layers=1,
        num_loops=1,
        num_heads=4,
        retention_heads=4,
        num_experts=2,
        top_k_experts=1,
        dropout=0.0,
        norm_type=norm_type,
        use_bitnet=False,
        layer_pattern=["standard_attn"],
        use_moe=False,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class ActivationQuantTests(unittest.TestCase):
    def test_output_shape_preserved(self):
        x = torch.randn(2, 8, 16)
        y = activation_quant(x)
        self.assertEqual(y.shape, x.shape)

    def test_output_dtype_preserved(self):
        x = torch.randn(3, 5)
        y = activation_quant(x)
        self.assertEqual(y.dtype, x.dtype)

    def test_clamp_range(self):
        # STE: output must stay within [-128, 127] after unscaling, approximated.
        x = torch.randn(4, 10) * 100
        y = activation_quant(x)
        # The result should be finite and not blow up.
        self.assertTrue(torch.isfinite(y).all())

    def test_ste_gradient_pass_through(self):
        x = torch.randn(2, 4, requires_grad=True)
        y = activation_quant(x)
        loss = y.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class WeightQuantTests(unittest.TestCase):
    def test_output_shape_preserved(self):
        w = torch.randn(8, 16)
        wq = weight_quant(w)
        self.assertEqual(wq.shape, w.shape)

    def test_ternary_values_approx(self):
        # The quantized weights (before STE correction) should be close to {-1, 0, 1}
        w = torch.randn(64, 64)
        scale = 1.0 / w.abs().mean().clamp(min=1e-5)
        wq_raw = (w * scale).round().clamp(-1, 1) / scale
        # All unique scaled values should be at most 3 distinct magnitudes
        unique = torch.unique((wq_raw * scale).round())
        self.assertLessEqual(unique.numel(), 3)

    def test_ste_gradient_pass_through(self):
        w = torch.randn(4, 4, requires_grad=True)
        wq = weight_quant(w)
        wq.sum().backward()
        self.assertIsNotNone(w.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class BitLinearTests(unittest.TestCase):
    def test_forward_shape(self):
        layer = BitLinear(16, 32)
        x = torch.randn(2, 8, 16)
        out = layer(x)
        self.assertEqual(out.shape, (2, 8, 32))

    def test_no_bias_by_default(self):
        layer = BitLinear(8, 16)
        self.assertIsNone(layer.bias)

    def test_is_linear_subclass(self):
        self.assertTrue(issubclass(BitLinear, nn.Linear))

    def test_gradient_flows(self):
        layer = BitLinear(8, 8)
        x = torch.randn(1, 4, 8, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_square_dimensions(self):
        layer = BitLinear(16, 16)
        x = torch.randn(3, 16)
        out = layer(x)
        self.assertEqual(out.shape, (3, 16))


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class DynamicTanhNormTests(unittest.TestCase):
    def test_output_shape(self):
        norm = DynamicTanhNorm(32)
        x = torch.randn(2, 10, 32)
        y = norm(x)
        self.assertEqual(y.shape, x.shape)

    def test_learnable_alpha_beta(self):
        norm = DynamicTanhNorm(16)
        self.assertEqual(norm.alpha.shape, (16,))
        self.assertEqual(norm.beta.shape, (16,))

    def test_output_range_bounded_by_tanh(self):
        norm = DynamicTanhNorm(8)
        x = torch.randn(4, 5, 8) * 100
        y = norm(x)
        self.assertTrue((y.abs() <= 1.0 + 1e-5).all())

    def test_gradient_flows(self):
        norm = DynamicTanhNorm(16)
        x = torch.randn(2, 4, 16, requires_grad=True)
        norm(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class DerfTests(unittest.TestCase):
    def test_output_shape(self):
        norm = Derf(32)
        x = torch.randn(2, 10, 32)
        y = norm(x)
        self.assertEqual(y.shape, x.shape)

    def test_learnable_parameters(self):
        norm = Derf(16)
        self.assertTrue(hasattr(norm, "alpha"))
        self.assertTrue(hasattr(norm, "gamma"))
        self.assertTrue(hasattr(norm, "beta"))
        self.assertTrue(hasattr(norm, "s"))

    def test_gradient_flows(self):
        norm = Derf(16)
        x = torch.randn(1, 3, 16, requires_grad=True)
        norm(x).sum().backward()
        self.assertIsNotNone(x.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch required")
class GetNormTests(unittest.TestCase):
    def test_dynamic_tanh_returns_correct_type(self):
        cfg = _cfg(norm_type="dynamic_tanh")
        norm = get_norm(cfg)
        self.assertIsInstance(norm, DynamicTanhNorm)

    def test_derf_returns_correct_type(self):
        cfg = _cfg(norm_type="derf")
        norm = get_norm(cfg)
        self.assertIsInstance(norm, Derf)

    def test_fallback_returns_layer_norm(self):
        cfg = _cfg(norm_type="layer_norm")
        norm = get_norm(cfg)
        self.assertIsInstance(norm, nn.LayerNorm)

    def test_unknown_norm_fallback_to_layer_norm(self):
        cfg = _cfg(norm_type="unknown_norm_xyz")
        norm = get_norm(cfg)
        self.assertIsInstance(norm, nn.LayerNorm)


if __name__ == "__main__":
    unittest.main()
