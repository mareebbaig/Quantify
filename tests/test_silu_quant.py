"""
Tests for the Quantized SiLU Activation Function.

Covers:
    - Core SiLU + fixed-point quantization math
    - Calibration logic (LSB search, buffer updates)
    - Brevitas integration via QuantConv2d / QuantLinear
    - ONNX export with custom `mydomain::QuantSiLU` node
    - Gradient flow (STE)
    - Edge cases (all zeros, negative inputs, extreme values)
    - State-dict roundtrips
    - Rounding modes
"""

import math
import tempfile
import unittest

import pytest
import torch
import torch.nn as nn

from quantizers import (
    SiLUTensorQuant,
    QuantSiLUActivationQuant,
    RoundingMode,
    quantize_fixed_point,
)


# =========================================================================
# 1. Core SiLU + Quantization Math
# =========================================================================


class TestSiLUQuantization:
    """Tests for the SiLU activation followed by fixed-point quantization."""

    def test_silu_output_range(self):
        """SiLU output is bounded below by ~-0.28 and above by max_input."""
        x = torch.randn(100)
        x_silu = torch.nn.functional.silu(x)
        assert (x_silu >= -0.28).all()
        assert (x_silu <= x.max()).all()

    def test_silu_quantization_preserves_grid(self):
        """Quantized SiLU outputs should lie exactly on the fixed-point grid."""
        quantizer = SiLUTensorQuant(bit_width=4)
        x = torch.randn(64)
        # Run forward to calibrate
        q, s, z, bw = quantizer(x)
        step = s.item()
        codes = q / step
        residual = torch.abs(codes - torch.round(codes))
        assert residual.max().item() < 1e-5, f"Some values are not on the grid: max residual {residual.max().item()}"

    def test_silu_quantization_unsigned_default(self):
        """SiLU outputs are non-negative, so unsigned quantization should be used."""
        quantizer = SiLUTensorQuant(bit_width=4, signed=False)
        x = torch.randn(64)
        q, _, _, _ = quantizer(x)
        assert (q >= 0).all(), "SiLU outputs should be non-negative"


# =========================================================================
# 2. Calibration Logic
# =========================================================================


class TestSiLUCalibration:
    """Tests for the calibration and search logic."""

    def test_calibrate_finds_valid_lsb(self):
        """Calibration should find a valid LSB and save it to buffer."""
        quantizer = SiLUTensorQuant(bit_width=4)
        x = torch.randn(128)
        params = quantizer._calibrate(x)
        assert 'lsb' in params
        assert isinstance(params['lsb'], int)

    def test_save_and_load_calibration(self):
        """Saving and loading calibration should preserve the LSB."""
        quantizer = SiLUTensorQuant(bit_width=4)
        x = torch.randn(128)
        params = quantizer._calibrate(x)
        quantizer._save_calibration(params)
        loaded = quantizer._load_calibration()
        assert loaded['lsb'] == params['lsb']
        assert quantizer.search_done.item() is True

    def test_calibration_updates_buffers(self):
        """Buffers should be updated after calibration."""
        quantizer = SiLUTensorQuant(bit_width=4)
        x = torch.randn(128)
        quantizer(x)  # Triggers calibration via BaseQuantizer.forward
        assert quantizer.search_done.item() is True
        # LSB could be 0, but search_done must be True
        assert quantizer.search_result_lsb.item() is not None


# =========================================================================
# 3. Module Behavior
# =========================================================================


class TestSiLUTensorQuantModule:
    """Tests for the SiLUTensorQuant nn.Module."""

    def test_output_shape(self):
        quantizer = SiLUTensorQuant(bit_width=4)
        x = torch.randn(32, 64)
        q, s, z, bw = quantizer(x)
        assert q.shape == x.shape

    def test_returns_four_tuple(self):
        quantizer = SiLUTensorQuant(bit_width=4)
        x = torch.randn(32, 64)
        result = quantizer(x)
        assert len(result) == 4
        q, s, z, bw = result
        assert isinstance(q, torch.Tensor)
        assert isinstance(s, torch.Tensor)
        assert isinstance(z, torch.Tensor)
        assert isinstance(bw, torch.Tensor)

    def test_bit_width_returned(self):
        for bw_val in [2, 4, 8, 16]:
            quantizer = SiLUTensorQuant(bit_width=bw_val)
            _, _, _, bw = quantizer(torch.randn(16))
            assert bw.item() == float(bw_val)

    def test_scale_is_power_of_two(self):
        quantizer = SiLUTensorQuant(bit_width=4)
        x = torch.randn(128)
        _, scale, _, _ = quantizer(x)
        log2_scale = math.log2(scale.item())
        assert log2_scale == pytest.approx(round(log2_scale)), f"Scale {scale.item()} is not a power of 2"

    def test_zero_point_is_zero(self):
        quantizer = SiLUTensorQuant(bit_width=4)
        _, _, zp, _ = quantizer(torch.randn(64))
        assert zp.item() == 0.0

    def test_floor_rounding_mode(self):
        quantizer = SiLUTensorQuant(
            bit_width=4, rounding_mode=RoundingMode.FLOOR
        )
        x = torch.randn(128)
        q, _, _, _ = quantizer(x)
        assert q.shape == x.shape
        assert torch.isfinite(q).all()


# =========================================================================
# 4. Brevitas Integration
# =========================================================================


class TestBrevitasIntegration:
    """Tests for integration with Brevitas layers."""

    def test_quantconv2d_with_silu_act(self):
        from brevitas.nn import QuantConv2d
        layer = QuantConv2d(3, 16, 3, padding=1, act_quant=QuantSiLUActivationQuant)
        x = torch.randn(1, 3, 32, 32)
        out = layer(x)
        assert out.shape == (1, 16, 32, 32)
        assert torch.isfinite(out).all()

    def test_quantlinear_with_silu_act(self):
        from brevitas.nn import QuantLinear
        layer = QuantLinear(64, 32, bias=True, act_quant=QuantSiLUActivationQuant)
        x = torch.randn(1, 64)
        out = layer(x)
        assert out.shape == (1, 32)
        assert torch.isfinite(out).all()

    def test_custom_bit_width_via_subclass(self):
        from brevitas.nn import QuantConv2d

        class SiLU4bit(QuantSiLUActivationQuant):
            bit_width = 4

        layer = QuantConv2d(3, 16, 3, padding=1, act_quant=SiLU4bit)
        x = torch.randn(1, 3, 32, 32)
        out = layer(x)
        assert out.shape == (1, 16, 32, 32)


# =========================================================================
# 5. Edge Cases
# =========================================================================


class TestEdgeCases:
    def test_all_zeros(self):
        quantizer = SiLUTensorQuant(bit_width=4)
        x = torch.zeros(64)
        q, s, z, bw = quantizer(x)
        assert torch.isfinite(q).all()
        assert (q == 0.0).all()

    def test_negative_inputs(self):
        """SiLU should clamp negatives to ~0, quantization should handle it."""
        quantizer = SiLUTensorQuant(bit_width=4)
        x = torch.randn(64) * -5.0
        q, _, _, _ = quantizer(x)
        assert (q >= 0).all()

    def test_very_large_inputs(self):
        quantizer = SiLUTensorQuant(bit_width=8)
        x = torch.randn(64) * 1e6
        q, s, z, bw = quantizer(x)
        assert torch.isfinite(q).all()

    def test_single_element(self):
        quantizer = SiLUTensorQuant(bit_width=4)
        x = torch.tensor([1.234])
        q, s, z, bw = quantizer(x)
        assert q.shape == (1,)
        assert torch.isfinite(q).all()

    def test_two_bit_extreme(self):
        quantizer = SiLUTensorQuant(bit_width=2)
        x = torch.randn(64)
        q, s, z, bw = quantizer(x)
        assert torch.unique(q).numel() <= 4
        assert torch.isfinite(q).all()


# =========================================================================
# 6. Gradient Flow
# =========================================================================


class TestGradientFlow:
    def test_ste_gradient_flow(self):
        quantizer = SiLUTensorQuant(bit_width=4)
        x = torch.randn(32, 64, requires_grad=True)
        q, _, _, _ = quantizer(x)
        loss = q.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# =========================================================================
# 7. ONNX Export
# =========================================================================


class TestONNXExport:
    def test_onnx_export_custom_node(self):
        quantizer = SiLUTensorQuant(
            bit_width=8, rounding_mode=RoundingMode.ROUND_TO_NEAREST_EVEN
        )
        x = torch.randn(10, 10)
        _ = quantizer(x)  # Calibrate

        class DummyModel(nn.Module):
            def __init__(self, quantizer):
                super().__init__()
                self.quantizer = quantizer
            def forward(self, x):
                q, s, z, b = self.quantizer(x)
                return q

        model = DummyModel(quantizer)
        model.eval()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name

        dummy_input = torch.randn(2, 10)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            opset_version=14,
            dynamo=False,
            input_names=["input"],
            output_names=["output"]
        )

        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            custom_nodes = [n for n in onnx_model.graph.node if n.op_type == "QuantSiLU" and n.domain == "mydomain"]
            assert len(custom_nodes) == 1, f"Expected 1 custom node, found {len(custom_nodes)}"

            node = custom_nodes[0]
            attr_names = [a.name for a in node.attribute]
            assert "lsb" in attr_names
            assert "bit_width" in attr_names
            assert "signed" in attr_names
            assert "rounding_mode" in attr_names
        finally:
            import os
            if os.path.exists(onnx_path):
                os.remove(onnx_path)

    def test_onnx_export_does_not_recalibrate(self):
        quantizer = SiLUTensorQuant(bit_width=8)
        x = torch.randn(32, 64)
        _ = quantizer(x)
        initial_lsb = quantizer.search_result_lsb.item()

        class DummyModel(nn.Module):
            def __init__(self, quantizer):
                super().__init__()
                self.quantizer = quantizer
            def forward(self, x):
                return self.quantizer(x)

        model = DummyModel(quantizer)
        model.eval()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name

        dummy_input = torch.randn(2, 32, 64)
        torch.onnx.export(
            model, dummy_input, onnx_path,
            opset_version=14, dynamo=False
        )

        assert quantizer.search_result_lsb.item() == initial_lsb

        import os
        if os.path.exists(onnx_path):
            os.remove(onnx_path)

    def test_onnx_model_validates(self):
        quantizer = SiLUTensorQuant(bit_width=8)
        x = torch.randn(10, 10)
        _ = quantizer(x)

        class DummyModel(nn.Module):
            def __init__(self, quantizer):
                super().__init__()
                self.quantizer = quantizer
            def forward(self, x):
                q, s, z, b = self.quantizer(x)
                return q

        model = DummyModel(quantizer)
        model.eval()

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name

        dummy_input = torch.randn(2, 10)
        torch.onnx.export(
            model, dummy_input, onnx_path,
            opset_version=14, dynamo=False
        )

        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
        finally:
            import os
            if os.path.exists(onnx_path):
                os.remove(onnx_path)


# =========================================================================
# 8. State-Dict Roundtrip
# =========================================================================


class TestStateDictRoundtrip:
    def test_silu_quantizer_roundtrip(self):
        quantizer = SiLUTensorQuant(bit_width=8)
        x = torch.randn(32, 64)
        _ = quantizer(x)

        sd = quantizer.state_dict()
        new_quantizer = SiLUTensorQuant(bit_width=8)
        new_quantizer.load_state_dict(sd)

        q1, s1, _, _ = quantizer(x)
        q2, s2, _, _ = new_quantizer(x)

        assert torch.allclose(q1, q2)
        assert torch.allclose(s1, s2)


if __name__ == "__main__":
    unittest.main()
