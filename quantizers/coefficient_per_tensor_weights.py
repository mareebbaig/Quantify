"""
Coefficient Per-Tensor Weight Quantizer for Brevitas.

This quantizer rounds weights to the nearest value from a set of predefined 
coefficients provided in a text file. It searches for the optimal coefficient 
set and a power-of-two scaling factor (2^n) that minimizes the Sum of 
Absolute Differences (SAD) between the original and quantized weights.

The text file format:
    Each line contains one set of coefficients.
    Coefficients within a set are separated by spaces.

Example:
    -1.0 0.0 1.0
    -0.5 -0.25 0.0 0.25 0.5
"""

import torch
import torch.nn as nn
from typing import Tuple, Any

from quantizers.base_injector import BaseWeightQuant
from quantizers.base_quantizer import BaseQuantizer
from torch.autograd import Function
from torch.onnx import symbolic_helper


class CoefficientQuantFn(Function):
    """Symbolic shim: emits a single `mydomain::CoefficientQuant` ONNX node."""

    @staticmethod
    def symbolic(g, x, coefficients, n, bit_width, signed):
        coeffs_val = symbolic_helper._maybe_get_const(coefficients, "t")
        
        quantized = g.op(
            "mydomain::CoefficientQuant",
            x,
            coefficients_t=coeffs_val,
            n_i=int(n),
            bit_width_i=int(bit_width),
            signed_i=int(signed),
        ).setType(x.type())
        
        # Brevitas expects a 4-tuple output; create auxiliary constants
        scale = g.op("Constant", value_t=torch.tensor(2.0 ** n))
        zero_point = g.op("Constant", value_t=torch.tensor(0.0))
        bw = g.op("Constant", value_t=torch.tensor(float(bit_width)))
        return quantized, scale, zero_point, bw

    @staticmethod
    def forward(ctx, x, coefficients, n, bit_width, signed):
        ctx.save_for_backward(x)
        s = 2.0 ** n
        scaled_coeffs = coefficients * s
        diffs = torch.abs(x.unsqueeze(-1) - scaled_coeffs)
        min_indices = torch.argmin(diffs, dim=-1)
        quantized = scaled_coeffs[min_indices]
        bw = torch.tensor(float(bit_width), dtype=x.dtype, device=x.device)
        return quantized, torch.tensor(s, dtype=x.dtype, device=x.device), torch.tensor(0.0, dtype=x.dtype, device=x.device), bw

    @staticmethod
    def backward(ctx, grad_quantized, grad_scale, grad_zero_point, grad_bw):
        # Straight-Through Estimator: pass gradient through for the first input
        return grad_quantized, None, None, None, None


class CoefficientPerTensorWeightQuantizer(BaseQuantizer):
    """
    A self-contained coefficient-based per-tensor weight quantizer.
    Inherits infrastructure from BaseQuantizer (gating, calibration state, ONNX guards).
    """

    def __init__(self, filepath: str, bit_width: int = 8):
        super().__init__(bit_width=bit_width)
        self.filepath = filepath
        
        # Read coefficient sets from the text file during initialization
        self.coefficient_sets = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    coeffs = torch.tensor([float(x) for x in line.split()], dtype=torch.float32)
                    self.coefficient_sets.append(coeffs)

        if not self.coefficient_sets:
            raise ValueError(f"No valid coefficient sets found in file: {filepath}")

        # Register search results as buffers (handled by base class for state-dict)
        self.register_buffer('best_set_idx', torch.tensor(0, dtype=torch.long))
        self.register_buffer('best_n', torch.tensor(0, dtype=torch.long))

    def _calibrate(self, x: torch.Tensor) -> Any:
        """Run calibration/search logic and return a params dict."""
        device = x.device
        best_sad = float("inf")
        best_set_idx = 0
        best_n = 0

        for idx, coeffs in enumerate(self.coefficient_sets):
            coeffs_dev = coeffs.to(device)
            for n in range(-12, 13):
                s = 2.0 ** n
                scaled_coeffs = coeffs_dev * s
                diffs = torch.abs(x.unsqueeze(-1) - scaled_coeffs)
                min_indices = torch.argmin(diffs, dim=-1)
                quantized_temp = scaled_coeffs[min_indices]
                sad = torch.sum(torch.abs(x - quantized_temp)).item()
                
                if sad < best_sad:
                    best_sad = sad
                    best_set_idx = idx
                    best_n = n

        return {'set_idx': best_set_idx, 'n': best_n}

    def _save_calibration(self, params: Any) -> None:
        """Save calibration results to buffers."""
        self.best_set_idx.fill_(params['set_idx'])
        self.best_n.fill_(params['n'])
        self.search_done.fill_(True)

    def _load_calibration(self) -> Any:
        """Load calibration results from buffers."""
        return {
            'set_idx': self.best_set_idx.item(),
            'n': self.best_n.item()
        }

    def _quantize(self, x: torch.Tensor, params: Any) -> torch.Tensor:
        """Apply quantization using the provided parameters."""
        if torch.onnx.is_in_onnx_export():
            chosen_coeffs = self.coefficient_sets[params['set_idx']].to(x.device)
            quantized, _, _, _ = CoefficientQuantFn.apply(
                x,
                chosen_coeffs,
                params['n'],
                len(self.coefficient_sets[params['set_idx']]),
                1  # signed
            )
            return quantized
            
        chosen_coeffs = self.coefficient_sets[params['set_idx']].to(x.device)
        s = 2.0 ** params['n']
        scaled_coeffs = chosen_coeffs * s
        
        diffs = torch.abs(x.unsqueeze(-1) - scaled_coeffs)
        min_indices = torch.argmin(diffs, dim=-1)
        quantized = scaled_coeffs[min_indices]
        return quantized

    def _get_metadata(self, params: Any, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return scale, zero_point, and bit_width tensors matching x's dtype/device."""
        scale = torch.tensor(2.0 ** params['n'], dtype=x.dtype, device=x.device)
        zero_point = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        # Bit width corresponds to the number of coefficients in the chosen set
        bit_width = torch.tensor(float(len(self.coefficient_sets[params['set_idx']])), dtype=x.dtype, device=x.device)
        return scale, zero_point, bit_width


class CoefficientPerTensorWeightQuant(BaseWeightQuant):
    """
    Brevitas-compatible Injector for the coefficient-based per-tensor weight quantizer.
    """
    tensor_quant = CoefficientPerTensorWeightQuantizer
    filepath = "coefficients.txt"
    # signed inherited from BaseWeightQuant (True)
