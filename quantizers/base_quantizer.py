"""
Base Quantizer Infrastructure for Brevitas.

Provides shared boilerplate for per-tensor quantizers, including:
- Manager integration & inference gating
- Calibration state management
- ONNX export guards
- Brevitas 4-tuple return contract
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Any

from quantizers.manager import quantizer_manager


class BaseQuantizer(nn.Module, ABC):
    """
    Abstract base class for per-tensor quantizers.
    
    Handles manager registration, inference gating, calibration state,
    and ONNX export guards. Subclasses implement domain-specific calibration
    and quantization math.
    """

    def __init__(self, bit_width: int = 8, **kwargs):
        super().__init__()
        self.bit_width = bit_width
        self.inference_counter = 0
        self.inference_sequence_id = -1
        
        # Calibration state buffers
        self.register_buffer('search_done', torch.tensor(False, dtype=torch.bool))
        
        # Register with global manager
        quantizer_manager.register_quantizer(self)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Inference gating
        if self.inference_sequence_id == -1:
            self.inference_sequence_id = quantizer_manager.get_inference_sequence_id()
            
        perform_quantization = True
        if not quantizer_manager.quantization_is_enabled_globally:
            perform_quantization = False
        elif self.inference_counter < self.inference_sequence_id * quantizer_manager.quantization_start_gap:
            self.inference_counter += 1
            perform_quantization = False
            
        if not perform_quantization:
            return x, torch.tensor(1.0, dtype=x.dtype, device=x.device), \
                   torch.tensor(0.0, dtype=x.dtype, device=x.device), \
                   torch.tensor(float(self.bit_width), dtype=x.dtype, device=x.device)

        # 2. Calibration check
        is_exporting = torch.onnx.is_in_onnx_export()
        should_calibrate = not self.search_done.item() or quantizer_manager.force_recalibration
        
        if not is_exporting and should_calibrate:
            params = self._calibrate(x)
            self._save_calibration(params)
        else:
            params = self._load_calibration()
            
        # 3. Quantize & format output
        quantized = self._quantize(x, params)
        scale, zero_point, bit_width = self._get_metadata(params, x)
        return quantized, scale, zero_point, bit_width

    # Abstract methods for subclasses
    @abstractmethod
    def _calibrate(self, x: torch.Tensor) -> Any:
        """Run calibration/search logic and return a params dict."""
        raise NotImplementedError

    @abstractmethod
    def _save_calibration(self, params: Any) -> None:
        """Save calibration results to buffers."""
        raise NotImplementedError

    @abstractmethod
    def _load_calibration(self) -> Any:
        """Load calibration results from buffers."""
        raise NotImplementedError

    @abstractmethod
    def _quantize(self, x: torch.Tensor, params: Any) -> torch.Tensor:
        """Apply quantization using the provided parameters."""
        raise NotImplementedError

    @abstractmethod
    def _get_metadata(self, params: Any, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return scale, zero_point, and bit_width tensors matching x's dtype/device."""
        raise NotImplementedError
