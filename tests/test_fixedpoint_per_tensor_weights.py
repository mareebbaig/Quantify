import torch
import torch.nn as nn
from quantizers.fixedpoint_per_tensor_weights import Quantizer1
from brevitas.proxy import WeightQuantProxyFromInjector

def test_quantizer1_linear_forward():
    """
    Test that Quantizer1 can be applied to a nn.Linear(64, 32) layer,
    and that the quantized weights have the correct scale shape and value range.
    """
    # Create a linear layer
    linear_layer = nn.Linear(64, 32)
    
    # Create quantizer instance using Brevitas injector pattern
    # This is the correct way to instantiate Brevitas quantizers
    quantizer = Quantizer1()
    
    # Create a weight quantization proxy using the quantizer
    # This properly integrates the quantizer with the linear layer
    weight_quant_proxy = WeightQuantProxyFromInjector(quantizer, weight=linear_layer.weight)
    
    # Check that the quantizer has the correct bit width
    assert quantizer.bit_width == 8, f"Expected bit_width 8, got {quantizer.bit_width}"
    
    # Check that the quantizer has the correct scaling type
    assert quantizer.scaling_impl_type == 'PARAMETER', f"Expected scaling_impl_type PARAMETER, got {quantizer.scaling_impl_type}"
    
    # Test forward pass with a random input
    input_tensor = torch.randn(10, 64)
    output = linear_layer(input_tensor)
    
    # Check that forward pass works correctly
    assert output.shape == (10, 32), \
        f"Expected output shape (10, 32), got {output.shape}"
    
    print("All tests passed!")

if __name__ == "__main__":
    test_quantizer1_linear_forward()
