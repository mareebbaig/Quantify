# Quantizer Specifications

## Framework
Use Brevitas (latest stable). All quantizers should be self-contained
Python modules with tests.

### Quantizer 1: Fixedpoint Per-Tensor Weight Quantizer (FixedPointPerTensorWeightQuantizer)
- Applies to: weights only, Per-Tensor
- Example: When signed=False, msb=2 and lsb=-1 the quantizer can represent the following values: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5
- Example: When signed=True, msb=2 and lsb=-1 the quantizer can represent the following values: -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5
- Rounding: round-to-nearest-even or by flooring
- Feature: The quantizer is given a bit-width and a rounding sheme (either round-to-nearest-even or by flooring). The msb and lsb position are then calculated based on the weights. The quantizer can just test a bunch of msb and lsb positions and choose the most fitting based on the variety. So, quantize the weights for each msb, lsb position and count how many unique values are present. Choose the one with the biggest number of unique values. If there are multiple msb, lsb settings that share the same number of unique values, choose the one where the rounding error is the smallest.
- Feature: Another feature is that the quantizer should automatically detect if the quantizer should use signed or unsiged numbers. This is also choosen based on the weights. If only positive weights are present, choose unsigned, otherwise signed.

# Brevitas Quantizer Usage
Quantizers are Injector subclasses. NEVER instantiate them.
WRONG:  quantizer = MyQuantizer()
CORRECT: layer = QuantLinear(64, 32, bias=True, weight_quant=MyQuantizer)

# Dependencies
When you need a new package, add it to requirements.txt before importing it.

# Conventions for Brevitas Fixed-Point Quantizers
 
## CRITICAL IMPORT RULES (Brevitas v0.12.x)
 
WRONG (old API, will crash):
    from brevitas.quant import QuantType
 
CORRECT:
    from brevitas.inject.enum import QuantType
    from brevitas.inject import BaseInjector as Injector
 
## Classes That DO NOT EXIST — Never use these
- ExtendedInjector
- QuantInjector
- BaseQuantizer
- TensorQuantizer
## Brevitas Quantizer Usage
Quantizers are Injector subclasses. NEVER instantiate them.
WRONG:  quantizer = MyQuantizer()
CORRECT: layer = QuantLinear(64, 32, bias=True, weight_quant=MyQuantizer)
 
## Custom Quantizers
For custom quantization logic, implement a torch.nn.Module with a forward()
that returns (quantized_tensor, scale, zero_point, bit_width) and set it
as the tensor_quant attribute of an Injector.
 

# AGENT_NOTES.md
AGENT_NOTES.md is your scratchpad. Every time you encounter an 
error, a wrong import, or learn something about the Brevitas API 
that wasn't obvious, write it down there. Check it before writing 
any new code.

# CRITICAL IMPORT RULES — READ BEFORE WRITING ANY CODE

WRONG (old API, will crash):
    from brevitas.quant import QuantType
    from brevitas.quant import ScalingImplType
    from brevitas.quant import BitWidthImplType

CORRECT (current API, v0.12.x):
    from brevitas.inject.enum import QuantType
    from brevitas.inject.enum import BitWidthImplType
    from brevitas.inject.enum import ScalingImplType
    from brevitas.inject.enum import RestrictValueType
    from brevitas.inject.enum import StatsOp
    from brevitas.inject import BaseInjector as Injector
    from brevitas.core.zero_point import ZeroZeroPoint
