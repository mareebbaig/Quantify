# Brevitas Pitfalls

## 1. Hallucinating Non-Existent `Quant*` Layers
**The Problem:**
When building models with Brevitas, it's easy to assume that every standard PyTorch layer has a corresponding quantized wrapper (e.g., `qnn.QuantGlobalAvgPool2d`). However, Brevitas only provides quantization wrappers for a specific subset of layers (primarily convolutions, linear layers, batch normalization, and basic activations). Pooling layers, normalization layers beyond BatchNorm, and other custom operations do not have built-in `Quant*` equivalents.

**How to Prevent It:**
- Always verify the existence of a layer in the official [Brevitas API documentation](https://brevitas.readthedocs.io/) or the source code before using it.
- For unsupported layers, use the standard PyTorch implementation (e.g., `nn.AdaptiveAvgPool2d(1)`).
- If you need to quantize the output of an unsupported layer, wrap it with `qnn.QuantIdentity` or apply quantization explicitly in the forward pass.
