# Custom Brevitas Quantizers — Reference

This document specifies how to build custom quantizers in Brevitas. It covers
the three-layer architecture, the dependency-injection system, the
`tensor_quant` contract, parameterization patterns, ONNX export, calibration,
and common bugs.

Audience: developers writing Brevitas quantizers, and LLM agents generating
quantizer code from this reference.

All claims in this document have been verified against Brevitas 0.12.1 with
PyTorch 2.11.

---

## Quick Reference (Code-Generation Checklist)

When writing a new custom quantizer, complete every item:

1. Define an `nn.Module` subclass (the `tensor_quant`) whose `forward(x)`
   returns a 4-tuple `(quantized, scale, zero_point, bit_width)`.
2. The 4-tuple elements should be tensors. `bit_width` is conventionally a
   `float`-dtype scalar tensor for consistency with built-in quantizers,
   though Brevitas tolerates `int` and other dtypes.
3. Define an `Injector` or `ExtendedInjector` subclass with these required
   attributes: `quant_type`, `proxy_class`, `bit_width`, `signed`, plus any
   custom knobs.
4. Attach the `tensor_quant` to the injector by either (a) assigning the
   class directly — Brevitas auto-injects matching `__init__` parameters —
   or (b) using a `@value`-decorated factory function for renaming, derived
   values, or other custom logic.
5. Register all calibration state as buffers (`register_buffer`) so it
   survives `state_dict` roundtrips.
6. Guard any `.item()` calls or Python control flow with
   `torch.onnx.is_in_onnx_export()` to keep export traceable.
7. If the quantizer needs custom ONNX semantics, route the forward through
   a `torch.autograd.Function` with a `symbolic` static method, and pass
   `dynamo=False` to `torch.onnx.export` to use the legacy exporter.
8. Ensure `signed` on the injector matches what the `tensor_quant` actually
   produces. Do not auto-detect signedness from data unless the injector
   attribute is also derived consistently.
9. Test: standalone forward, layer integration, kwarg overrides, state-dict
   roundtrip, gradient flow, and ONNX export.

---

## 1. Architecture

A Brevitas quantizer has four cooperating levels. Data flows top to bottom;
configuration flows top to bottom; computation happens at the bottom.

**Level 1 — The Layer.** A user-facing module like `QuantLinear`,
`QuantConv2d`, or `QuantReLU`. Accepts kwargs of the form
`weight_quant=...`, `weight_bit_width=...`, `input_quant=...`, etc. The
prefix (`weight_`, `input_`, `output_`, `bias_`) selects which injector
the kwarg targets, and the rest of the name is forwarded as an attribute
override on that injector.

**Level 2 — The Injector.** A class (not an instance) that declares
quantizer configuration as class attributes. Brevitas reads these
attributes to construct the runtime quantizer. Dependencies between
attributes are resolved using the `@value` decorator.

**Level 3 — The Proxy.** A module like `WeightQuantProxyFromInjector` or
`ActQuantProxyFromInjector`. Wraps the `tensor_quant` module and adds
caching, training/eval state tracking, and `QuantTensor` construction.
You almost never write a custom proxy.

**Level 4 — The `tensor_quant` Module.** An `nn.Module` whose `forward`
performs the actual quantization. Returns the 4-tuple
`(quantized, scale, zero_point, bit_width)`. This is where you write
quantization math.

The Injector is **declarative**, not runtime. Brevitas inspects its class
attributes and builds the actual `tensor_quant` instance from them. You
do not normally instantiate the Injector yourself.

---

## 2. The `tensor_quant` Contract

### Signature

```python
def forward(self, x: torch.Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return quantized, scale, zero_point, bit_width
```

### Element specification

| Element       | Type             | Shape                       | Notes                                                 |
|---------------|------------------|-----------------------------|-------------------------------------------------------|
| `quantized`   | `torch.Tensor`   | same as `x`                 | Dequantized form (values lie on the quantization grid)|
| `scale`       | `torch.Tensor`   | scalar or per-channel       | Step size                                             |
| `zero_point`  | `torch.Tensor`   | scalar or per-channel       | Asymmetric offset; `0` for symmetric                  |
| `bit_width`   | `torch.Tensor`   | scalar                      | Conventionally `float` dtype                          |

The integer code is implicitly `(quantized - zero_point) / scale`. Brevitas
uses this to construct an `IntQuantTensor`, which enables graph-level
integer reasoning downstream (e.g., bit-exact integer accumulation in
`QuantConv2d`).

### `bit_width` flexibility

Brevitas accepts `bit_width` as a Python `int`, a `torch.long` tensor, or
a float tensor. The float tensor form is the convention used by built-in
quantizers and is recommended for consistency:

```python
bw = torch.tensor(float(self.bit_width), dtype=x.dtype, device=x.device)
```

The 4-tuple structure itself is strictly required. Returning a 3-tuple
raises `TypeError: IntQuantTensor.__new__() missing 1 required positional argument`.

### Dtype/device handling

```python
scale = torch.tensor(step, dtype=x.dtype, device=x.device)
zero_point = torch.tensor(0.0, dtype=x.dtype, device=x.device)
bw = torch.tensor(float(self.bit_width), dtype=x.dtype, device=x.device)
```

All four returned tensors must live on the same device as `x` and use
compatible dtypes. Building them with explicit `device=x.device` avoids
cross-device errors when the model is moved with `.cuda()` or `.to(...)`.

---

## 3. The Injector Pattern

### Required attributes for an integer weight quantizer

```python
from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import QuantType
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector

class MyWeightQuant(ExtendedInjector):
    quant_type   = QuantType.INT                  # required
    proxy_class  = WeightQuantProxyFromInjector   # required, role-dependent
    bit_width    = 4                              # required for INT
    signed       = True                           # required for INT
    narrow_range = True                           # custom knob
    tensor_quant = MyTensorQuantModule            # the nn.Module class
```

### How attribute injection works

Brevitas's injectors are built on the [`dependencies`](https://github.com/proofit404/dependencies)
library. When the proxy needs to instantiate `tensor_quant`, the
dependency resolver:

1. Inspects the `__init__` signature of the assigned class (or the
   parameters of a `@value`-decorated factory).
2. For each parameter name, looks up a matching attribute on the injector
   (or any subclass / kwarg override).
3. Calls the constructor with those resolved values.

This is why direct class assignment works: if `MyTensorQuantModule.__init__`
takes `bit_width`, `signed`, `narrow_range`, the resolver finds those names
on the injector and forwards them automatically.

```python
class MyTensorQuantModule(nn.Module):
    def __init__(self, bit_width, signed, narrow_range):  # names match injector
        super().__init__()
        ...
```

### When to use `@value` instead of direct assignment

Use `@value` when the constructor parameters need different names than
the injector attributes, when one parameter must be derived from others,
or when multiple injector attrs are combined into a single constructor
arg.

```python
from brevitas.inject import value

class MyWeightQuant(ExtendedInjector):
    bit_width            = 8
    signed               = True
    fractional_fraction  = 0.5

    @value
    def fractional_bits(bit_width, fractional_fraction):
        return int(bit_width * fractional_fraction)

    @value
    def tensor_quant(bit_width, fractional_bits):
        # Module wants different parameter names; @value bridges
        return MyFixedPointModule(
            total_bits=bit_width,
            frac_bits=fractional_bits,
        )
```

In this example, `weight_fractional_fraction=0.75` at the layer level
correctly propagates through the derived `fractional_bits` value.

### The silent-default trap

When direct class assignment is used, a `__init__` parameter that has a
default value but does **not** appear on the injector silently falls back
to its default. This is the actual gotcha to watch for.

```python
class MyTQ(nn.Module):
    def __init__(self, bit_width=4, mystery=999):  # mystery not on injector
        ...

class MyQuant(ExtendedInjector):
    bit_width = 4
    # ... no `mystery` attribute
    tensor_quant = MyTQ   # mystery=999 used silently
```

A `__init__` parameter that is **required** (no default) and not on the
injector raises `DependencyError` immediately, which is loud and easy
to fix. The silent-default case is the one to test for.

Layer kwargs can inject parameters the injector does not declare —
`weight_mystery=42` will reach the module's `__init__` even when
`mystery` is not on the injector. This works but is surprising; prefer
to declare every knob on the injector explicitly.

### `Injector` vs `ExtendedInjector`

| Base class           | Source                       | When to use                         |
|----------------------|------------------------------|-------------------------------------|
| `Injector`           | `dependencies` library       | Most cases work fine                |
| `ExtendedInjector`   | `brevitas.inject`            | Recommended for new code            |

`ExtendedInjector` adds `@this` for cross-references between values,
permits enums and arbitrary Python objects as attributes more
forgivingly, and is what built-in Brevitas quantizers use. `Injector`
(often imported as `BaseInjector`) is the strict base from the upstream
`dependencies` library and works for most patterns shown here.

### Injector classes are immutable

The `dependencies` library forbids attribute assignment on injector
classes after creation:

```python
class MyQuant(ExtendedInjector):
    bit_width = 4

MyQuant.bit_width = 8   # raises DependencyError
```

To create configuration variants programmatically, set attributes inside
the class body or use `type()` (see Pattern C below).

---

## 4. Parameterization Patterns

Four standard patterns exist for configuring a quantizer at use time.

### Pattern A — Layer kwarg overrides

Override any injector attribute by prefixing with the role.

```python
layer = QuantLinear(
    in_features=64, out_features=32, bias=True,
    weight_quant=MyWeightQuant,
    weight_bit_width=8,
    weight_narrow_range=False,
)
```

**Use when:** tweaking individual layers without defining new classes.

**Prefixes:** `weight_`, `bias_`, `input_`, `output_`.

Enum-typed attributes work the same way:

```python
weight_rounding_mode=RoundingMode.FLOOR
```

### Pattern B — Subclass presets

Define preset configurations as named subclasses.

```python
class MyWeightQuant4bit(MyWeightQuant):
    bit_width = 4

class MyWeightQuant8bitFloor(MyWeightQuant):
    bit_width = 8
    rounding_mode = RoundingMode.FLOOR
```

**Use when:** the project has a fixed set of supported configurations.

### Pattern C — Factory function

Generate quantizer classes programmatically. Two working approaches:

**Class-body assignment (preferred for readability):**

```python
def make_weight_quant(bw, rmode=RoundingMode.ROUND_TO_NEAREST_EVEN, narrow=True):
    class _Generated(MyWeightQuant):
        bit_width     = bw       # set inside the class body
        rounding_mode = rmode
        narrow_range  = narrow
    return _Generated

WQ = make_weight_quant(bw=cfg["wbits"])
layer = QuantLinear(..., weight_quant=WQ)
```

**`type()` form (for programmatic dict construction):**

```python
def make_weight_quant(bit_width, **kwargs):
    return type('_Generated', (MyWeightQuant,), {
        'bit_width': bit_width,
        **kwargs,
    })
```

**Do not** assign attributes after the class is defined:

```python
class _Generated(MyWeightQuant):
    pass
_Generated.bit_width = bw   # raises DependencyError
```

**Use when:** building models from config files or sweeping hyperparameters.

### Pattern D — Derived attributes via `@value`

Compute one attribute from another inside the injector.

```python
class MyWeightQuant(ExtendedInjector):
    bit_width = 8

    @value
    def half_bit_width(bit_width):
        return bit_width // 2
```

**Use when:** one knob should automatically determine other settings,
or when constructor parameter names differ from injector attribute names.

### Attribute types

Injector attributes accept any Python object: ints, bools, strings, enums,
custom classes. They flow unchanged through both direct class assignment
and `@value` factories.

```python
from enum import Enum

class RoundingMode(Enum):
    ROUND_TO_NEAREST_EVEN = "round_to_nearest_even"
    FLOOR = "floor"

class MyWeightQuant(ExtendedInjector):
    rounding_mode = RoundingMode.ROUND_TO_NEAREST_EVEN
    tensor_quant = MyTensorQuant   # __init__ takes rounding_mode
```

Layer-level overrides (`weight_rounding_mode=RoundingMode.FLOOR`) work
identically for enum values.

---

## 5. The Proxy Layer

Set `proxy_class` on the injector to match the role of the quantizer.

| Role            | Proxy class                              |
|-----------------|------------------------------------------|
| Weights         | `WeightQuantProxyFromInjector`           |
| Biases          | `BiasQuantProxyFromInjector`             |
| Input acts      | `ActQuantProxyFromInjector`              |
| Output acts     | `ActQuantProxyFromInjector`              |

The proxy handles caching during eval, training-mode tracking, `QuantTensor`
construction, and state-dict integration. Custom proxies are rarely
necessary; solve problems inside `tensor_quant` first.

---

## 6. The `signed` Attribute Trap

The injector's `signed` attribute is used by the proxy to construct the
output `IntQuantTensor`. If `tensor_quant` decides signedness internally
at runtime (e.g., from data statistics), there is a mismatch:

- The proxy assumes `signed = True` from the injector.
- The module may have selected unsigned representation.

This is not a silent issue. Calling `qw.int()` on the resulting
`QuantTensor` raises `RuntimeError: QuantTensor not valid.` because the
integer-range invariants do not hold. ONNX export of the integer view
also fails for the same reason.

### Recommended fix: make `signed` an explicit knob

```python
class MyTensorQuantModule(nn.Module):
    def __init__(self, bit_width, signed: Optional[bool] = None, ...):
        super().__init__()
        self._signed_override = signed   # None means auto-detect
        ...

    def forward(self, x):
        signed = self._signed_override
        if signed is None:
            signed = bool((x < 0).any().item())
        ...
```

If signedness must remain data-dependent, document the limitation and
provide two preset injectors (`MySignedQuant`, `MyUnsignedQuant`) so
users can pick the consistent configuration explicitly.

---

## 7. Stateful Quantizers (Calibration & Search)

### Buffer registration for calibration state

Persistent state (search results, running statistics, cached scales) must
be stored as buffers, not Python attributes. Buffers serialize into
`state_dict`; plain attributes do not.

```python
class MyTensorQuantModule(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.register_buffer('search_done', torch.tensor(False))
        self.register_buffer('cached_lsb', torch.tensor(0, dtype=torch.long))
```

State-dict keys appear under the proxy path, e.g.
`weight_quant.tensor_quant.search_done`.

For learnable quantizer parameters (e.g., a learned scale), use
`nn.Parameter` instead.

### Lazy / one-shot calibration pattern

Calibration is run once on the first non-export forward, the result is
cached, and subsequent forwards short-circuit. This avoids the chicken-
and-egg problem of needing weights at construction time when Brevitas
builds quantizers before the weight tensor is finalized.

```python
def forward(self, x):
    if not torch.onnx.is_in_onnx_export() and not self.search_done.item():
        lsb = self._search(x)
        self.cached_lsb.fill_(lsb)
        self.search_done.fill_(True)
    lsb = self.cached_lsb.item()
    return self._quantize(x, lsb)
```

### Retry-on-degenerate-result

If the first calibration produces a degenerate result (single unique
value, all zeros, etc.), do not lock it in. Set `search_done` only when
the result is non-trivial.

```python
if num_unique > 1:
    self.search_done.fill_(True)
else:
    self.search_done.fill_(False)   # retry on next forward
```

### ONNX export guard

Any `.item()` call breaks ONNX tracing because it materializes a Python
scalar that the tracer cannot record. Guard data-dependent control flow
with `torch.onnx.is_in_onnx_export()`.

```python
if not torch.onnx.is_in_onnx_export() and not self.search_done.item():
    ...
```

### Cache invalidation

If weights are updated after the first forward (checkpoint load, fold
operation, post-training fine-tuning), the cached calibration is stale.
Two strategies:

1. **Manual reset:** expose a `reset_calibration()` method that calls
   `self.search_done.fill_(False)`. Call after weight updates.
2. **Reset on `load_state_dict`:** override `_load_from_state_dict` to
   clear the flag.

The lazy pattern's cache is sticky by design — once `search_done = True`,
the cached LSB persists even if weights are scaled by 100x. Choose a
strategy explicitly rather than relying on automatic recalibration.

---

## 8. Activation Quantizers vs Weight Quantizers

The patterns above target weight quantizers. Activation quantizers differ
in two ways:

1. `proxy_class = ActQuantProxyFromInjector`.
2. The forward signature is identical — `(x) -> 4-tuple` — but `x` is a
   per-batch activation, so per-call calibration is not appropriate.
   Activation quantizers use observers (running min/max, percentile, MSE)
   populated during a calibration pass.

Reusable building blocks live under `brevitas.core.scaling`, e.g.
`StatsFromParameterScaling`, `ParameterFromRuntimeStatsScaling`. Two
strategies for custom activation quantizers:

- **Reuse Brevitas observers, swap the integer math.** Reference an
  existing scaling module via `@value` and only override the integer
  quantization core.
- **Write a custom observer module** and reference it from the injector
  via `@value`.

Bias quantizers use `BiasQuantProxyFromInjector` and traditionally derive
their scale from `weight_scale * input_scale`, making them tightly
coupled to the surrounding layer. This is the one role where reading
other quantizers' state inside a `tensor_quant` is normal.

---

## 9. ONNX Export

### When custom symbolic export is needed

Brevitas's QONNX/QCDQ exporters handle built-in integer quantization
ops. Custom semantics (non-uniform grids, exotic rounding, fixed-point
LSB metadata, lookup tables) require a custom symbolic export, otherwise
the default exporter emits a generic Quantize/Dequantize pair that loses
the custom behavior.

### PyTorch 2.x compatibility

PyTorch 2.x defaults to the dynamo-based exporter, which does **not**
work with the legacy `torch.autograd.Function.symbolic` mechanism.
Pass `dynamo=False` to use the legacy TorchScript-based exporter:

```python
torch.onnx.export(
    model, dummy, output_path,
    opset_version=13,
    custom_opsets={'mydomain': 1},
    dynamo=False,                  # required in PyTorch 2.x
)
```

Without `dynamo=False`, export fails inside `torch.export.export` with
`TorchExportError`, because the new exporter cannot trace the
data-dependent `.item()` calls in calibration code or the legacy
symbolic registration on the `Function`.

### Pattern: `torch.autograd.Function` with `symbolic`

```python
from torch.autograd import Function
from torch.onnx import symbolic_helper

class MyQuantFn(Function):
    @staticmethod
    def symbolic(g, x, scale, zero_point, bit_width, *attrs):
        scale_v = symbolic_helper._maybe_get_const(scale, "t")
        quantized = g.op(
            "mydomain::MyQuant",
            x,
            scale_f=scale_v,            # see suffix coercion below
            bit_width_i=int(bit_width),
            ...
        ).setType(x.type())             # propagate output type info

        # Build auxiliary outputs for Brevitas's 4-tuple expectation
        bw = g.op("Constant", value_t=torch.tensor(float(bit_width)))
        return quantized, scale, zero_point, bw

    @staticmethod
    def forward(ctx, x, scale, zero_point, bit_width, *attrs):
        ctx.save_for_backward(x)
        return _do_real_quantization(x, scale, ...)

    @staticmethod
    def backward(ctx, grad_out, *_):
        # Straight-through estimator
        return grad_out, None, None, None, *([None] * len(attrs))
```

Then call `MyQuantFn.apply(...)` from the module's `forward`.

### ONNX attribute type suffixes

ONNX op attributes are typed and the suffix on the kwarg name selects
the type.

| Suffix | ONNX type     | Use for                                  |
|--------|---------------|------------------------------------------|
| `_i`   | int           | bit-width, signed (as 0/1), narrow_range |
| `_f`   | float         | scalar Python float values               |
| `_s`   | string        | enum values, mode names                  |
| `_t`   | tensor        | per-channel scales, lookup tables        |
| `_is`  | list of ints  | shape vectors                            |
| `_fs`  | list of floats| per-channel scale (alternative to `_t`)  |

When passing values to the symbolic graph, PyTorch coerces 0-d tensors
to Python scalars automatically when the suffix demands it, so
`scale_f=tensor_value` works even though `tensor_value` came from
`_maybe_get_const(..., "t")`. For multi-element values or per-channel
scales, use `_t`/`_fs` explicitly.

### Encoding enums as ONNX attributes

ONNX has no native enum type. Serialize via the enum's `.value`:

```python
rounding_mode_s=str(rounding_mode.value),   # "round_to_nearest_even"
```

The consumer-side op handler must parse the string back to the enum.

### `setType` for shape/dtype propagation

Always call `.setType(x.type())` on the symbolic op result. Without it,
ONNX shape inference treats the output as opaque and downstream
optimizations or validators may fail.

```python
g.op("mydomain::MyQuant", x, ...).setType(x.type())
```

### Producing the 4-tuple in the symbolic graph

Brevitas expects `tensor_quant` to return four values, but a custom
symbolic op typically emits one node. Build auxiliary outputs as
`Constant` nodes:

```python
bw = g.op("Constant", value_t=torch.tensor(float(bit_width)))
return quantized, scale, zero_point, bw
```

`scale` and `zero_point` are typically passed through as the original
`torch._C.Value` inputs because they were already constants in the
calling forward.

### `backward` argument count

The number of `None`s returned from `backward` must equal the number of
positional arguments to `forward` (excluding `ctx`), including non-tensor
arguments like ints, bools, and enums. Mismatch produces a runtime
error during the backward pass.

### Custom domains require runtime support

A custom op like `mydomain::FixedPointQuant` is not part of standard
ONNX. ONNX Runtime, TensorRT, and similar engines will reject it unless
a custom kernel is registered for the target runtime. Three deployment
strategies:

1. **Custom kernel** — implement a C++/CUDA op for the target runtime.
   Most flexible, most work.
2. **Graph lowering pass** — run a transform that rewrites the custom
   op into standard ONNX ops (`QuantizeLinear`/`DequantizeLinear`)
   before deployment. Loses custom semantics but works on stock runtimes.
3. **QONNX export** — use Brevitas's built-in QONNX exporter. Produces
   a well-defined IR that downstream toolchains (FINN, hls4ml)
   understand natively. Best for FPGA/ASIC targets.

If the goal is research or hardware co-design, choose option 1 or 3.
For commodity inference, option 2 is the path of least resistance and
custom symbolic export may be unnecessary.

---

## 10. The `quant_type` Enum

The `quant_type` attribute selects the family of operations Brevitas
wires up.

| QuantType  | Meaning                                                  |
|------------|----------------------------------------------------------|
| `INT`      | Integer or fixed-point quantization (most common)        |
| `FP`       | Low-precision float (FP8, etc.)                          |
| `BINARY`   | ±1                                                       |
| `TERNARY`  | {-1, 0, +1}                                              |

For most custom integer-flavored quantizers — including arbitrary
fixed-point and non-uniform grids — `QuantType.INT` is the correct
choice. The enum tells Brevitas the output should be an `IntQuantTensor`,
not what the internal math looks like.

---

## 11. Testing

A working custom quantizer must pass all six checks.

**Test 1 — Standalone forward.** Instantiate `tensor_quant` directly and
verify outputs lie on the quantization grid.

```python
q, s, zp, bw = my_module(weights)
codes = (q - zp) / s
assert torch.allclose(codes, torch.round(codes))
```

**Test 2 — Layer integration.** Wrap in `QuantLinear`, run a forward.

```python
layer = QuantLinear(64, 32, weight_quant=MyWeightQuant)
y = layer(torch.randn(1, 64))
```

**Test 3 — Kwarg overrides.** Verify that `weight_bit_width=N` actually
changes the quantization.

```python
l4 = QuantLinear(64, 32, weight_quant=MyWeightQuant, weight_bit_width=4)
l8 = QuantLinear(64, 32, weight_quant=MyWeightQuant, weight_bit_width=8)
_ = l4(torch.randn(2, 64))
_ = l8(torch.randn(2, 64))
n4 = torch.unique(l4.quant_weight().value).numel()
n8 = torch.unique(l8.quant_weight().value).numel()
assert n4 < n8   # 4-bit produces fewer unique levels than 8-bit
```

**Test 4 — State-dict roundtrip.** Calibration state must persist.

```python
sd = model.state_dict()
model2 = build_same_model()
model2.load_state_dict(sd)
```

Inspect that calibration buffers appear under
`weight_quant.tensor_quant.<buffer_name>`.

**Test 5 — Gradient flow.** Verify gradients reach weights in training
mode (`weight.grad is not None` after backward).

**Test 6 — ONNX export.** Pass `dynamo=False` for legacy export with
`Function.symbolic`. Even if not deploying to ONNX, exporting catches
`.item()` calls outside the `is_in_onnx_export()` guard, Python control
flow, and tracing issues.

```python
torch.onnx.export(
    layer, torch.randn(1, 64), "out.onnx",
    opset_version=13,
    custom_opsets={'mydomain': 1},
    dynamo=False,
)
```

**Bonus — `qw.int()` validity.** If your quantizer auto-detects
signedness, verify that `layer.quant_weight().int()` does not raise
`RuntimeError: QuantTensor not valid.`. If it does, the injector's
`signed` attribute is inconsistent with the module's runtime decision.

---

## 12. Common Pitfalls

| Pitfall                                            | Symptom                                  | Fix                                                          |
|----------------------------------------------------|------------------------------------------|--------------------------------------------------------------|
| `__init__` param not on injector, has default      | Default used silently                    | Declare the knob on the injector explicitly                  |
| Modifying injector class after definition          | `DependencyError`                        | Set attributes inside class body or via `type()`             |
| Calibration state in plain Python attrs            | Lost on save/load                        | `register_buffer`                                            |
| `.item()` in forward, unguarded                    | ONNX export fails or traces incorrectly  | Guard with `is_in_onnx_export()`                             |
| Hard-coded `signed = True` + runtime detection     | `qw.int()` raises `QuantTensor not valid`| Make `signed` an explicit knob                               |
| Wrong `proxy_class` for the role                   | `AttributeError` or wrong tensor type    | Match proxy to role (weight/act/bias)                        |
| Returning fewer than 4 elements                    | `TypeError: missing 'training'`          | Always return the full 4-tuple                               |
| `scale` not on same device as input                | Cross-device errors                      | `torch.tensor(..., device=x.device)`                         |
| Using `dynamo=True` (PyTorch 2.x default) with custom symbolic | `TorchExportError`                | Pass `dynamo=False` to `torch.onnx.export`                   |
| Missing `setType()` on symbolic op output          | Downstream shape inference breaks        | `.setType(x.type())` on every custom op                      |
| Cache locked in on degenerate first forward        | Quantizer stuck at trivial calibration   | Only set `search_done = True` when result is non-trivial     |
| Stale calibration after weight updates             | Wrong scale after checkpoint load        | Reset on `load_state_dict` or expose `reset_calibration()`   |
| `backward` returns wrong number of `None`s         | Runtime error during backward            | Match count to `forward` positional args (including non-tensors)|
| Non-default `__init__` param not on injector       | `DependencyError` at construction        | Add the attribute or give the param a default                |

---

## 13. Minimal Template

A working skeleton with all required pieces. Fill in the marked sections.
Uses direct class assignment (Brevitas auto-injects `__init__` parameters
matching injector attribute names).

```python
import torch
import torch.nn as nn
from typing import Tuple

from brevitas.inject import ExtendedInjector
from brevitas.inject.enum import QuantType
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector


class MyTensorQuant(nn.Module):
    def __init__(self, bit_width: int, signed: bool, narrow_range: bool):
        # Parameter names MUST match injector attribute names for
        # auto-injection to forward them. Use a @value factory if names
        # need to differ.
        super().__init__()
        self.bit_width = bit_width
        self.signed = signed
        self.narrow_range = narrow_range
        # Register buffers for any calibration state here.

    def forward(self, x: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # 1. Compute scale (and optionally zero_point) from x.
        scale = ...  # FILL IN
        zero_point = torch.tensor(0.0, dtype=x.dtype, device=x.device)

        # 2. Quantize.
        codes = torch.round(x / scale)

        # 3. Clamp to representable range.
        if self.signed:
            lo = -(2 ** (self.bit_width - 1)) + (1 if self.narrow_range else 0)
            hi = 2 ** (self.bit_width - 1) - 1
        else:
            lo, hi = 0, 2 ** self.bit_width - 1
        codes = codes.clamp(lo, hi)

        # 4. Dequantize.
        quantized = codes * scale

        # 5. Build bit_width tensor with correct dtype/device.
        bw = torch.tensor(float(self.bit_width), dtype=x.dtype, device=x.device)

        return quantized, scale, zero_point, bw


class MyWeightQuant(ExtendedInjector):
    quant_type   = QuantType.INT
    proxy_class  = WeightQuantProxyFromInjector
    bit_width    = 8
    signed       = True
    narrow_range = True
    tensor_quant = MyTensorQuant   # auto-injects matching __init__ params
```

### Template variant: `@value` factory

Use this when constructor parameter names must differ from injector
attribute names, or when one parameter is computed from others.

```python
from brevitas.inject import value

class MyWeightQuant(ExtendedInjector):
    quant_type   = QuantType.INT
    proxy_class  = WeightQuantProxyFromInjector
    bit_width    = 8
    signed       = True
    narrow_range = True

    @value
    def tensor_quant(bit_width, signed, narrow_range):
        return MyTensorQuant(
            bit_width=bit_width,
            signed=signed,
            narrow_range=narrow_range,
        )
```

Both templates produce identical runtime behavior for the simple case.
Choose direct assignment for brevity, `@value` when explicit control is
needed.

---

## 14. Reference Source Code

These Brevitas modules are good reference reading:

| Path                              | Contents                                                |
|-----------------------------------|---------------------------------------------------------|
| `brevitas.quant.scaled_int`       | Standard integer quantizer presets                      |
| `brevitas.quant.fixed_point`      | Brevitas's own fixed-point implementation               |
| `brevitas.quant.binary`           | Minimal example of a non-INT `quant_type`               |
| `brevitas.core.quant.int`         | The `IntQuant` module (a polished `tensor_quant`)       |
| `brevitas.quant.solver`           | How `@value` chains resolve under the hood              |
| `brevitas.core.scaling`           | Reusable scaling/observer modules for activation quants |