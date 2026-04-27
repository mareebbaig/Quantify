# Agent Notes
## Lessons learned from errors — check before writing code

- `QuantType` lives in `brevitas.inject.enum`, NOT `brevitas.quant`
- Injectors must never be instantiated, pass them as class references
- Use `QuantLinear(weight_quant=MyQuantizer)`, never `MyQuantizer()`
