"""
Base Injectors for Brevitas Quantizers.

Provides shared boilerplate for Brevitas injectors, including:
- Common quant_type and proxy_class definitions
- Default bit_width and signed attributes
- Centralized proxy resolution fallbacks
"""

from brevitas.inject import BaseInjector as Injector
from brevitas.inject.enum import QuantType
from brevitas.proxy.parameter_quant import WeightQuantProxyFromInjector

try:
    from brevitas.proxy.runtime_quant import ActQuantProxyFromInjector as ActivationQuantProxyFromInjector
except ImportError:
    try:
        from brevitas.proxy.activation_quant import ActivationQuantProxyFromInjector
    except ImportError:
        try:
            from brevitas.proxy.activation import ActivationQuantProxyFromInjector
        except ImportError:
            raise ImportError(
                "Could not find ActivationQuantProxyFromInjector. "
                "Please ensure you have a compatible version of Brevitas installed."
            )


class BaseWeightQuant(Injector):
    """Base injector for weight quantizers."""
    quant_type = QuantType.INT
    proxy_class = WeightQuantProxyFromInjector
    bit_width = 8
    signed = True
    quantizer_role = "weight"


class BaseActivationQuant(Injector):
    """Base injector for activation quantizers."""
    quant_type = QuantType.INT
    proxy_class = ActivationQuantProxyFromInjector
    bit_width = 8
    signed = False
    quantizer_role = "activation"
