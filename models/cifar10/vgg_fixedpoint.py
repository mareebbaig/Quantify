import torch.nn as nn
import brevitas.nn as qnn
from quantizers.fixedpoint_per_tensor import FixedPointPerTensorWeightQuant


class QuantVGG(nn.Module):
    """
    Small VGG-style quantized CNN for CIFAR-10 using Fixed-Point weights.
    """

    def __init__(self,
                 num_classes: int = 10,
                 weight_bit_width: int = 8,
                 act_bit_width: int = 8):
        super().__init__()

        # Create a local subclass of the injector to set the bit_width dynamically
        class FixedPointWeightQuant(FixedPointPerTensorWeightQuant):
            bit_width = weight_bit_width

        self.quant_inp = nn.Identity()

        self.features = nn.Sequential(
            *self._conv_block(3,   64,  weight_bit_width, act_bit_width, FixedPointWeightQuant),
            *self._conv_block(64,  64,  weight_bit_width, act_bit_width, FixedPointWeightQuant),
            nn.MaxPool2d(2),                        # 32 -> 16
            *self._conv_block(64,  128, weight_bit_width, act_bit_width, FixedPointWeightQuant),
            *self._conv_block(128, 128, weight_bit_width, act_bit_width, FixedPointWeightQuant),
            nn.MaxPool2d(2),                        # 16 -> 8
            *self._conv_block(128, 256, weight_bit_width, act_bit_width, FixedPointWeightQuant),
            *self._conv_block(256, 256, weight_bit_width, act_bit_width, FixedPointWeightQuant),
            nn.AdaptiveAvgPool2d(1),                #  8 -> 1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            qnn.QuantLinear(256, 256, bias=False,
                            weight_bit_width=weight_bit_width,
                            weight_quant=FixedPointWeightQuant),
            nn.BatchNorm1d(256),
            qnn.QuantReLU(bit_width=act_bit_width) if act_bit_width is not None else nn.ReLU(),
            qnn.QuantLinear(256, num_classes, bias=True,
                            weight_bit_width=weight_bit_width,
                            weight_quant=FixedPointWeightQuant),
        )

    @staticmethod
    def _conv_block(in_ch, out_ch, w_bits, a_bits, weight_quant):
        return [
            qnn.QuantConv2d(in_ch, out_ch, kernel_size=3, padding=1,
                            bias=False, weight_bit_width=w_bits,
                            weight_quant=weight_quant),
            nn.BatchNorm2d(out_ch),
            qnn.QuantReLU(bit_width=a_bits) if a_bits is not None else nn.ReLU(),
        ]

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
