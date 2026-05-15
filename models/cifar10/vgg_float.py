import torch.nn as nn

class VGG(nn.Module):
    """
    Small VGG-style CNN for CIFAR-10 (Floating Point).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.inp = nn.Identity()

        self.features = nn.Sequential(
            *self._conv_block(3,   64),
            *self._conv_block(64,  64),
            nn.MaxPool2d(2),                        # 32 -> 16
            *self._conv_block(64,  128),
            *self._conv_block(128, 128),
            nn.MaxPool2d(2),                        # 16 -> 8
            *self._conv_block(128, 256),
            *self._conv_block(256, 256),
            nn.AdaptiveAvgPool2d(1),                #  8 -> 1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes, bias=True),
        )

    @staticmethod
    def _conv_block(in_ch, out_ch):
        return [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        ]

    def forward(self, x):
        x = self.inp(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
