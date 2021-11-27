# vgg11_bn_pruned.py COPYRIGHT Fujitsu Limited 2021

import torch.nn as nn

__all__ = [
    "VGG",
    "vgg11_bn"
]


class VGG11_BN(nn.Module):
    def __init__(self,
                 batch_norm=True,
                 init_weights=True,
                 num_classes=10,
                 out_ch_conv1=30,
                 out_ch_conv2=53,
                 out_ch_conv3=105,
                 out_ch_conv4=94,
                 out_ch_conv5=151,
                 out_ch_conv6=109,
                 out_ch_conv7=62,
                 out_ch_conv8=22,
                 out_ch_fc1=445,
                 out_ch_fc2=282):
        super(VGG11_BN, self).__init__()
        self.cfg = [out_ch_conv1, "M", out_ch_conv2, "M",
                    out_ch_conv3, out_ch_conv4, "M", out_ch_conv5,
                    out_ch_conv6, "M", out_ch_conv7, out_ch_conv8, "M"]
        self.features = make_layers(self.cfg, batch_norm=batch_norm)
        # CIFAR 10 (7, 7) to (1, 1)
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(out_ch_conv8 * 1 * 1, out_ch_fc1),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(out_ch_fc1, out_ch_fc2),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(out_ch_fc2, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
