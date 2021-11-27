# resnet18.py COPYRIGHT Fujitsu Limited 2021

import torch.nn as nn

__all__ = [
    "ResNet",
    "resnet18"
]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        n_in_channels=None,
        n_out_channels1=None,
        n_out_channels2=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(n_in_channels, n_out_channels1, stride)
        self.bn1 = norm_layer(n_out_channels1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(n_out_channels1, n_out_channels2)
        self.bn2 = norm_layer(n_out_channels2)
        self.downsample = downsample #if dawnsample else downsample(n_in_channels, n_out_channels3)
        self.stride = stride


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(
        self,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        out_ch_conv1=64,

        out_ch_l1_0_1=64,
        out_ch_l1_0_2=64,
        out_ch_l1_1_1=64,
        out_ch_l1_1_2=64,

        out_ch_l2_0_1=128,
        out_ch_l2_0_2=128,
        out_ch_l2_0_ds=128,
        out_ch_l2_1_1=128,
        out_ch_l2_1_2=128,

        out_ch_l3_0_1=256,
        out_ch_l3_0_2=256,
        out_ch_l3_0_ds=256,
        out_ch_l3_1_1=256,
        out_ch_l3_1_2=256,

        out_ch_l4_0_1=512,
        out_ch_l4_0_2=512,
        out_ch_l4_0_ds=512,
        out_ch_l4_1_1=512,
        out_ch_l4_1_2=512,
    ):
        super(ResNet18, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, out_ch_conv1, kernel_size=3, stride=1, padding=1, bias=False
        )

        # END
        self.bn1 = norm_layer(out_ch_conv1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block=block, planes=64, blocks=layers[0],
                                       n_in_channels=out_ch_conv1,
                                       n_out_channels1=out_ch_l1_0_1,
                                       n_out_channels2=out_ch_l1_0_2,
                                       n_out_channels3=out_ch_l1_1_1,
                                       n_out_channels4=out_ch_l1_1_2,
                                       n_out_channels_ds=None,
                                       )

        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       n_in_channels=out_ch_l1_1_2,
                                       n_out_channels1=out_ch_l2_0_1,
                                       n_out_channels2=out_ch_l2_0_2,
                                       n_out_channels3=out_ch_l2_1_1,
                                       n_out_channels4=out_ch_l2_1_2,
                                       n_out_channels_ds=out_ch_l2_0_ds,
                                       )

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       n_in_channels=out_ch_l2_1_2,
                                       n_out_channels1=out_ch_l3_0_1,
                                       n_out_channels2=out_ch_l3_0_2,
                                       n_out_channels3=out_ch_l3_1_1,
                                       n_out_channels4=out_ch_l3_1_2,
                                       n_out_channels_ds=out_ch_l3_0_ds,
                                       )

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       n_in_channels=out_ch_l3_1_2,
                                       n_out_channels1=out_ch_l4_0_1,
                                       n_out_channels2=out_ch_l4_0_2,
                                       n_out_channels3=out_ch_l4_1_1,
                                       n_out_channels4=out_ch_l4_1_2,
                                       n_out_channels_ds=out_ch_l4_0_ds,
                                       )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_ch_l4_1_2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False,
                    n_in_channels=None,
                    n_out_channels1=None, n_out_channels2=None,
                    n_out_channels3=None, n_out_channels4=None,
                    n_out_channels_ds=None):

        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(n_in_channels, n_out_channels_ds, stride),
                norm_layer(n_out_channels_ds),
            )

        self.inplanes = planes * block.expansion
        layers = []

        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                n_in_channels=n_in_channels,
                n_out_channels1=n_out_channels1,
                n_out_channels2=n_out_channels2,
            )
        )
        layers.append(
            block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
                n_in_channels=n_out_channels2,
                n_out_channels1=n_out_channels3,
                n_out_channels2=n_out_channels4,
            )
        )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x
