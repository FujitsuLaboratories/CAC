# vgg11.py COPYRIGHT Fujitsu Limited 2021

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG11(nn.Module):

    def __init__(
        self,
        num_classes=10,
        out_ch_conv1=64,
        out_ch_conv2=128,
        out_ch_conv3=256,
        out_ch_conv4=256,
        out_ch_conv5=512,
        out_ch_conv6=512,
        out_ch_conv7=512,
        out_ch_conv8=512,
        out_ch_fc1=4096,
        out_ch_fc2=4096
    ):
        super(VGG11, self).__init__()
        self.conv1 = nn.Conv2d(3, out_ch_conv1, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(out_ch_conv1, out_ch_conv2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(out_ch_conv2, out_ch_conv3, kernel_size=3, padding=1)

        self.conv4 = nn.Conv2d(out_ch_conv3, out_ch_conv4, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(out_ch_conv4, out_ch_conv5, kernel_size=3, padding=1)

        self.conv6 = nn.Conv2d(out_ch_conv5, out_ch_conv6, kernel_size=3, padding=1)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(out_ch_conv6, out_ch_conv7, kernel_size=3, padding=1)

        self.conv8 = nn.Conv2d(out_ch_conv7, out_ch_conv8, kernel_size=3, padding=1)
        self.pool8 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc1 = nn.Linear(out_ch_conv8 * 7 * 7, out_ch_fc1)
        self.drop1 = nn.Dropout()

        self.fc2 = nn.Linear(out_ch_fc1, out_ch_fc2)
        self.drop2 = nn.Dropout()
        self.fc3 = nn.Linear(out_ch_fc2, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x, inplace=True)

        x = self.conv4(x)
        x = F.relu(x, inplace=True)
        x = self.pool4(x)

        x = self.conv5(x)
        x = F.relu(x, inplace=True)

        x = self.conv6(x)
        x = F.relu(x, inplace=True)
        x = self.pool6(x)

        x = self.conv7(x)
        x = F.relu(x, inplace=True)

        x = self.conv8(x)
        x = F.relu(x, inplace=True)
        x = self.pool8(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.drop1(x)

        x = self.fc2(x)
        x = F.relu(x, inplace=True)
        x = self.drop2(x)
        x = self.fc3(x)

        return x
