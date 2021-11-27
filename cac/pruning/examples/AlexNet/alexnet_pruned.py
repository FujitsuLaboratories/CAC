# alexnet_pruned.py COPYRIGHT Fujitsu Limited 2021

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(
        self,
        num_classes=10,
        out_ch_conv1=33,
        out_ch_conv2=95,
        out_ch_conv3=142,
        out_ch_conv4=60,
        out_ch_conv5=67,
        out_ch_fc1=74,
        out_ch_fc2=859
    ):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, out_ch_conv1, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(out_ch_conv1, out_ch_conv2, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(out_ch_conv2, out_ch_conv3, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(out_ch_conv3, out_ch_conv4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(out_ch_conv4, out_ch_conv5, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.drop1 = nn.Dropout()
        self.fc1 = nn.Linear(out_ch_conv5 * 4 * 4, out_ch_fc1)
        self.drop2 = nn.Dropout()
        self.fc2 = nn.Linear(out_ch_fc1, out_ch_fc2)
        self.fc3 = nn.Linear(out_ch_fc2, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.drop1(x)
        x = self.fc1(x)

        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)

        return x
