# vgg16_bn.py COPYRIGHT Fujitsu Limited 2021

import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16_BN(nn.Module):
    def __init__(
        self,
        num_classes=10,
        ch1=64,
        ch2=64,
        ch3=128,
        ch4=128,
        ch5=256,
        ch6=256,
        ch7=256,
        ch8=512,
        ch9=512,
        ch10=512,
        ch11=512,
        ch12=512,
        ch13=512,
        ch14=512
    ):
        super(VGG16_BN, self).__init__()
        self.conv1 = nn.Conv2d(3, ch1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch1)

        self.conv2 = nn.Conv2d(ch1, ch2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(ch2, ch3, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(ch3)

        self.conv4 = nn.Conv2d(ch3, ch4, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(ch4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(ch4, ch5, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(ch5)

        self.conv6 = nn.Conv2d(ch5, ch6, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(ch6)

        self.conv7 = nn.Conv2d(ch6, ch7, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(ch7)
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Conv2d(ch7, ch8, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(ch8)

        self.conv9 = nn.Conv2d(ch8, ch9, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(ch9)

        self.conv10 = nn.Conv2d(ch9, ch10, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(ch10)
        self.pool10 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(ch10, ch11, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(ch11)

        self.conv12 = nn.Conv2d(ch11, ch12, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(ch12)

        self.conv13 = nn.Conv2d(ch12, ch13, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(ch13)
        self.pool13 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc14 = nn.Linear(ch13, ch14)
        self.bn14 = nn.BatchNorm1d(ch14)
        self.fc15 = nn.Linear(ch14, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x, inplace=True)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x, inplace=True)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x, inplace=True)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x, inplace=True)

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x, inplace=True)
        x = self.pool7(x)

        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x, inplace=True)

        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x, inplace=True)

        x = self.conv10(x)
        x = self.bn10(x)
        x = F.relu(x, inplace=True)
        x = self.pool10(x)

        x = self.conv11(x)
        x = self.bn11(x)
        x = F.relu(x, inplace=True)

        x = self.conv12(x)
        x = self.bn12(x)
        x = F.relu(x, inplace=True)

        x = self.conv13(x)
        x = self.bn13(x)
        x = F.relu(x, inplace=True)
        x = self.pool13(x)

        x = torch.flatten(x, 1)
        x = self.fc14(x)
        x = self.bn14(x)
        x = self.fc15(x)

        return x
