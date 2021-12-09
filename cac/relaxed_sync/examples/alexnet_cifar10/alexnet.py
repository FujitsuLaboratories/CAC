# alexnet.py COPYRIGHT Fujitsu Limited 2021

#!/usr/bin/env python
# coding: utf-8

##### Reference #####
# https://github.com/sh-tatsuno/pytorch/blob/master/tutorials/Pytorch_Tutorials.ipynb
# https://github.com/sh-tatsuno/pytorch/blob/master/tutorials/Learning_PyTorch_with_Examples.ipynb
# https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
# how to chekc intermediate gradient
# https://tutorialmore.com/questions-1905405.htm
# https://discuss.pytorch.org/t/why-cant-i-see-grad-of-an-intermediate-variable/94/5
# AlexNet for cifar10
# http://cedro3.com/ai/pytorch-alexnet/

# ====================================
# how to run DNN training with pytorch
# 1. import library
# 2. load dataset
# 3. define network model
#	- network structure
#	- loss function
#	- optimizer
# 4. run training
# 5. run test
# ====================================
# import library
import torch
import torch.nn as nn
import torch.nn.functional as F
# ====================================


## To change "channels for conv layer" & "nodes for fc layer" by pruning, custum model is defined.
# for CIFAR-10
class AlexNet(nn.Module):
    def __init__(
        self,
        num_classes=10,
        out_ch_conv1=64,
        out_ch_conv2=256,
        out_ch_conv3=384,
        out_ch_conv4=256,
        out_ch_conv5=256,
        out_ch_fc1=4096,
        out_ch_fc2=4096
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
