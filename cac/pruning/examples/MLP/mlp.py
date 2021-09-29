# mlp.py COPYRIGHT Fujitsu Limited 2021

import torch.nn as nn
import torch.nn.functional as F

# for MNIST
class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        num_classes=10,
        out_ch_fc1=1024,
        out_ch_fc2=1024
    ):
        super(MultiLayerPerceptron, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, out_ch_fc1)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(out_ch_fc1, out_ch_fc2)
        self.drop2 = nn.Dropout()
        self.fc3 = nn.Linear(out_ch_fc2, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc3(x)
        return x
