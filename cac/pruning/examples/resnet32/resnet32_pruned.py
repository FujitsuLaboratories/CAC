# resnet32_pruned.py COPYRIGHT Fujitsu Limited 2021

import torch
import torch.nn as nn
import torch.nn.functional as F

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

# To change "channels for conv layer" & "nodes for fc layer" by pruning, custum model is defined.
# for CIFAR-10
class ResNet32(nn.Module):
    def __init__(
        self,
        num_classes=10,

        ch_conv1=16,

        ch_l10conv1=8,
        ch_l10conv2=16,
        ch_l11conv1=8,
        ch_l11conv2=16,
        ch_l12conv1=7,
        ch_l12conv2=16,
        ch_l13conv1=8,
        ch_l13conv2=16,
        ch_l14conv1=6,
        ch_l14conv2=16,

        ch_l20conv1=17,
        ch_l20conv2=32,
        ch_l21conv1=11,
        ch_l21conv2=32,
        ch_l22conv1=17,
        ch_l22conv2=32,
        ch_l23conv1=17,
        ch_l23conv2=32,
        ch_l24conv1=12,
        ch_l24conv2=32,

        ch_l30conv1=24,
        ch_l30conv2=63,
        ch_l31conv1=18,
        ch_l31conv2=63,
        ch_l32conv1=18,
        ch_l32conv2=63,
        ch_l33conv1=18,
        ch_l33conv2=63,
        ch_l34conv1=18,
        ch_l34conv2=63,
    ):
        super(ResNet32, self).__init__()
        self.conv1 = nn.Conv2d(3, ch_conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_conv1)
        self.relu = nn.ReLU(inplace=True)

        # layer1-0
        self.l10_conv1 = nn.Conv2d(ch_conv1, ch_l10conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l10_bn1   = nn.BatchNorm2d(ch_l10conv1)
        self.l10_conv2 = nn.Conv2d(ch_l10conv1, ch_l10conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l10_bn2   = nn.BatchNorm2d(ch_l10conv2)
        # layer1-1
        self.l11_conv1 = nn.Conv2d(ch_l10conv2, ch_l11conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l11_bn1   = nn.BatchNorm2d(ch_l11conv1)
        self.l11_conv2 = nn.Conv2d(ch_l11conv1, ch_l11conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l11_bn2   = nn.BatchNorm2d(ch_l11conv2)
        # layer1-2
        self.l12_conv1 = nn.Conv2d(ch_l11conv2, ch_l12conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l12_bn1   = nn.BatchNorm2d(ch_l12conv1)
        self.l12_conv2 = nn.Conv2d(ch_l12conv1, ch_l12conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l12_bn2   = nn.BatchNorm2d(ch_l12conv2)
        # layer1-3
        self.l13_conv1 = nn.Conv2d(ch_l12conv2, ch_l13conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l13_bn1   = nn.BatchNorm2d(ch_l13conv1)
        self.l13_conv2 = nn.Conv2d(ch_l13conv1, ch_l13conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l13_bn2   = nn.BatchNorm2d(ch_l13conv2)
        # layer1-4
        self.l14_conv1 = nn.Conv2d(ch_l13conv2, ch_l14conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l14_bn1   = nn.BatchNorm2d(ch_l14conv1)
        self.l14_conv2 = nn.Conv2d(ch_l14conv1, ch_l14conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l14_bn2   = nn.BatchNorm2d(ch_l14conv2)

        # zero padding : add zero to resize tensor 16 -> 32
        ch_diff12 = ch_l20conv2 - ch_l14conv2
        # just through input to output
        self.zeropad11  = LambdaLayer(lambda x:
                                     F.pad(x[:, :, :, :], (0, 0, 0, 0, 0, 0), "constant", 0))
        self.zeropad12  = LambdaLayer(lambda x:
                                     F.pad(x[:, :, :, :], (0, 0, 0, 0, 0, 0), "constant", 0))

        if ch_diff12 >0:
            if ch_diff12%2 ==0:
                self.zeropad11 = LambdaLayer(lambda x: 
                                             F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, ch_diff12//2, ch_diff12//2), "constant", 0))
            else:
                self.zeropad11 = LambdaLayer(lambda x:
                                             F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, ch_diff12//2, (ch_diff12//2)+1), "constant", 0))
        elif ch_diff12 <0:
            ch_diff12 = ch_diff12 * -1.0
            if ch_diff12%2 ==0:
                self.zeropad12 = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, ch_diff12//2, ch_diff12//2), "constant", 0))
            else:
                self.zeropad12 = LambdaLayer(lambda x:
                                             F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, ch_diff12//2, (ch_diff12//2)+1), "constant", 0))


        # layer2-0
        self.l20_conv1 = nn.Conv2d(ch_l14conv2, ch_l20conv1, kernel_size=3, stride=2, padding=1, bias=False)
        self.l20_bn1   = nn.BatchNorm2d(ch_l20conv1)
        self.l20_conv2 = nn.Conv2d(ch_l20conv1, ch_l20conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l20_bn2   = nn.BatchNorm2d(ch_l20conv2)
        # layer2-1
        self.l21_conv1 = nn.Conv2d(ch_l20conv2, ch_l21conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l21_bn1   = nn.BatchNorm2d(ch_l21conv1)
        self.l21_conv2 = nn.Conv2d(ch_l21conv1, ch_l21conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l21_bn2   = nn.BatchNorm2d(ch_l21conv2)
        # layer2-2
        self.l22_conv1 = nn.Conv2d(ch_l21conv2, ch_l22conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l22_bn1   = nn.BatchNorm2d(ch_l22conv1)
        self.l22_conv2 = nn.Conv2d(ch_l22conv1, ch_l22conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l22_bn2   = nn.BatchNorm2d(ch_l22conv2)
        # layer2-3
        self.l23_conv1 = nn.Conv2d(ch_l22conv2, ch_l23conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l23_bn1   = nn.BatchNorm2d(ch_l23conv1)
        self.l23_conv2 = nn.Conv2d(ch_l23conv1, ch_l23conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l23_bn2   = nn.BatchNorm2d(ch_l23conv2)
        # layer2-4
        self.l24_conv1 = nn.Conv2d(ch_l23conv2, ch_l24conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l24_bn1   = nn.BatchNorm2d(ch_l24conv1)
        self.l24_conv2 = nn.Conv2d(ch_l24conv1, ch_l24conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l24_bn2   = nn.BatchNorm2d(ch_l24conv2)

        # zero padding : add zero to resize tensor 32 -> 64 
        ch_diff23 = ch_l30conv2 - ch_l24conv2
        # just through input to output
        self.zeropad21  = LambdaLayer(lambda x:
                                     F.pad(x[:, :, :, :], (0, 0, 0, 0, 0, 0), "constant", 0))
        self.zeropad22  = LambdaLayer(lambda x:
                                     F.pad(x[:, :, :, :], (0, 0, 0, 0, 0, 0), "constant", 0))
        
        #x.size ([mini-batch, out_ch, feature_map_size, feature_map_size])
        #x[:,:,::2,::2] : downsample input_feature_map to half
        if ch_diff23 >0:
            if ch_diff23%2 ==0:
                self.zeropad21 = LambdaLayer(lambda x:
                                             F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, ch_diff23//2, ch_diff23//2), "constant", 0))
            else:
                self.zeropad21 = LambdaLayer(lambda x:
                                             F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, ch_diff23//2, (ch_diff23//2)+1), "constant", 0))
        elif ch_diff23 <0:
            ch_diff23 = ch_diff23 * -1.0
            if ch_diff23%2 ==0:
                self.zeropad22 = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, ch_diff23//2, ch_diff23//2), "constant", 0))
            else:
                self.zeropad22 = LambdaLayer(lambda x:
                                             F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, ch_diff23//2, (ch_diff23//2)+1), "constant", 0))

        # layer3-0
        self.l30_conv1 = nn.Conv2d(ch_l24conv2, ch_l30conv1, kernel_size=3, stride=2, padding=1, bias=False)
        self.l30_bn1   = nn.BatchNorm2d(ch_l30conv1)
        self.l30_conv2 = nn.Conv2d(ch_l30conv1, ch_l30conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l30_bn2   = nn.BatchNorm2d(ch_l30conv2)
        # layer3-1
        self.l31_conv1 = nn.Conv2d(ch_l30conv2, ch_l31conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l31_bn1   = nn.BatchNorm2d(ch_l31conv1)
        self.l31_conv2 = nn.Conv2d(ch_l31conv1, ch_l31conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l31_bn2   = nn.BatchNorm2d(ch_l31conv2)
        # layer3-2
        self.l32_conv1 = nn.Conv2d(ch_l31conv2, ch_l32conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l32_bn1   = nn.BatchNorm2d(ch_l32conv1)
        self.l32_conv2 = nn.Conv2d(ch_l32conv1, ch_l32conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l32_bn2   = nn.BatchNorm2d(ch_l32conv2)
        # layer3-3
        self.l33_conv1 = nn.Conv2d(ch_l32conv2, ch_l33conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l33_bn1   = nn.BatchNorm2d(ch_l33conv1)
        self.l33_conv2 = nn.Conv2d(ch_l33conv1, ch_l33conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l33_bn2   = nn.BatchNorm2d(ch_l33conv2)
        # layer3-4
        self.l34_conv1 = nn.Conv2d(ch_l33conv2, ch_l34conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l34_bn1   = nn.BatchNorm2d(ch_l34conv1)
        self.l34_conv2 = nn.Conv2d(ch_l34conv1, ch_l34conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l34_bn2   = nn.BatchNorm2d(ch_l34conv2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(ch_l34conv2, num_classes)

    def forward(self, x):
        x = self.conv1(x) 
        x = self.bn1(x)     
        x = self.relu(x)

        # layer1-0
        identity = x
        x = F.relu(self.l10_bn1(self.l10_conv1(x)))
        x = self.l10_bn2(self.l10_conv2(x))
        x += identity
        x = F.relu(x)
        # layer1-1
        identity = x
        x = F.relu(self.l11_bn1(self.l11_conv1(x)))
        x = self.l11_bn2(self.l11_conv2(x))
        x += identity
        x = F.relu(x)
        # layer1-2
        identity = x
        x = F.relu(self.l12_bn1(self.l12_conv1(x)))
        x = self.l12_bn2(self.l12_conv2(x))
        x += identity
        x = F.relu(x)
        # layer1-3
        identity = x
        x = F.relu(self.l13_bn1(self.l13_conv1(x)))
        x = self.l13_bn2(self.l13_conv2(x))
        x += identity
        x = F.relu(x)
        # layer1-4
        identity = x
        x = F.relu(self.l14_bn1(self.l14_conv1(x)))
        x = self.l14_bn2(self.l14_conv2(x))
        x += identity
        x = F.relu(x)

        # layer2-0
        identity = x
        x = F.relu(self.l20_bn1(self.l20_conv1(x)))
        x = self.l20_bn2(self.l20_conv2(x))
        x = self.zeropad12(x)                    # zero padding on main path
        identity = self.zeropad11(identity)      # zero padding on shortcut path
        x += identity
        x = F.relu(x)
        # layer2-1
        identity = x
        x = F.relu(self.l21_bn1(self.l21_conv1(x)))
        x = self.l21_bn2(self.l21_conv2(x))
        x += identity
        x = F.relu(x)
        # layer2-2
        identity = x
        x = F.relu(self.l22_bn1(self.l22_conv1(x)))
        x = self.l22_bn2(self.l22_conv2(x))
        x += identity
        x = F.relu(x)
        # layer2-3
        identity = x
        x = F.relu(self.l23_bn1(self.l23_conv1(x)))
        x = self.l23_bn2(self.l23_conv2(x))
        x += identity
        x = F.relu(x)
        # layer2-4
        identity = x
        x = F.relu(self.l24_bn1(self.l24_conv1(x)))
        x = self.l24_bn2(self.l24_conv2(x))
        x += identity
        x = F.relu(x)

        # layer3-0   
        identity = x
        x = F.relu(self.l30_bn1(self.l30_conv1(x)))
        x = self.l30_bn2(self.l30_conv2(x))
        x = self.zeropad22(x)                    # zero padding on main path
        identity = self.zeropad21(identity)      # zero padding on shortcut path
        x += identity
        x = F.relu(x)
        # layer3-1
        identity = x
        x = F.relu(self.l31_bn1(self.l31_conv1(x)))
        x = self.l31_bn2(self.l31_conv2(x))
        x += identity
        x = F.relu(x)
        # layer3-2
        identity = x
        x = F.relu(self.l32_bn1(self.l32_conv1(x)))
        x = self.l32_bn2(self.l32_conv2(x))
        x += identity
        x = F.relu(x)
        # layer3-3
        identity = x
        x = F.relu(self.l33_bn1(self.l33_conv1(x)))
        x = self.l33_bn2(self.l33_conv2(x))
        x += identity
        x = F.relu(x)
        # layer3-4
        identity = x
        x = F.relu(self.l34_bn1(self.l34_conv1(x)))
        x = self.l34_bn2(self.l34_conv2(x))
        x += identity
        x = F.relu(x)
        
        x = F.avg_pool2d(x,x.size()[3])
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        
        return x
