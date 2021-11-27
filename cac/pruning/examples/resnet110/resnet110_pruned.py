# resnet110_pruned.py COPYRIGHT Fujitsu Limited 2021

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
class ResNet110(nn.Module):
    def __init__(
        self,
        num_classes=10,

        ch_conv1=16,

        ch_l10conv1=5,
        ch_l10conv2=16,
        ch_l11conv1=5,
        ch_l11conv2=16,
        ch_l12conv1=5,
        ch_l12conv2=16,
        ch_l13conv1=5,
        ch_l13conv2=16,
        ch_l14conv1=5,
        ch_l14conv2=16,
        ch_l15conv1=5,
        ch_l15conv2=16,
        ch_l16conv1=5,
        ch_l16conv2=16,
        ch_l17conv1=5,
        ch_l17conv2=16,
        ch_l18conv1=5,
        ch_l18conv2=16,
        ch_l19conv1=5,
        ch_l19conv2=16,
        ch_l110conv1=5,
        ch_l110conv2=16,
        ch_l111conv1=5,
        ch_l111conv2=16,
        ch_l112conv1=5,
        ch_l112conv2=16,
        ch_l113conv1=5,
        ch_l113conv2=16,
        ch_l114conv1=5,
        ch_l114conv2=16,
        ch_l115conv1=5,
        ch_l115conv2=16,
        ch_l116conv1=5,
        ch_l116conv2=16,
        ch_l117conv1=5,
        ch_l117conv2=16,

        ch_l20conv1=5,
        ch_l20conv2=32,
        ch_l21conv1=5,
        ch_l21conv2=32,
        ch_l22conv1=5,
        ch_l22conv2=32,
        ch_l23conv1=5,
        ch_l23conv2=32,
        ch_l24conv1=5,
        ch_l24conv2=32,
        ch_l25conv1=5,
        ch_l25conv2=32,
        ch_l26conv1=5,
        ch_l26conv2=32,
        ch_l27conv1=5,
        ch_l27conv2=32,
        ch_l28conv1=5,
        ch_l28conv2=32,
        ch_l29conv1=5,
        ch_l29conv2=32,
        ch_l210conv1=5,
        ch_l210conv2=32,
        ch_l211conv1=5,
        ch_l211conv2=32,
        ch_l212conv1=5,
        ch_l212conv2=32,
        ch_l213conv1=5,
        ch_l213conv2=32,
        ch_l214conv1=5,
        ch_l214conv2=32,
        ch_l215conv1=5,
        ch_l215conv2=32,
        ch_l216conv1=5,
        ch_l216conv2=32,
        ch_l217conv1=5,
        ch_l217conv2=32,

        ch_l30conv1=10,
        ch_l30conv2=64,
        ch_l31conv1=9,
        ch_l31conv2=64,
        ch_l32conv1=9,
        ch_l32conv2=64,
        ch_l33conv1=9,
        ch_l33conv2=64,
        ch_l34conv1=9,
        ch_l34conv2=64,
        ch_l35conv1=9,
        ch_l35conv2=64,
        ch_l36conv1=9,
        ch_l36conv2=64,
        ch_l37conv1=9,
        ch_l37conv2=64,
        ch_l38conv1=9,
        ch_l38conv2=64,
        ch_l39conv1=9,
        ch_l39conv2=64,
        ch_l310conv1=9,
        ch_l310conv2=64,
        ch_l311conv1=9,
        ch_l311conv2=64,
        ch_l312conv1=9,
        ch_l312conv2=64,
        ch_l313conv1=9,
        ch_l313conv2=64,
        ch_l314conv1=9,
        ch_l314conv2=64,
        ch_l315conv1=9,
        ch_l315conv2=64,
        ch_l316conv1=9,
        ch_l316conv2=64,
        ch_l317conv1=9,
        ch_l317conv2=64,
    ):
        super(ResNet110, self).__init__()
        self.conv1 = nn.Conv2d(3, ch_conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch_conv1)

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
        #layer1-4
        self.l14_conv1 = nn.Conv2d(ch_l13conv2, ch_l14conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l14_bn1   = nn.BatchNorm2d(ch_l14conv1)
        self.l14_conv2 = nn.Conv2d(ch_l14conv1, ch_l14conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l14_bn2   = nn.BatchNorm2d(ch_l14conv2)
        #layer1-5
        self.l15_conv1 = nn.Conv2d(ch_l14conv2, ch_l15conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l15_bn1   = nn.BatchNorm2d(ch_l15conv1)
        self.l15_conv2 = nn.Conv2d(ch_l15conv1, ch_l15conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l15_bn2   = nn.BatchNorm2d(ch_l15conv2)
        #layer1-6
        self.l16_conv1 = nn.Conv2d(ch_l15conv2, ch_l16conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l16_bn1   = nn.BatchNorm2d(ch_l16conv1)
        self.l16_conv2 = nn.Conv2d(ch_l16conv1, ch_l16conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l16_bn2   = nn.BatchNorm2d(ch_l16conv2)
        #layer1-7
        self.l17_conv1 = nn.Conv2d(ch_l16conv2, ch_l17conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l17_bn1   = nn.BatchNorm2d(ch_l17conv1)
        self.l17_conv2 = nn.Conv2d(ch_l17conv1, ch_l17conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l17_bn2   = nn.BatchNorm2d(ch_l17conv2)
        #layer1-8
        self.l18_conv1 = nn.Conv2d(ch_l17conv2, ch_l18conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l18_bn1   = nn.BatchNorm2d(ch_l18conv1)
        self.l18_conv2 = nn.Conv2d(ch_l18conv1, ch_l18conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l18_bn2   = nn.BatchNorm2d(ch_l18conv2)
        #layer1-9
        self.l19_conv1 = nn.Conv2d(ch_l18conv2, ch_l19conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l19_bn1   = nn.BatchNorm2d(ch_l19conv1)
        self.l19_conv2 = nn.Conv2d(ch_l19conv1, ch_l19conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l19_bn2   = nn.BatchNorm2d(ch_l19conv2)
        #layer1-10       
        self.l110_conv1 = nn.Conv2d(ch_l19conv2, ch_l110conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l110_bn1   = nn.BatchNorm2d(ch_l110conv1)
        self.l110_conv2 = nn.Conv2d(ch_l110conv1, ch_l110conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l110_bn2   = nn.BatchNorm2d(ch_l110conv2)
        #layer1-11                
        self.l111_conv1 = nn.Conv2d(ch_l110conv2, ch_l111conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l111_bn1   = nn.BatchNorm2d(ch_l111conv1)
        self.l111_conv2 = nn.Conv2d(ch_l111conv1, ch_l111conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l111_bn2   = nn.BatchNorm2d(ch_l111conv2)
        #layer1-12
        self.l112_conv1 = nn.Conv2d(ch_l111conv2, ch_l112conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l112_bn1   = nn.BatchNorm2d(ch_l112conv1)
        self.l112_conv2 = nn.Conv2d(ch_l112conv1, ch_l112conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l112_bn2   = nn.BatchNorm2d(ch_l112conv2)
        #layer1-13
        self.l113_conv1 = nn.Conv2d(ch_l112conv2, ch_l113conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l113_bn1   = nn.BatchNorm2d(ch_l113conv1)
        self.l113_conv2 = nn.Conv2d(ch_l113conv1, ch_l113conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l113_bn2   = nn.BatchNorm2d(ch_l113conv2)
        #layer1-14        
        self.l114_conv1 = nn.Conv2d(ch_l113conv2, ch_l114conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l114_bn1   = nn.BatchNorm2d(ch_l114conv1)
        self.l114_conv2 = nn.Conv2d(ch_l114conv1, ch_l114conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l114_bn2   = nn.BatchNorm2d(ch_l114conv2)
        #layer1-15     
        self.l115_conv1 = nn.Conv2d(ch_l114conv2, ch_l115conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l115_bn1   = nn.BatchNorm2d(ch_l115conv1)
        self.l115_conv2 = nn.Conv2d(ch_l115conv1, ch_l115conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l115_bn2   = nn.BatchNorm2d(ch_l115conv2)
        #layer1-16  
        self.l116_conv1 = nn.Conv2d(ch_l115conv2, ch_l116conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l116_bn1   = nn.BatchNorm2d(ch_l116conv1)
        self.l116_conv2 = nn.Conv2d(ch_l116conv1, ch_l116conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l116_bn2   = nn.BatchNorm2d(ch_l116conv2)
        #layer1-17      
        self.l117_conv1 = nn.Conv2d(ch_l116conv2, ch_l117conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l117_bn1   = nn.BatchNorm2d(ch_l117conv1)
        self.l117_conv2 = nn.Conv2d(ch_l117conv1, ch_l117conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l117_bn2   = nn.BatchNorm2d(ch_l117conv2)

        # zero padding 1 : add zero to resize tensor 16 -> 32
        ch_diff12 = ch_l20conv2 - ch_l117conv2
        # just through input to output
        self.zeropad11  = LambdaLayer(lambda x: F.pad(x[:, :, :, :], (0, 0, 0, 0, 0, 0), "constant", 0))
        self.zeropad12  = LambdaLayer(lambda x: F.pad(x[:, :, :, :], (0, 0, 0, 0, 0, 0), "constant", 0))
        
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
        self.l20_conv1 = nn.Conv2d(ch_l117conv2, ch_l20conv1, kernel_size=3, stride=2, padding=1, bias=False)
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
        #layer2-4
        self.l24_conv1 = nn.Conv2d(ch_l23conv2, ch_l24conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l24_bn1   = nn.BatchNorm2d(ch_l24conv1)
        self.l24_conv2 = nn.Conv2d(ch_l24conv1, ch_l24conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l24_bn2   = nn.BatchNorm2d(ch_l24conv2)
        #layer2-5
        self.l25_conv1 = nn.Conv2d(ch_l24conv2, ch_l25conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l25_bn1   = nn.BatchNorm2d(ch_l25conv1)
        self.l25_conv2 = nn.Conv2d(ch_l25conv1, ch_l25conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l25_bn2   = nn.BatchNorm2d(ch_l25conv2)
        #layer2-6
        self.l26_conv1 = nn.Conv2d(ch_l25conv2, ch_l26conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l26_bn1   = nn.BatchNorm2d(ch_l26conv1)
        self.l26_conv2 = nn.Conv2d(ch_l26conv1, ch_l26conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l26_bn2   = nn.BatchNorm2d(ch_l26conv2)
        #layer2-7
        self.l27_conv1 = nn.Conv2d(ch_l26conv2, ch_l27conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l27_bn1   = nn.BatchNorm2d(ch_l27conv1)
        self.l27_conv2 = nn.Conv2d(ch_l27conv1, ch_l27conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l27_bn2   = nn.BatchNorm2d(ch_l27conv2)
        #layer2-8
        self.l28_conv1 = nn.Conv2d(ch_l27conv2, ch_l28conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l28_bn1   = nn.BatchNorm2d(ch_l28conv1)
        self.l28_conv2 = nn.Conv2d(ch_l28conv1, ch_l28conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l28_bn2   = nn.BatchNorm2d(ch_l28conv2)
        #layer2-9
        self.l29_conv1 = nn.Conv2d(ch_l28conv2, ch_l29conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l29_bn1   = nn.BatchNorm2d(ch_l29conv1)
        self.l29_conv2 = nn.Conv2d(ch_l29conv1, ch_l29conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l29_bn2   = nn.BatchNorm2d(ch_l29conv2)
        #layer2-10
        self.l210_conv1 = nn.Conv2d(ch_l29conv2, ch_l210conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l210_bn1   = nn.BatchNorm2d(ch_l210conv1)
        self.l210_conv2 = nn.Conv2d(ch_l210conv1, ch_l210conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l210_bn2   = nn.BatchNorm2d(ch_l210conv2)
        #layer2-11
        self.l211_conv1 = nn.Conv2d(ch_l210conv2, ch_l211conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l211_bn1   = nn.BatchNorm2d(ch_l211conv1)
        self.l211_conv2 = nn.Conv2d(ch_l211conv1, ch_l211conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l211_bn2   = nn.BatchNorm2d(ch_l211conv2)
        #layer2-12
        self.l212_conv1 = nn.Conv2d(ch_l211conv2, ch_l212conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l212_bn1   = nn.BatchNorm2d(ch_l212conv1)
        self.l212_conv2 = nn.Conv2d(ch_l212conv1, ch_l212conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l212_bn2   = nn.BatchNorm2d(ch_l212conv2)
        #layer2-13
        self.l213_conv1 = nn.Conv2d(ch_l212conv2, ch_l213conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l213_bn1   = nn.BatchNorm2d(ch_l213conv1)
        self.l213_conv2 = nn.Conv2d(ch_l213conv1, ch_l213conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l213_bn2   = nn.BatchNorm2d(ch_l213conv2)
        #layer2-14
        self.l214_conv1 = nn.Conv2d(ch_l213conv2, ch_l214conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l214_bn1   = nn.BatchNorm2d(ch_l214conv1)
        self.l214_conv2 = nn.Conv2d(ch_l214conv1, ch_l214conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l214_bn2   = nn.BatchNorm2d(ch_l214conv2)
        #layer2-15
        self.l215_conv1 = nn.Conv2d(ch_l214conv2, ch_l215conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l215_bn1   = nn.BatchNorm2d(ch_l215conv1)
        self.l215_conv2 = nn.Conv2d(ch_l215conv1, ch_l215conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l215_bn2   = nn.BatchNorm2d(ch_l215conv2)
        #layer2-16
        self.l216_conv1 = nn.Conv2d(ch_l215conv2, ch_l216conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l216_bn1   = nn.BatchNorm2d(ch_l216conv1)
        self.l216_conv2 = nn.Conv2d(ch_l216conv1, ch_l216conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l216_bn2   = nn.BatchNorm2d(ch_l216conv2)
        #layer2-17
        self.l217_conv1 = nn.Conv2d(ch_l216conv2, ch_l217conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l217_bn1   = nn.BatchNorm2d(ch_l217conv1)
        self.l217_conv2 = nn.Conv2d(ch_l217conv1, ch_l217conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l217_bn2   = nn.BatchNorm2d(ch_l217conv2)


        # zero padding 2 : add zero to resize tensor 32 -> 64 
        ch_diff23 = ch_l30conv2 - ch_l217conv2
        # just through input to output
        self.zeropad21  = LambdaLayer(lambda x: F.pad(x[:, :, :, :], (0, 0, 0, 0, 0, 0), "constant", 0))
        self.zeropad22  = LambdaLayer(lambda x: F.pad(x[:, :, :, :], (0, 0, 0, 0, 0, 0), "constant", 0))
        
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
        self.l30_conv1 = nn.Conv2d(ch_l217conv2, ch_l30conv1, kernel_size=3, stride=2, padding=1, bias=False)
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
        # layer3-5
        self.l35_conv1 = nn.Conv2d(ch_l34conv2, ch_l35conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l35_bn1   = nn.BatchNorm2d(ch_l35conv1)
        self.l35_conv2 = nn.Conv2d(ch_l35conv1, ch_l35conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l35_bn2   = nn.BatchNorm2d(ch_l35conv2)
        # layer3-6
        self.l36_conv1 = nn.Conv2d(ch_l35conv2, ch_l36conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l36_bn1   = nn.BatchNorm2d(ch_l36conv1)
        self.l36_conv2 = nn.Conv2d(ch_l36conv1, ch_l36conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l36_bn2   = nn.BatchNorm2d(ch_l36conv2)
        # layer3-7
        self.l37_conv1 = nn.Conv2d(ch_l36conv2, ch_l37conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l37_bn1   = nn.BatchNorm2d(ch_l37conv1)
        self.l37_conv2 = nn.Conv2d(ch_l37conv1, ch_l37conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l37_bn2   = nn.BatchNorm2d(ch_l37conv2)
        # layer3-8
        self.l38_conv1 = nn.Conv2d(ch_l37conv2, ch_l38conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l38_bn1   = nn.BatchNorm2d(ch_l38conv1)
        self.l38_conv2 = nn.Conv2d(ch_l38conv1, ch_l38conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l38_bn2   = nn.BatchNorm2d(ch_l38conv2)
        # layer3-9
        self.l39_conv1 = nn.Conv2d(ch_l38conv2, ch_l39conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l39_bn1   = nn.BatchNorm2d(ch_l39conv1)
        self.l39_conv2 = nn.Conv2d(ch_l39conv1, ch_l39conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l39_bn2   = nn.BatchNorm2d(ch_l39conv2)
        # layer3-10
        self.l310_conv1 = nn.Conv2d(ch_l39conv2, ch_l310conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l310_bn1   = nn.BatchNorm2d(ch_l310conv1)
        self.l310_conv2 = nn.Conv2d(ch_l310conv1, ch_l310conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l310_bn2   = nn.BatchNorm2d(ch_l310conv2)
        # layer3-11
        self.l311_conv1 = nn.Conv2d(ch_l310conv2, ch_l311conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l311_bn1   = nn.BatchNorm2d(ch_l311conv1)
        self.l311_conv2 = nn.Conv2d(ch_l311conv1, ch_l311conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l311_bn2   = nn.BatchNorm2d(ch_l311conv2)
        # layer3-12
        self.l312_conv1 = nn.Conv2d(ch_l311conv2, ch_l312conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l312_bn1   = nn.BatchNorm2d(ch_l312conv1)
        self.l312_conv2 = nn.Conv2d(ch_l312conv1, ch_l312conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l312_bn2   = nn.BatchNorm2d(ch_l312conv2)
        # layer3-13
        self.l313_conv1 = nn.Conv2d(ch_l312conv2, ch_l313conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l313_bn1   = nn.BatchNorm2d(ch_l313conv1)
        self.l313_conv2 = nn.Conv2d(ch_l313conv1, ch_l313conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l313_bn2   = nn.BatchNorm2d(ch_l313conv2)
        # layer3-14
        self.l314_conv1 = nn.Conv2d(ch_l313conv2, ch_l314conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l314_bn1   = nn.BatchNorm2d(ch_l314conv1)
        self.l314_conv2 = nn.Conv2d(ch_l314conv1, ch_l314conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l314_bn2   = nn.BatchNorm2d(ch_l314conv2)
        # layer3-15
        self.l315_conv1 = nn.Conv2d(ch_l314conv2, ch_l315conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l315_bn1   = nn.BatchNorm2d(ch_l315conv1)
        self.l315_conv2 = nn.Conv2d(ch_l315conv1, ch_l315conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l315_bn2   = nn.BatchNorm2d(ch_l315conv2)
        # layer3-16
        self.l316_conv1 = nn.Conv2d(ch_l315conv2, ch_l316conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l316_bn1   = nn.BatchNorm2d(ch_l316conv1)
        self.l316_conv2 = nn.Conv2d(ch_l316conv1, ch_l316conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l316_bn2   = nn.BatchNorm2d(ch_l316conv2)
        # layer3-17
        self.l317_conv1 = nn.Conv2d(ch_l316conv2, ch_l317conv1, kernel_size=3, stride=1, padding=1, bias=False)
        self.l317_bn1   = nn.BatchNorm2d(ch_l317conv1)
        self.l317_conv2 = nn.Conv2d(ch_l317conv1, ch_l317conv2, kernel_size=3, stride=1, padding=1, bias=False)
        self.l317_bn2   = nn.BatchNorm2d(ch_l317conv2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(ch_l317conv2, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) 

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
        # layer1-5
        identity = x
        x = F.relu(self.l15_bn1(self.l15_conv1(x)))
        x = self.l15_bn2(self.l15_conv2(x))
        x += identity
        x = F.relu(x)
        # layer1-6
        identity = x
        x = F.relu(self.l16_bn1(self.l16_conv1(x)))
        x = self.l16_bn2(self.l16_conv2(x))
        x += identity
        x = F.relu(x)
        # layer1-7
        identity = x
        x = F.relu(self.l17_bn1(self.l17_conv1(x)))
        x = self.l17_bn2(self.l17_conv2(x))
        x += identity
        x = F.relu(x)
        # layer1-8
        identity = x
        x = F.relu(self.l18_bn1(self.l18_conv1(x)))
        x = self.l18_bn2(self.l18_conv2(x))
        x += identity
        x = F.relu(x)
        # layer1-9
        identity = x
        x = F.relu(self.l19_bn1(self.l19_conv1(x)))
        x = self.l19_bn2(self.l19_conv2(x))
        x += identity
        x = F.relu(x)
        # layer1-10
        identity = x
        x = F.relu(self.l110_bn1(self.l110_conv1(x)))
        x = self.l110_bn2(self.l110_conv2(x))
        x += identity
        x = F.relu(x)
        # layer1-11
        identity = x
        x = F.relu(self.l111_bn1(self.l111_conv1(x)))
        x = self.l111_bn2(self.l111_conv2(x))
        x += identity
        x = F.relu(x)
        # layer1-12
        identity = x
        x = F.relu(self.l112_bn1(self.l112_conv1(x)))
        x = self.l112_bn2(self.l112_conv2(x))
        x += identity
        x = F.relu(x)
        # layer1-13
        identity = x
        x = F.relu(self.l113_bn1(self.l113_conv1(x)))
        x = self.l113_bn2(self.l113_conv2(x))
        x += identity
        x = F.relu(x)
        # layer1-14
        identity = x
        x = F.relu(self.l114_bn1(self.l114_conv1(x)))
        x = self.l114_bn2(self.l114_conv2(x))
        x += identity
        x = F.relu(x)
        # layer1-15
        identity = x
        x = F.relu(self.l115_bn1(self.l115_conv1(x)))
        x = self.l115_bn2(self.l115_conv2(x))
        x += identity
        x = F.relu(x)
        # layer1-16
        identity = x
        x = F.relu(self.l116_bn1(self.l116_conv1(x)))
        x = self.l116_bn2(self.l116_conv2(x))
        x += identity
        x = F.relu(x)
        # layer1-17
        identity = x
        x = F.relu(self.l117_bn1(self.l117_conv1(x)))
        x = self.l117_bn2(self.l117_conv2(x))
        x += identity
        x = F.relu(x)


        # layer2-0
        identity = x
        x = F.relu(self.l20_bn1(self.l20_conv1(x)))
        x = self.l20_bn2(self.l20_conv2(x))
        x = self.zeropad12(x)                    # zero padding on main path
        x += self.zeropad11(identity)            # zero padding on shortcut path
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
        # layer2-5
        identity = x
        x = F.relu(self.l25_bn1(self.l25_conv1(x)))
        x = self.l25_bn2(self.l25_conv2(x))
        x += identity
        x = F.relu(x)
        # layer2-6
        identity = x
        x = F.relu(self.l26_bn1(self.l26_conv1(x)))
        x = self.l26_bn2(self.l26_conv2(x))
        x += identity
        x = F.relu(x)
        # layer2-7
        identity = x
        x = F.relu(self.l27_bn1(self.l27_conv1(x)))
        x = self.l27_bn2(self.l27_conv2(x))
        x += identity
        x = F.relu(x)
        # layer2-8
        identity = x
        x = F.relu(self.l28_bn1(self.l28_conv1(x)))
        x = self.l28_bn2(self.l28_conv2(x))
        x += identity
        x = F.relu(x)
        # layer2-9
        identity = x
        x = F.relu(self.l29_bn1(self.l29_conv1(x)))
        x = self.l29_bn2(self.l29_conv2(x))
        x += identity
        x = F.relu(x)
        # layer2-10
        identity = x
        x = F.relu(self.l210_bn1(self.l210_conv1(x)))
        x = self.l210_bn2(self.l210_conv2(x))
        x += identity
        x = F.relu(x)
        # layer2-11
        identity = x
        x = F.relu(self.l211_bn1(self.l211_conv1(x)))
        x = self.l211_bn2(self.l211_conv2(x))
        x += identity
        x = F.relu(x)
        # layer2-12
        identity = x
        x = F.relu(self.l212_bn1(self.l212_conv1(x)))
        x = self.l212_bn2(self.l212_conv2(x))
        x += identity
        x = F.relu(x)
        # layer2-13
        identity = x
        x = F.relu(self.l213_bn1(self.l213_conv1(x)))
        x = self.l213_bn2(self.l213_conv2(x))
        x += identity
        x = F.relu(x)
        # layer2-14
        identity = x
        x = F.relu(self.l214_bn1(self.l214_conv1(x)))
        x = self.l214_bn2(self.l214_conv2(x))
        x += identity
        x = F.relu(x)
        # layer2-15
        identity = x
        x = F.relu(self.l215_bn1(self.l215_conv1(x)))
        x = self.l215_bn2(self.l215_conv2(x))
        x += identity
        x = F.relu(x)
        # layer2-16
        identity = x
        x = F.relu(self.l216_bn1(self.l216_conv1(x)))
        x = self.l216_bn2(self.l216_conv2(x))
        x += identity
        x = F.relu(x)
        # layer2-17
        identity = x
        x = F.relu(self.l217_bn1(self.l217_conv1(x)))
        x = self.l217_bn2(self.l217_conv2(x))
        x += identity
        x = F.relu(x)


        # layer3-0   
        identity = x
        x = F.relu(self.l30_bn1(self.l30_conv1(x)))
        x = self.l30_bn2(self.l30_conv2(x))
        x = self.zeropad22(x)                    # zero padding on main path
        x+= self.zeropad21(identity)             # zero padding on shortcut path
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
        # layer3-5
        identity = x
        x = F.relu(self.l35_bn1(self.l35_conv1(x)))
        x = self.l35_bn2(self.l35_conv2(x))
        x += identity
        x = F.relu(x)
        # layer3-6
        identity = x
        x = F.relu(self.l36_bn1(self.l36_conv1(x)))
        x = self.l36_bn2(self.l36_conv2(x))
        x += identity
        x = F.relu(x)
        # layer3-7
        identity = x
        x = F.relu(self.l37_bn1(self.l37_conv1(x)))
        x = self.l37_bn2(self.l37_conv2(x))
        x += identity
        x = F.relu(x)
        # layer3-8
        identity = x
        x = F.relu(self.l38_bn1(self.l38_conv1(x)))
        x = self.l38_bn2(self.l38_conv2(x))
        x += identity
        x = F.relu(x)
        # layer3-9
        identity = x
        x = F.relu(self.l39_bn1(self.l39_conv1(x)))
        x = self.l39_bn2(self.l39_conv2(x))
        x += identity
        x = F.relu(x)
        # layer3-10
        identity = x
        x = F.relu(self.l310_bn1(self.l310_conv1(x)))
        x = self.l310_bn2(self.l310_conv2(x))
        x += identity
        x = F.relu(x)
        # layer3-11
        identity = x
        x = F.relu(self.l311_bn1(self.l311_conv1(x)))
        x = self.l311_bn2(self.l311_conv2(x))
        x += identity
        x = F.relu(x)
        # layer3-12
        identity = x
        x = F.relu(self.l312_bn1(self.l312_conv1(x)))
        x = self.l312_bn2(self.l312_conv2(x))
        x += identity
        x = F.relu(x)
        # layer3-13
        identity = x
        x = F.relu(self.l313_bn1(self.l313_conv1(x)))
        x = self.l313_bn2(self.l313_conv2(x))
        x += identity
        x = F.relu(x)
        # layer3-14
        identity = x
        x = F.relu(self.l314_bn1(self.l314_conv1(x)))
        x = self.l314_bn2(self.l314_conv2(x))
        x += identity
        x = F.relu(x)
        # layer3-15
        identity = x
        x = F.relu(self.l315_bn1(self.l315_conv1(x)))
        x = self.l315_bn2(self.l315_conv2(x))
        x += identity
        x = F.relu(x)
        # layer3-16
        identity = x
        x = F.relu(self.l316_bn1(self.l316_conv1(x)))
        x = self.l316_bn2(self.l316_conv2(x))
        x += identity
        x = F.relu(x)
        # layer3-17
        identity = x
        x = F.relu(self.l317_bn1(self.l317_conv1(x)))
        x = self.l317_bn2(self.l317_conv2(x))
        x += identity
        x = F.relu(x)
        
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        
        return x
