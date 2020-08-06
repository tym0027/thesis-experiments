'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
# import IndependentComponent
from .IndependentComponent import *


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.ic1 = IndependentComponentLayer("STATIC-BATCH", 32, 32, in_channels=planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.ic2 = IndependentComponentLayer("STATIC-BATCH", 32, 32, in_channels=planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(self.expansion*planes)
                IndependentComponentLayer("STATIC-BATCH", 32, 32, in_channels=self.expansion*planes)
            )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.ic1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out = F.relu(self.ic2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        # print(in_planes, planes, self.expansion*planes) 
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.ic1 = IndependentComponentLayer("STATIC-BATCH", 32, 32, in_channels=in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        
        self.ic2 = IndependentComponentLayer("STATIC-BATCH", 32, 32, in_channels=planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.ic2 = IndependentComponentLayer("STATIC-BATCH", 32, 32, in_channels=planes)
        self.ic3 = IndependentComponentLayer("STATIC-BATCH", 32, 32, in_channels=planes)

        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        # self.ic3 = IndependentComponentLayer("STATIC-BATCH", 32, 32, in_channels=self.expansion*planes)

        # self.shortcut1 = nn.Sequential()
        self.shortcut1 = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                IndependentComponentLayer("STATIC-BATCH", 32, 32, in_channels=planes)
                # nn.BatchNorm2d(self.expansion*planes)
                # IndependentComponentLayer("STATIC-BATCH", 32, 32, in_channels=in_planes),    
                # nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )
    
    def forward(self, x):
        # print("X: ", x.shape)
        # out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(x)
        out = self.ic1(out)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.ic2(out)
        out = self.conv2(out)
        out = self.ic3(out)
        # print("out: ", out.shape)
        # feed_foward = self.shortcut1(x)
        out += self.shortcut1(x) # feed_foward

        out = self.conv3(out)
        # print("out: ", out.shape)
        # out += self.shortcut1(x)
        out = F.relu(out)
        return out


class ICResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ICResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.ic1 = IndependentComponentLayer("STATIC-BATCH", 32, 32, in_channels=64) 
        # self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.ic1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    
    def update_ic_layers_p(self):
        # start
        self.ic1.update_p()

        # layer 1
        self.layer1[0].ic1.update_p()
        self.layer1[0].ic2.update_p()
        self.layer1[0].ic3.update_p()
        self.layer1[0].shortcut1[1].update_p()
        self.layer1[1].ic1.update_p()
        self.layer1[1].ic2.update_p()
        self.layer1[1].ic3.update_p()
        self.layer1[1].shortcut1[1].update_p()
        self.layer1[2].ic1.update_p()
        self.layer1[2].ic2.update_p()
        self.layer1[2].ic3.update_p()
        self.layer1[2].shortcut1[1].update_p()

        # layer 2
        self.layer2[0].ic1.update_p()
        self.layer2[0].ic2.update_p()
        self.layer2[0].ic3.update_p()
        self.layer2[0].shortcut1[1].update_p()
        self.layer2[1].ic1.update_p()
        self.layer2[1].ic2.update_p()
        self.layer2[1].ic3.update_p()
        self.layer2[1].shortcut1[1].update_p()
        self.layer2[2].ic1.update_p()
        self.layer2[2].ic2.update_p()
        self.layer2[2].ic3.update_p()
        self.layer2[2].shortcut1[1].update_p()
        self.layer2[3].ic1.update_p()
        self.layer2[3].ic2.update_p()
        self.layer2[3].ic3.update_p()
        self.layer2[3].shortcut1[1].update_p()
        
        # layer 3
        self.layer3[0].ic1.update_p()
        self.layer3[0].ic2.update_p()
        self.layer3[0].ic3.update_p()
        self.layer3[0].shortcut1[1].update_p()
        self.layer3[1].ic1.update_p()
        self.layer3[1].ic2.update_p()
        self.layer3[1].ic3.update_p()
        self.layer3[1].shortcut1[1].update_p()
        self.layer3[2].ic1.update_p()
        self.layer3[2].ic2.update_p()
        self.layer3[2].ic3.update_p()
        self.layer3[2].shortcut1[1].update_p()
        self.layer3[3].ic1.update_p()
        self.layer3[3].ic2.update_p()
        self.layer3[3].ic3.update_p()
        self.layer3[3].shortcut1[1].update_p()
        self.layer3[4].ic1.update_p()
        self.layer3[4].ic2.update_p()
        self.layer3[4].ic3.update_p()
        self.layer3[4].shortcut1[1].update_p()
        self.layer3[5].ic1.update_p()
        self.layer3[5].ic2.update_p()
        self.layer3[5].ic3.update_p()
        self.layer3[5].shortcut1[1].update_p()

        # layer 4
        self.layer4[0].ic1.update_p()
        self.layer4[0].ic2.update_p()
        self.layer4[0].ic3.update_p()
        self.layer4[0].shortcut1[1].update_p()
        self.layer4[1].ic1.update_p()
        self.layer4[1].ic2.update_p()
        self.layer4[1].ic3.update_p()
        self.layer4[1].shortcut1[1].update_p()
        self.layer4[2].ic1.update_p()
        self.layer4[2].ic2.update_p()
        self.layer4[2].ic3.update_p()
        self.layer4[2].shortcut1[1].update_p()



    def update_ic_layers_ab(self, correct, incorrect):
        # start
        self.ic1.set_a_and_b(correct, incorrect)
        
        # layer 1
        self.layer1[0].ic1.set_a_and_b(correct, incorrect)
        self.layer1[0].ic2.set_a_and_b(correct, incorrect)
        self.layer1[0].ic3.set_a_and_b(correct, incorrect)
        self.layer1[0].shortcut1[1].set_a_and_b(correct, incorrect)
        self.layer1[1].ic1.set_a_and_b(correct, incorrect)
        self.layer1[1].ic2.set_a_and_b(correct, incorrect)
        self.layer1[1].ic3.set_a_and_b(correct, incorrect)
        self.layer1[1].shortcut1[1].set_a_and_b(correct, incorrect)
        self.layer1[2].ic1.set_a_and_b(correct, incorrect)
        self.layer1[2].ic2.set_a_and_b(correct, incorrect)
        self.layer1[2].ic3.set_a_and_b(correct, incorrect)
        self.layer1[2].shortcut1[1].set_a_and_b(correct, incorrect)
        
        # layer 2
        self.layer2[0].ic1.set_a_and_b(correct, incorrect)
        self.layer2[0].ic2.set_a_and_b(correct, incorrect)
        self.layer2[0].ic3.set_a_and_b(correct, incorrect)
        self.layer2[0].shortcut1[1].set_a_and_b(correct, incorrect)
        self.layer2[1].ic1.set_a_and_b(correct, incorrect)
        self.layer2[1].ic2.set_a_and_b(correct, incorrect)
        self.layer2[1].ic3.set_a_and_b(correct, incorrect)
        self.layer2[1].shortcut1[1].set_a_and_b(correct, incorrect)
        self.layer2[2].ic1.set_a_and_b(correct, incorrect)
        self.layer2[2].ic2.set_a_and_b(correct, incorrect)
        self.layer2[2].ic3.set_a_and_b(correct, incorrect)
        self.layer2[2].shortcut1[1].set_a_and_b(correct, incorrect)
        self.layer2[3].ic1.set_a_and_b(correct, incorrect)
        self.layer2[3].ic2.set_a_and_b(correct, incorrect)
        self.layer2[3].ic3.set_a_and_b(correct, incorrect)
        self.layer2[3].shortcut1[1].set_a_and_b(correct, incorrect)        

        # layer 3
        self.layer3[0].ic1.set_a_and_b(correct, incorrect)
        self.layer3[0].ic2.set_a_and_b(correct, incorrect)
        self.layer3[0].ic3.set_a_and_b(correct, incorrect)
        self.layer3[0].shortcut1[1].set_a_and_b(correct, incorrect)
        self.layer3[1].ic1.set_a_and_b(correct, incorrect)
        self.layer3[1].ic2.set_a_and_b(correct, incorrect)
        self.layer3[1].ic3.set_a_and_b(correct, incorrect)
        self.layer3[1].shortcut1[1].set_a_and_b(correct, incorrect)
        self.layer3[2].ic1.set_a_and_b(correct, incorrect)
        self.layer3[2].ic2.set_a_and_b(correct, incorrect)
        self.layer3[2].ic3.set_a_and_b(correct, incorrect)
        self.layer3[2].shortcut1[1].set_a_and_b(correct, incorrect)
        self.layer3[3].ic1.set_a_and_b(correct, incorrect)
        self.layer3[3].ic2.set_a_and_b(correct, incorrect)
        self.layer3[3].ic3.set_a_and_b(correct, incorrect)
        self.layer3[3].shortcut1[1].set_a_and_b(correct, incorrect)
        self.layer3[4].ic1.set_a_and_b(correct, incorrect)
        self.layer3[4].ic2.set_a_and_b(correct, incorrect)
        self.layer3[4].ic3.set_a_and_b(correct, incorrect)
        self.layer3[4].shortcut1[1].set_a_and_b(correct, incorrect)
        self.layer3[5].ic1.set_a_and_b(correct, incorrect)
        self.layer3[5].ic2.set_a_and_b(correct, incorrect)
        self.layer3[5].ic3.set_a_and_b(correct, incorrect)
        self.layer3[5].shortcut1[1].set_a_and_b(correct, incorrect)

        # layer 4
        self.layer4[0].ic1.set_a_and_b(correct, incorrect)
        self.layer4[0].ic2.set_a_and_b(correct, incorrect)
        self.layer4[0].ic3.set_a_and_b(correct, incorrect)
        self.layer4[0].shortcut1[1].set_a_and_b(correct, incorrect)
        self.layer4[1].ic1.set_a_and_b(correct, incorrect)
        self.layer4[1].ic2.set_a_and_b(correct, incorrect)
        self.layer4[1].ic3.set_a_and_b(correct, incorrect)
        self.layer4[1].shortcut1[1].set_a_and_b(correct, incorrect)
        self.layer4[2].ic1.set_a_and_b(correct, incorrect)
        self.layer4[2].ic2.set_a_and_b(correct, incorrect)
        self.layer4[2].ic3.set_a_and_b(correct, incorrect)
        self.layer4[2].shortcut1[1].set_a_and_b(correct, incorrect)



def ICResNet18():
    return ICResNet(BasicBlock, [2,2,2,2])

def ICResNet34():
    return ICResNet(BasicBlock, [3,4,6,3])

def ICResNet50(num_classes):
    return ICResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)

def ICResNet101():
    return ICResNet(Bottleneck, [3,4,23,3])

def ICResNet152():
    return ICResNet(Bottleneck, [3,8,36,3])


def test():
    net = ICResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
