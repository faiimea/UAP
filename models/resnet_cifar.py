"""CIFAR-style ResNet (He et al. 2015, section 4.2): 3 stages of n basic blocks,
16/32/64 channels, 8x8 average pool. resnet20 = n:3."""
import torch.nn as nn
import torch.nn.functional as F


class BasicBlockA(nn.Module):
    """Basic block with option-A (parameter-free, zero-padded) shortcut."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.pad = planes - in_planes
            self.shortcut = lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 0, self.pad))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + (self.shortcut(x) if self.shortcut is not None else x)
        return F.relu(out)


class ResNetCifar(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, num_blocks, stride=1)
        self.layer2 = self._make_layer(32, num_blocks, stride=2)
        self.layer3 = self._make_layer(64, num_blocks, stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlockA(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        return self.linear(out)


def resnet20(num_classes=10):
    return ResNetCifar(3, num_classes)


def resnet32(num_classes=10):
    return ResNetCifar(5, num_classes)
