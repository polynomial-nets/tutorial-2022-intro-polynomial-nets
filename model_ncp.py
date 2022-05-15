'''Model for Π-net based 2nd degree blocks without activation functions:
https://ieeexplore.ieee.org/document/9353253 (or https://arxiv.org/abs/2006.13026). 

This file implements an NCP-based product of polynomials.
'''
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_norm(norm_local):
    """ Define the appropriate function for normalization. """
    if norm_local is None or norm_local == 0:
        norm_local = nn.BatchNorm2d
    elif norm_local == 1:
        norm_local = nn.InstanceNorm2d
    elif isinstance(norm_local, int) and norm_local < 0:
        norm_local = lambda a: lambda x: x
    return norm_local


class SinglePoly(nn.Module):
    def __init__(self, in_planes, planes, stride=1, use_alpha=False, kernel_sz=3,
                 norm_S=None, norm_layer=None, kernel_size_S=1,
                 use_only_first_conv=False, **kwargs):
        """ This class implements a single second degree NCP model. """ 
        super(SinglePoly, self).__init__()
        self._norm_layer = get_norm(norm_layer)
        self._norm_S = get_norm(norm_S)
        self.use_only_first_conv = use_only_first_conv

        pad1 = kernel_sz // 2
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_sz, stride=stride, padding=pad1, bias=False)
        self.bn1 = self._norm_layer(planes)
        if not self.use_only_first_conv:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = self._norm_layer(planes)

        pad = kernel_size_S // 2
        self.conv_S = nn.Conv2d(in_planes, planes, kernel_size=kernel_size_S, stride=stride, padding=pad, bias=False)
        self.bnS = self._norm_S(planes)

        self.use_alpha = use_alpha
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.zeros(1))
            self.monitor_alpha = []

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        if not self.use_only_first_conv:
            out = self.bn2(self.conv2(out))
        out1 = self.bnS(self.conv_S(x))
        out_so = out * out1
        if self.use_alpha:
            out1 = out1 + self.alpha * out_so
            self.monitor_alpha.append(self.alpha)
        else:
            out1 = out1 + out_so
        return out1


class ModelNCP(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm_layer=None,
                 pool_adapt=False, n_channels=[64, 128, 256, 512], ch_in=3, **kwargs):
        super(ModelNCP, self).__init__()
        self.in_planes = n_channels[0]
        self._norm_layer = nn.BatchNorm2d if norm_layer is None else get_norm(norm_layer)
        assert len(n_channels) >= 4
        self.n_channels = n_channels
        self.pool_adapt = pool_adapt
        if pool_adapt:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = partial(F.avg_pool2d, kernel_size=4)

        self.conv1 = nn.Conv2d(ch_in, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self._norm_layer(n_channels[0])
        self.layer1 = self._make_layer(block, n_channels[0], num_blocks[0], stride=1, **kwargs)
        self.layer2 = self._make_layer(block, n_channels[1], num_blocks[1], stride=2, **kwargs)
        self.layer3 = self._make_layer(block, n_channels[2], num_blocks[2], stride=2, **kwargs)
        self.layer4 = self._make_layer(block, n_channels[3], num_blocks[3], stride=2, **kwargs)
        self.linear = nn.Linear(n_channels[-1], num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, **kwargs):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, norm_layer=self._norm_layer, **kwargs))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ModelNCP_wrapper(num_blocks=None, **kwargs):
    if num_blocks is None:
        num_blocks = [1, 1, 1, 1]
    return ModelNCP(SinglePoly, num_blocks, **kwargs)


def test():
    net = ModelNCP_wrapper()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
