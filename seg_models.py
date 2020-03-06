#!/usr/bin/env python3
"""
Implementation of DeepLabv3 and DeepLabv3+ using torchvision building blocks
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None

from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F


from torchvision.models.segmentation.deeplabv3 import ASPP, DeepLabHead as DeepLabv3Head
from torchvision.models.resnet import Bottleneck, BasicBlock
from torchvision.models.resnet import *
from torchvision.models._utils import IntermediateLayerGetter

from utils.misc import EMSG

RESNET_ARCHS = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnext50_32x4d': resnext50_32x4d,
    'resnext101_32x8d': resnext101_32x8d,
    'wide_resnet50_2': wide_resnet50_2,
    'wide_resnet101_2': wide_resnet101_2
}


RESNET_LOW_FEATURE_SIZES = {
    'resnet18': 64,
    'resnet34': 64,
    'resnet50': 256,
    'resnet101': 256,
    'resnet152': 256
}


RESNET_HIGH_FEATURE_SIZES = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048
}

# Commented out because defaul pytorch initializations seems good ? (Might test later)
# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
#         if m.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)

#         m.bias.data.fill_(0.01)

def create_resnet_backbone(arch=None, block=None, layers=None,
                           groups=None, width_multiple=None,
                           replace_stride_with_dilation=None,
                           pretrained=None):
    
    if arch is None and block is None:
        raise ValueError('Specify one of ResNet name or structure.')
    if arch and block:
        raise ValueError('Specify either ResNet name or structure, not both.')

    zero_init_residual = True
    if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, True, True]
            print(EMSG("INFO"), "Using default output stride of 16")

    if arch:
        if pretrained is None:
            pretrained = True
        progress = True
        model = RESNET_ARCHS[arch](pretrained, progress,
                                   zero_init_residual=zero_init_residual,
                                   replace_stride_with_dilation=replace_stride_with_dilation)
    else:
        if layers is None:
            raise ValueError('Specify ResNet Block structure when arch is missing.')
        if groups is None:
            groups = 1
        if width_multiple is None:
            width_per_group = 64
        else:
            width_per_group = width_multiple * 64 # For WideResNets in multiples of 64
        model = ResNet(block, layers, zero_init_residual=zero_init_residual,
                       groups=groups, width_per_group=width_per_group,
                       replace_stride_with_dilation=replace_stride_with_dilation )

    return model


class DeepLabv3PlusHead(nn.Module):
    """DeepLabv3+ Head"""
    def __init__(self, low_channels, low_channels_reduced, high_channels, num_classes, atrous_rates):
        super(DeepLabv3PlusHead, self).__init__()

        aspp_channels = 256

        feature_channels = low_channels_reduced + aspp_channels # Number of ASPP features
        self.low_features_reduce = nn.Sequential(nn.Conv2d(low_channels, low_channels_reduced, 1, padding=0, bias=False),
                                                 nn.BatchNorm2d(low_channels_reduced),
                                                 nn.ReLU(inplace=True))

        self.output_head = nn.Sequential(nn.Conv2d(feature_channels, aspp_channels, 3, padding=1, bias=False),
                                         nn.BatchNorm2d(aspp_channels),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.5),
                                         nn.Conv2d(aspp_channels, aspp_channels, 3, padding=1, bias=False),
                                         nn.BatchNorm2d(aspp_channels),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.1),
                                         nn.Conv2d(aspp_channels, num_classes, 1))

    def forward(self, features):

        low_features_size = features["low"].shape[-2:]
        # high_features_size = features["high"].shape[-2:]
        # print(low_features_size, high_features_size)

        
        x = F.interpolate(features["aspp"], size=low_features_size, mode='bilinear', align_corners=False)
        xl = self.low_features_reduce(features["low"])

        # print(x.shape, xl.shape)
        x = torch.cat((x, xl), 1)

        x = self.output_head(x)

        return x


class DeepLabv3PlusModel(nn.Module):
    """DeepLabv3+"""
    def __init__(self, arch, pretrained_backbone=True, output_stride=16, num_classes=20):
        super(DeepLabv3PlusModel, self).__init__()

        if output_stride == 16:
            replace_stride_with_dilation = [False, False, True]
            atrous_rates=[6, 12, 18]
        elif output_stride == 8:
            replace_stride_with_dilation = [False, True, True]
            atrous_rates=[12, 24, 36]
        else:
            raise ValueError('output_stride can be 8 or 16.')
        backbone = create_resnet_backbone(arch, pretrained=pretrained_backbone,
                                          replace_stride_with_dilation=replace_stride_with_dilation)
        return_layers = {'layer1': 'low', 'layer4': 'high'}
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.aspp_features = ASPP(high_channels, atrous_rates)
        self.classifier = DeepLabv3PlusHead(RESNET_LOW_FEATURE_SIZES[arch], 48, RESNET_HIGH_FEATURE_SIZES[arch], num_classes, atrous_rates)
    
    def forward(self, batch):

        result = OrderedDict()

        input_size = batch.shape[-2:]
        features = self.backbone(batch)
        features["aspp"] = self.aspp_features(features["high"])
        x = self.classifier(features)

        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        result["out"] = x
        result["features"] = features["aspp"]

        return result



class ChangeDetectionHead(nn.Module):
    """DeepLabv3+ Head"""
    def __init__(self, change_channels, num_classes):
        super(DeepLabv3PlusHead, self).__init__()

        aspp_channels = 256

        self.output_head = nn.Sequential(nn.Conv2d(aspp_channels, change_channels, 3, padding=1, bias=False),
                                         nn.BatchNorm2d(change_channels),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.5),
                                         nn.Conv2d(change_channels, change_channels, 3, padding=1, bias=False),
                                         nn.BatchNorm2d(change_channels),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(0.1),
                                         nn.Conv2d(change_channels, num_classes, 1))

    def forward(self, pre_features, post_features):

        features_diff = pre_features - post_features
        x = self.output_head(x)

        return x


if __name__ == "__main__":
    segmodel = DeepLabv3PlusModel(arch="resnet50", output_stride=8, pretrained_backbone=True, num_classes=5)
    dummy_batch = torch.rand(2, 3, 1024, 1024)
    result = segmodel(dummy_batch)
    print(segmodel)
    print(result.shape)
