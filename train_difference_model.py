


import os

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.utils import save_image

from config_parser import read_config
from utils import clean_distributed_state_dict

class FeaturesModel(nn.Module):
    def __init__(self, weight):
        super(FeaturesModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=weight.shape[1], out_channels=weight.shape[0],
                              kernel_size=weight.shape[2], stride=1, padding=weight.shape[2]//2, dilation=1,
                              groups=1, bias=False, padding_mode='zeros')

        self.conv.weight = weight

    def forward(self, x):
        return self.conv(x)

config_name = os.environ["XVIEW_CONFIG"]
config = read_config(config_name, config_type="change")


semseg_model = deeplabv3_resnet50(pretrained=False,
                                  num_classes=5,
                                  aux_loss=None)
state_dict = clean_distributed_state_dict(torch.load(config["change"]["SEG_MODEL"], map_location=lambda storage, location: storage))
semseg_model.load_state_dict(state_dict)

