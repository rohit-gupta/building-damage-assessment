

import os

import torch
from torchvision.models.segmentation import deeplabv3_resnet50

from config_parser import read_config
from utils import clean_distributed_state_dict

config_name = os.environ["XVIEW_CONFIG"]
config = read_config(config_name, config_type="change")

semseg_model = deeplabv3_resnet50(pretrained=False,
                                  num_classes=5,
                                  aux_loss=None)

if torch.cuda.is_available():
    gpu = torch.device('cuda:0')
semseg_model.to(gpu)
semseg_model.load_state_dict(clean_distributed_state_dict(torch.load(config["change"]["SEG_MODEL"])))
print(semseg_model.backbone.conv1.weight.shape)