

import os

import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.utils import save_image

from torch.nn.functional import interpolate

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


layer1_weights = semseg_model.backbone.conv1.weight
print(layer1_weights.shape)
layer1_weights = interpolate(layer1_weights, scale_factor=20.0, mode='nearest')

save_image(layer1_weights, "layer1_weights_upsampled.png")