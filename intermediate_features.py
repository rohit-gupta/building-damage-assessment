

import os
import pathlib

import tqdm

import torch
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.utils import save_image

import torch.nn as nn
import torch.nn.functional as F

from config_parser import read_config
from utils import clean_distributed_state_dict
from dataset import xview_train_loader_factory

class FeaturesModel(nn.Module):
    def __init__(self, weight):
        super(FeaturesModel, self).__init__()
        self.conv = nn.Conv2d(in_channels=weight.shape[1], out_channels=weight.shape[0],
                              kernel_size=weight.shape[2], stride=1, padding=0, dilation=1,
                              groups=1, bias=False, padding_mode='zeros')

        self.conv.weight = weight

    def forward(self, x):
        return self.conv(x)


class RegressChangeNet(nn.Module):
    def __init__(self):
        super(RegressChangeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=138, out_channels=5,
                               kernel_size=5, stride=1, padding=0, dilation=5,
                               groups=1, bias=False, padding_mode='zeros')

        nn.init.kaiming_uniform_(self.conv.weight)

    def forward(self, x):
        return self.conv1(x)


config_name = os.environ["XVIEW_CONFIG"]
config = read_config(config_name, config_type="change")

semseg_model = deeplabv3_resnet50(pretrained=False,
                                  num_classes=5,
                                  aux_loss=None)

if torch.cuda.is_available():
    gpu0 = torch.device('cuda:0')
    gpu1 = torch.device('cuda:1')
semseg_model.to(gpu1)
semseg_model.load_state_dict(clean_distributed_state_dict(torch.load(config["change"]["SEG_MODEL"])))
semseg_model.eval()
for p in semseg_model.parameters():
    p.requires_grad = False

layer1_weights = semseg_model.backbone.conv1.weight
layer1_features = FeaturesModel(layer1_weights)
for p in layer1_features.parameters():
    p.requires_grad = False
layer1_features.to(gpu1)

changenet = RegressChangeNet()
for p in changenet.parameters():
    p.requires_grad = True
changenet.to(gpu0)
trainloader, _, _ = xview_train_loader_factory("change",
                                            config["paths"]["XVIEW_ROOT"],
                                            config["dataloader"]["DATA_VERSION"],
                                            config["dataloader"]["USE_TIER3_TRAIN"],
                                            config["dataloader"]["CROP_SIZE"],
                                            config["dataloader"]["CROP_SIZE"],  # No tiling
                                            config["dataloader"]["BATCH_SIZE"],
                                            config["dataloader"]["THREADS"],
                                            False)


_, valloader, _ = xview_train_loader_factory("segmentation",
                                          config["paths"]["XVIEW_ROOT"],
                                          config["dataloader"]["DATA_VERSION"],
                                          False,
                                          1024,
                                          1024,  # No tiling
                                          4,
                                          1,
                                          False)


MODELS_FOLDER = config["paths"]["MODELS"] + config_name + "/"
pathlib.Path(MODELS_FOLDER).mkdir(parents=True, exist_ok=True)

for epoch in range(int(config["hyperparams"]["NUM_EPOCHS"])):
    train_pbar = tqdm.tqdm(total=len(trainloader))
    changenet = changenet.train()

    for idx, (pretiles, posttiles, prelabels, postlabels) in enumerate(trainloader):
        pretiles_batch = torch.cat(pretiles, dim=0)
        posttiles_batch = torch.cat(posttiles, dim=0)
        prelabels_batch = torch.cat(prelabels, dim=0)
        postlabels_batch = torch.cat(postlabels, dim=0)




# print(layer1_weights.shape)
# layer1_weights = F.interpolate(layer1_weights, scale_factor=20.0, mode='bilinear')
#
# save_image(layer1_weights, "layer1_weights_upsampled_bilinear.png")



