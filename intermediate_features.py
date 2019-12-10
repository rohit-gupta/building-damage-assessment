

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
                                          config["dataloader"]["BATCH_SIZE"],
                                          config["dataloader"]["THREADS"],
                                          False)


MODELS_FOLDER = config["paths"]["MODELS"] + config_name + "/"
pathlib.Path(MODELS_FOLDER).mkdir(parents=True, exist_ok=True)

optimizer = optim.AdamW(changenet.parameters(),
                        lr=config["hyperparams"]["INITIAL_LR"],
                        weight_decay=config["hyperparams"]["WEIGHT_DECAY"])


if config["misc"]["APEX_OPT_LEVEL"] != "None":
    changenet, optimizer = amp.initialize(changenet, optimizer, opt_level=config["misc"]["APEX_OPT_LEVEL"])


# Logging
train_loss = AverageMeter("train_loss")
train_loss_log = MetricLog("train_loss")

for epoch in range(int(config["hyperparams"]["NUM_EPOCHS"])):
    train_pbar = tqdm.tqdm(total=len(trainloader))
    changenet = changenet.train()

    for idx, (pretiles, posttiles, prelabels, postlabels) in enumerate(trainloader):
        pretiles_batch = torch.cat(pretiles, dim=0).to(gpu1)
        posttiles_batch = torch.cat(posttiles, dim=0).to(gpu1)
        prelabels_batch = torch.cat(prelabels, dim=0)
        postlabels_batch = torch.cat(postlabels, dim=0)
        pre_seg = semseg_model(pretiles_batch)['out']
        post_seg = semseg_model(posttiles_batch)['out']
        # TODO Logits to probs for batch
        pre_features = layer1_features(pretiles_batch)
        post_features = layer1_features(posttiles_batch)
        change_input = torch.cat((pre_seg, post_seg, pre_features, post_features), dim=1)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            preds = changenet(change_input).cpu()

            if config["hyperparams"]["LOSS"] == "crossentropy":
                loss = cross_entropy(postlabels_batch, preds)
            elif config["hyperparams"]["LOSS"] == "locaware":
                loc_loss, cls_loss, loss = localization_aware_loss(postlabels_batch, preds,
                                                                config["hyperparams"]["LOC_WEIGHT"],
                                                                config["hyperparams"]["CLS_WEIGHT"])
            
            if config["misc"]["APEX_OPT_LEVEL"] != "None":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            optimizer.step()
            
        loss_val = loss.item()
        train_loss.update(val=loss_val)

        train_pbar.update(1)
    
    train_pbar.close()
    train_loss_log.update(train_loss.value())
    
    changenet = changenet.eval()

    for idx, (pretiles, posttiles, prelabels, postlabels) in enumerate(valloader):
        pretiles_batch = torch.cat(pretiles, dim=0).to(gpu1)
        posttiles_batch = torch.cat(posttiles, dim=0).to(gpu1)
        segmentations = semseg_model(torch.cat((pretiles_batch, posttiles_batch), dim=0))['out'].cpu()
        # TODO Complete this









# print(layer1_weights.shape)
# layer1_weights = F.interpolate(layer1_weights, scale_factor=20.0, mode='bilinear')
#
# save_image(layer1_weights, "layer1_weights_upsampled_bilinear.png")



