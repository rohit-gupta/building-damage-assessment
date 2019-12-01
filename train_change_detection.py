
import os
import sys

import torch
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.optim as optim
from apex import amp

from change_detection_model import ChangeDetectionNet
from config_parser import read_config
from dataset import xview_train_loader_factory
from utils import clean_distributed_state_dict
from utils import reconstruct_from_tiles
from functools import partial
from losses import cross_entropy, localization_aware_cross_entropy


config_name = os.environ["XVIEW_CONFIG"]
config = read_config(config_name, config_type="change")


reconstruct_full = partial(reconstruct_from_tiles, CHANNELS=5,
                           TILE_SIZE=config["dataloader"]["TILE_SIZE"],
                           ACTUAL_SIZE=config["dataloader"]["CROP_SIZE"])

if torch.cuda.is_available():
    gpu0 = torch.device('cuda:0')
    gpu1 = torch.device('cuda:1')
else:
    sys.exit("Cannot run with less than 2 GPUs")

semseg_model = deeplabv3_resnet50(pretrained=False,
                                  num_classes=5,
                                  aux_loss=None)

changenet = ChangeDetectionNet(classes=5, num_layers=config["change"]["NUM_LAYERS"], feature_channels=15,
                               kernel_scales=config["change"]["KERNEL_SIZES"],
                               dilation_scales=config["change"]["KERNEL_DILATIONS"],
                               use_bn=True, padding_type="replication")

if config["hyperparameters"]["OPTIMIZER"] == "ADAMW":
    optimizer = optim.AdamW(semseg_model.parameters(),
                            lr=config["hyperparameters"]["INITIAL_LR"],
                            weight_decay=config["hyperparameters"]["WEIGHT_DECAY"])
elif config["hyperparameters"]["OPTIMIZER"] == "SGD":
    optimizer = optim.SGD(semseg_model.parameters(),
                          lr=config["hyperparameters"]["INITIAL_LR"],
                          momentum=config["hyperparameters"]["MOMENTUM"],
                          weight_decay=config["hyperparameters"]["WEIGHT_DECAY"])

semseg_model = semseg_model.to(gpu0)
changenet = changenet.to(gpu1)


if config["misc"]["APEX_OPT_LEVEL"] != "None":
    changenet, optimizer = amp.initialize(changenet, optimizer, opt_level=config["misc"]["APEX_OPT_LEVEL"])

print(changenet)

trainloader, valloader = xview_train_loader_factory("change",
                                                    config["paths"]["XVIEW_ROOT"],
                                                    config["dataloader"]["DATA_VERSION"],
                                                    config["dataloader"]["USE_TIER3_TRAIN"],
                                                    config["dataloader"]["CROP_SIZE"],
                                                    config["dataloader"]["TILE_SIZE"],
                                                    config["dataloader"]["BATCH_SIZE"],
                                                    config["dataloader"]["THREADS"])

# print("Beginning Test Inference using model from Epoch #", BEST_EPOCH, ":")
models_folder = str(config["paths"]["MODELS"]) + config_name + "/"
semseg_model.load_state_dict(clean_distributed_state_dict(torch.load(config["change"]["BASELINE_SEG_MODEL"])))

CROP_BEGIN = (config["dataloader"]["CROP_SIZE"] - config["change"]["POST_CROP"])//2
CROP_END = config["dataloader"]["CROP_SIZE"] - CROP_BEGIN
BATCH_SIZE = config["dataloader"]["BATCH_SIZE"]
NUM_TILES = config["dataloader"]["CROP_SIZE"] // config["dataloader"]["TILE_SIZE"]
NUM_TILES *= NUM_TILES
semseg_model.eval()


for epoch in range(int(config["hyperparameters"]["NUM_EPOCHS"])):
    for idx, (pretiles, posttiles, prelabels, postlabels) in enumerate(trainloader):
        with torch.set_grad_enabled(False):
            segs = []
            labels = []
            for i in range(BATCH_SIZE):
                pretiles[i] = pretiles[i].to(gpu1)
                posttiles[i] = posttiles[i].to(gpu1)
                segmentations = semseg_model(torch.cat((pretiles[i], posttiles[i])))
                segmentations = segmentations.cpu()

                pre_seg = [reconstruct_full(segmentations[:NUM_TILES, :, :, :])]
                post_seg = [reconstruct_full(segmentations[NUM_TILES:, :, :, :])]
                segs += [torch.cat((pre_seg, post_seg), dim=0)]

                # pre_labels = reconstruct_full(prelabels[i].cpu())
                # post_labels = reconstruct_full(postlabels[i].cpu())
                # labels += [torch.cat((pre_labels[0, :, :], post_labels[1:, :, :]), dim=0)]
                labels = reconstruct_full(postlabels[i].cpu())

            input_batch = torch.stack(segs, dim=0)

            # crop off labels for edges
            labels_batch = torch.stack(labels[:, CROP_BEGIN:CROP_END, CROP_BEGIN:CROP_END], dim=0)

            # free up GPU memory, hopefully
            del labels
            del segs
            input_batch = input_batch.to(gpu1)
            labels_batch = labels_batch.to(gpu1)

        changenet.train()
        # zero the parameter gradients
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            preds = changenet(input_batch)
            cropped_preds = preds[:, CROP_BEGIN:CROP_END, CROP_BEGIN:CROP_END]


            if config["hyperparameters"]["LOSS"] == "crossentropy":
                loss = cross_entropy(labels_batch, cropped_preds)
            elif config["hyperparameters"]["LOSS"] == "locaware":
                loc_loss, cls_loss, loss = localization_aware_cross_entropy(labels_batch, cropped_preds,
                                                                            config["hyperparameters"]["LOCALIZATION_WEIGHT"],
                                                                            config["hyperparameters"]["CLASSIFICATION_WEIGHT"])

            if config["misc"]["APEX_OPT_LEVEL"] != "None":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

        #TODO save val samples
        


