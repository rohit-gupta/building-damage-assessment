
import os
import sys
import pathlib

import tqdm

import torch
from torchvision.models.segmentation import deeplabv3_resnet50
import torch.optim as optim
from apex import amp

from change_detection_model import ChangeDetectionNet
from config_parser import read_config
from dataset import xview_train_loader_factory
from utils import clean_distributed_state_dict
from utils import reconstruct_from_tiles
from utils import logits_to_probs
from functools import partial
from losses import cross_entropy, localization_aware_cross_entropy
from train_utils import save_model, save_val_results, save_val_gt, save_val_seg
from logger import MetricLog
from metrics import AverageMeter


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

changenet = ChangeDetectionNet(classes=5, num_layers=config["change"]["NUM_LAYERS"], feature_channels=5*len(config["change"]["KERNEL_SIZES"]),
                               kernel_scales=config["change"]["KERNEL_SIZES"],
                               dilation_scales=config["change"]["KERNEL_DILATIONS"],
                               use_bn=True, padding_type="replication")

if config["hyperparameters"]["OPTIMIZER"] == "ADAMW":
    optimizer = optim.AdamW(changenet.parameters(),
                            lr=config["hyperparameters"]["INITIAL_LR"],
                            weight_decay=config["hyperparameters"]["WEIGHT_DECAY"])
elif config["hyperparameters"]["OPTIMIZER"] == "SGD":
    optimizer = optim.SGD(changenet.parameters(),
                          lr=config["hyperparameters"]["INITIAL_LR"],
                          momentum=config["hyperparameters"]["MOMENTUM"],
                          weight_decay=config["hyperparameters"]["WEIGHT_DECAY"])

semseg_model = semseg_model.to(gpu0)
changenet = changenet.to(gpu1)


if config["misc"]["APEX_OPT_LEVEL"] != "None":
    torch.cuda.set_device(gpu1)
    changenet, optimizer = amp.initialize(changenet, optimizer, opt_level=config["misc"]["APEX_OPT_LEVEL"])

print(changenet)

trainloader, _, _ = xview_train_loader_factory("change",
                                            config["paths"]["XVIEW_ROOT"],
                                            config["dataloader"]["DATA_VERSION"],
                                            config["dataloader"]["USE_TIER3_TRAIN"],
                                            config["dataloader"]["CROP_SIZE"],
                                            config["dataloader"]["TILE_SIZE"],
                                            config["dataloader"]["BATCH_SIZE"],
                                            config["dataloader"]["THREADS"],
                                            False)


_, valloader, _ = xview_train_loader_factory("segmentation",
                                          config["paths"]["XVIEW_ROOT"],
                                          config["dataloader"]["DATA_VERSION"],
                                          False,
                                          1024,
                                          512,
                                          1,
                                          1,
                                          False)
# print("Beginning Test Inference using model from Epoch #", BEST_EPOCH, ":")
models_folder = str(config["paths"]["MODELS"]) + config_name + "/"
semseg_model.load_state_dict(clean_distributed_state_dict(torch.load(config["change"]["SEG_MODEL"])))

CROP_BEGIN = (config["dataloader"]["CROP_SIZE"] - config["change"]["POST_CROP"])//2
CROP_END = config["dataloader"]["CROP_SIZE"] - CROP_BEGIN
BATCH_SIZE = config["dataloader"]["BATCH_SIZE"]
NUM_TILES = config["dataloader"]["CROP_SIZE"] // config["dataloader"]["TILE_SIZE"]
NUM_TILES *= NUM_TILES
TILE_SIZE = config["dataloader"]["CROP_SIZE"]/NUM_TILES
semseg_model.eval()
for p in semseg_model.parameters():
    p.requires_grad = False

MODELS_FOLDER = config["paths"]["MODELS"] + config_name + "/"
pathlib.Path(MODELS_FOLDER).mkdir(parents=True, exist_ok=True)

# Logging
train_loss = AverageMeter("train_loss")
train_loss_log = MetricLog("train_loss")


for epoch in range(int(config["hyperparameters"]["NUM_EPOCHS"])):

    train_pbar = tqdm.tqdm(total=len(trainloader))
    changenet = changenet.train()
    train_loss.reset()
    for idx, (pretiles, posttiles, prelabels, postlabels) in enumerate(trainloader):
        with torch.set_grad_enabled(False):
            segs = []
            labels = []
            for i in range(len(pretiles)):
                pretiles[i] = pretiles[i].to(gpu0)
                posttiles[i] = posttiles[i].to(gpu0)
                segmentations = semseg_model(torch.cat((pretiles[i], posttiles[i])))['out']
                segmentations = segmentations.cpu()

                pre_seg = logits_to_probs(reconstruct_full(segmentations[:NUM_TILES, :, :, :]))
                post_seg = logits_to_probs(reconstruct_full(segmentations[NUM_TILES:, :, :, :]))
                segs += [torch.cat((pre_seg, post_seg), dim=0)]
                label_map = reconstruct_full(postlabels[i].cpu())
                # crop off labels for edges
                labels += [label_map[:, CROP_BEGIN: CROP_END, CROP_BEGIN: CROP_END]]

            input_batch = torch.stack(segs, dim=0)
            labels_batch = torch.stack(labels, dim=0)

            # free up GPU memory, hopefully
            del labels
            del segs
            input_batch = input_batch.to(gpu1)
            labels_batch = labels_batch.to(gpu1)

        # zero the parameter gradients
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            preds = changenet(input_batch)
            cropped_preds = preds[:, :, CROP_BEGIN:CROP_END, CROP_BEGIN:CROP_END]

            if config["hyperparameters"]["LOSS"] == "crossentropy":
                loss = cross_entropy(labels_batch, cropped_preds)
            elif config["hyperparameters"]["LOSS"] == "locaware":
                loc_loss, cls_loss, loss = localization_aware_cross_entropy(labels_batch, cropped_preds,
                                                                            config["hyperparameters"]["LOCALIZATION_WEIGHT"],
                                                                            config["hyperparameters"]["CLASSIFICATION_WEIGHT"])

            if config["misc"]["APEX_OPT_LEVEL"] != "None":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    optimizer.step()
            else:
                loss.backward()
                optimizer.step()
            loss_val = loss.item()
            train_loss.update(val=loss_val)

            train_pbar.update(1)

    train_loss_log.update(train_loss.value())
    changenet = changenet.eval()

    for idx, (pretiles, posttiles, prelabels, postlabels) in enumerate(valloader):

        pretiles[0] = pretiles[0].to(gpu0)
        posttiles[0] = posttiles[0].to(gpu0)
        segmentations = semseg_model(torch.cat((pretiles[0], posttiles[0])))['out']
        segmentations = segmentations.cpu()

        pre_seg = logits_to_probs(reconstruct_from_tiles(segmentations[:4, :, :, :], 5, 512, 1024))
        post_seg = logits_to_probs(reconstruct_from_tiles(segmentations[4:, :, :, :], 5, 512, 1024))
        seg = torch.cat((pre_seg, post_seg), dim=0)
        seg_result = torch.unsqueeze(seg, 0)
        labels = reconstruct_from_tiles(postlabels[0], 5, 512, 1024)

        seg_result = seg_result.to(gpu1)

        with torch.set_grad_enabled(False):
            preds = changenet(seg_result)[0, :, :, :]

        # Write to disk for visually tracking training progress
        save_path = "val_results/" + config_name + "/" + str(idx) + "/"
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        save_val_results(save_path, epoch, config["hyperparameters"]["LOSS"], preds, preds, tiled=False)
        # Save groundtruth
        if epoch == 0:  # Groundtruth only needs to be saved once
            save_val_gt(save_path, pretiles[0], posttiles[0], prelabels[0], postlabels[0], 512)
            save_val_seg(save_path, pre_seg, post_seg)
        # Save model
        if epoch % config["misc"]["SAVE_FREQ"] == 0:
            save_model(changenet.state_dict(), MODELS_FOLDER, epoch)
