#!/usr/bin/env python3
"""
Training script for xview challenge
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None

import numpy as np
from PIL import Image
import pathlib
import tqdm

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn

from dsilva_metrics.iou import IoU
from logger import MetricLog
from metrics import AverageMeter
from dataset import xview_train_loader_factory
from utils import input_tensor_to_pil_img
from utils import segmap_tensor_to_pil_img
from utils import reconstruct_from_tiles

# Configuration
import configparser
import os

config_name = os.environ["XVIEW_CONFIG"]
config_file = config_name + ".ini"
config = configparser.ConfigParser()
config.read(config_file)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

semseg_model = deeplabv3_resnet50(pretrained=False,
                                  num_classes=5,
                                  aux_loss=None)
semseg_model = semseg_model.to(device)
# create dataloader
_, valloader = xview_train_loader_factory(config["paths"]["XVIEW_ROOT"],
                                          int(config["dataloader"]["CROP_SIZE"]),
                                          int(config["dataloader"]["BATCH_SIZE"]),
                                          int(config["dataloader"]["THREADS"]))

for epoch in range(int(config["hyperparameters"]["NUM_EPOCHS"])):

    print("Beginning Inference for Epoch #" + str(epoch) + ":")
    models_folder = str(config["paths"]["MODELS"]) + config_name + "/"
    semseg_model.load_state_dict(torch.load(models_folder + str(epoch) + ".pth"))
    semseg_model.eval()

    val_pbar = tqdm.tqdm(total=len(valloader))
    # Validation Phase of epoch
    # Assume batch_size = 1 (higher sizes are impractical)
    for idx, (pretiles, posttiles, prelabels, postlabels) in enumerate(valloader):
        n_val = len(pretiles)

        pretiles[0] = pretiles[0].to(device)
        posttiles[0] = posttiles[0].to(device)

        prelabels[0] = prelabels[0].to(device)
        postlabels[0] = postlabels[0].to(device)

        with torch.set_grad_enabled(False):
            preoutputs = semseg_model(pretiles[0])
            pre_preds = preoutputs['out']
            postoutputs = semseg_model(posttiles[0])
            post_preds = postoutputs['out']

        # Write to disk for scoring

        save_path = "val_results/" + str(idx) + "/"
        pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
        pre_pred = reconstruct_from_tiles(pre_preds, 5, int(config["dataloader"]["CROP_SIZE"]))
        post_pred = reconstruct_from_tiles(post_preds, 5, int(config["dataloader"]["CROP_SIZE"]))

        r = segmap_tensor_to_pil_img(pre_pred)
        r.save(save_path + "pre_pred_epoch_" + str(epoch) + ".png")
        r = segmap_tensor_to_pil_img(post_pred)
        r.save(save_path + "post_pred_epoch_" + str(epoch) + ".png")

        # Groundtruth only needs to be saved once
        if epoch == 0:
            pre_img = reconstruct_from_tiles(pretiles[0], 3, int(config["dataloader"]["CROP_SIZE"]))
            post_img = reconstruct_from_tiles(posttiles[0], 3, int(config["dataloader"]["CROP_SIZE"]))

            pre_gt = reconstruct_from_tiles(prelabels[0], 5, int(config["dataloader"]["CROP_SIZE"]))
            post_gt = reconstruct_from_tiles(postlabels[0], 5, int(config["dataloader"]["CROP_SIZE"]))

            pilimg = input_tensor_to_pil_img(pre_img)
            pilimg.save(save_path + "pre_image.png")
            pilimg = input_tensor_to_pil_img(post_img)
            pilimg.save(save_path + "post_image.png")

            r = segmap_tensor_to_pil_img(pre_gt)
            r.save(save_path + "pre_gt.png")
            r = segmap_tensor_to_pil_img(post_gt)
            r.save(save_path + "post_gt.png")
        val_pbar.update(1)

    val_pbar.close()

