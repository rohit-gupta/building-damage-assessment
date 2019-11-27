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
import torch

from dataset import xview_test_loader_factory
from utils import input_tensor_to_pil_img
from utils import postprocess_segmap_tensor_to_pil_img, postprocess_combined_predictions
from utils import reconstruct_from_tiles

# Configuration
import configparser
import os

config_name = os.environ["XVIEW_CONFIG"]
config_file = config_name + ".ini"
config = configparser.ConfigParser()
config.read(config_file)

BEST_EPOCH = config["paths"]["BEST_MODEL"]

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

semseg_model = deeplabv3_resnet50(pretrained=False,
                                  num_classes=5,
                                  aux_loss=None)
semseg_model = semseg_model.to(device)
# create dataloader
test_loader = xview_test_loader_factory(config["paths"]["XVIEW_ROOT"],
                                        int(config["dataloader"]["TILE_SIZE"]))

print("Beginning Test Inference using model from Epoch #" + str(BEST_EPOCH) + ":")
models_folder = str(config["paths"]["MODELS"]) + config_name + "/"
semseg_model.load_state_dict(torch.load(models_folder + str(BEST_EPOCH) + ".pth"))
semseg_model.eval()

test_pbar = tqdm.tqdm(total=len(test_loader))
# Validation Phase of epoch
# Assume batch_size = 1 (higher sizes are impractical)
for idx, (pretiles, posttiles) in enumerate(test_loader):
    n_val = len(pretiles)

    pretiles[0] = pretiles[0].to(device)
    posttiles[0] = posttiles[0].to(device)

    with torch.set_grad_enabled(False):
        preoutputs = semseg_model(pretiles[0])
        pre_preds = preoutputs['out']
        pre_preds = torch.nn.functional.softmax(pre_preds, dim=1)
        postoutputs = semseg_model(posttiles[0])
        post_preds = postoutputs['out']
        post_preds = torch.nn.functional.softmax(post_preds, dim=1)

    # Write to disk for scoring
    save_path = "test_results/" + config_name + "/"
    visual_results_path = save_path + "viz/"
    leaderboard_results_path = save_path + "leaderboard_targets/"
    pathlib.Path(visual_results_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(leaderboard_results_path).mkdir(parents=True, exist_ok=True)

    num_id = str(idx)
    num_id = "".join(["0"] * (5 - len(num_id)) + list(num_id))

    pre_pred = reconstruct_from_tiles(pre_preds, 5, int(config["dataloader"]["TILE_SIZE"]))
    post_pred = reconstruct_from_tiles(post_preds, 5, int(config["dataloader"]["TILE_SIZE"]))

    # Save results for leaderboard

    r = postprocess_segmap_tensor_to_pil_img(pre_pred, apply_color=False, binarize=True)
    r.save(leaderboard_results_path + "test_localization_" + num_id + "_prediction.png")
    r = postprocess_combined_predictions(pre_pred, post_pred, apply_color=False)
    r.save(leaderboard_results_path + "test_damage_" + num_id + "_prediction.png")

    # Save visual results for examination
    r = postprocess_segmap_tensor_to_pil_img(pre_pred, binarize=True)
    r.save(visual_results_path + num_id + "_pre_pred.png")
    r = postprocess_segmap_tensor_to_pil_img(post_pred)
    r.save(visual_results_path + num_id + "_post_pred.png")
    r = postprocess_combined_predictions(pre_pred, post_pred)
    r.save(visual_results_path + num_id + "_combo_pred.png")

    # Save input images as well for visual reference
    pre_img = reconstruct_from_tiles(pretiles[0], 3, int(config["dataloader"]["TILE_SIZE"]))
    post_img = reconstruct_from_tiles(posttiles[0], 3, int(config["dataloader"]["TILE_SIZE"]))

    pilimg = input_tensor_to_pil_img(pre_img)
    pilimg.save(visual_results_path + num_id + "_pre_image.png")
    pilimg = input_tensor_to_pil_img(post_img)
    pilimg.save(visual_results_path + num_id + "_post_image.png")

    test_pbar.update(1)

test_pbar.close()

