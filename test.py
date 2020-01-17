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
from utils.post_processing import input_tensor_to_pil_img
from utils.post_processing import logits_to_probs, postprocess_segmap_tensor_to_pil_img, postprocess_combined_predictions
from utils.post_processing import reconstruct_from_tiles
from utils.misc import clean_distributed_state_dict

# Configuration
import configparser
import os

config_name = os.environ["XVIEW_CONFIG"]
config_file = "configs/" + config_name + ".ini"
config = configparser.ConfigParser()
config.read(config_file)

BEST_EPOCH = os.environ["XVIEW_BEST_EPOCH"]
INFERENCE_SIZE = int(os.environ["INFERENCE_SIZE"])
THRESHOLD = float(os.environ["THRESHOLD"])
# BEST_EPOCH = 98

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
                                        INFERENCE_SIZE)
                                        # config["dataloader"]["TILE_SIZE"])

print("Using inference size = ", INFERENCE_SIZE)
print("Beginning Test Inference using model from Epoch #" + str(BEST_EPOCH) + ":")
models_folder = str(config["paths"]["MODELS"]) + config_name + "/"

state_dict = torch.load(models_folder + str(BEST_EPOCH) + ".pth", map_location=torch.device("cuda:0")) # Load on GPU 0
state_dict = clean_distributed_state_dict(state_dict)
semseg_model.load_state_dict(state_dict)
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
        postoutputs = semseg_model(posttiles[0])
        post_preds = postoutputs['out']

    if INFERENCE_SIZE < 1024:
        pre_preds = reconstruct_from_tiles(pre_preds, 5, int(config["dataloader"]["TILE_SIZE"]))
        post_preds = reconstruct_from_tiles(post_preds, 5, int(config["dataloader"]["TILE_SIZE"]))
    else:
        pre_preds = pre_preds[0]
        post_preds = post_preds[0]

    if config["hyperparams"]["LOSS"] == "crossentropy":
        pre_probs = torch.nn.functional.softmax(pre_preds, dim=1)
        post_probs = torch.nn.functional.softmax(post_preds, dim=1)
    elif config["hyperparams"]["LOSS"] == "locaware":
        pre_probs = logits_to_probs(pre_preds)
        post_probs = logits_to_probs(post_preds)

    # Write to disk for scoring
    save_path = "test_results/" + config_name + "/"
    visual_results_path = save_path + "viz/"
    leaderboard_results_path = save_path + "leaderboard_targets/"
    pathlib.Path(visual_results_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(leaderboard_results_path).mkdir(parents=True, exist_ok=True)

    num_id = str(idx)
    num_id = "".join(["0"] * (5 - len(num_id)) + list(num_id))

    # Save results for leaderboard

    r = postprocess_segmap_tensor_to_pil_img(pre_probs, apply_color=False, binarize=True, threshold=THRESHOLD)
    r.save(leaderboard_results_path + "test_localization_" + num_id + "_prediction.png")
    r = postprocess_combined_predictions(pre_probs, post_probs, apply_color=False, threshold=THRESHOLD)
    r.save(leaderboard_results_path + "test_damage_" + num_id + "_prediction.png")

    # Save visual results for examination
    r = postprocess_segmap_tensor_to_pil_img(pre_probs, binarize=True, threshold=THRESHOLD)
    r.save(visual_results_path + num_id + "_pre_pred.png")
    r = postprocess_segmap_tensor_to_pil_img(post_probs)
    r.save(visual_results_path + num_id + "_post_pred.png", threshold=THRESHOLD)
    r = postprocess_combined_predictions(pre_probs, post_probs, threshold=THRESHOLD)
    r.save(visual_results_path + num_id + "_combo_pred.png")

    # Save input images as well for visual reference
    pre_img = reconstruct_from_tiles(pretiles[0], 3, INFERENCE_SIZE)
    post_img = reconstruct_from_tiles(posttiles[0], 3, INFERENCE_SIZE)

    pilimg = input_tensor_to_pil_img(pre_img)
    pilimg.save(visual_results_path + num_id + "_pre_image.png")
    pilimg = input_tensor_to_pil_img(post_img)
    pilimg.save(visual_results_path + num_id + "_post_image.png")

    test_pbar.update(1)

test_pbar.close()

