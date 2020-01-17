#!/usr/bin/env python3
"""
Testing script for xview challenge submission
"""

__author__ = "Rohit Gupta"
__version__ = "0.1beta1"
__license__ = None

from PIL import Image
import sys

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
import torch

from inference_utils import input_tensor_to_pil_img
from inference_utils import logits_to_probs
from inference_utils import postprocess_segmap_tensor_to_pil_img, postprocess_combined_predictions
from inference_utils import clean_distributed_state_dict

semseg_model = deeplabv3_resnet50(pretrained=False,
                                  num_classes=5,
                                  aux_loss=None)
device = "cuda:0"
semseg_model = semseg_model.to(device)

# Inference Options
INFERENCE_SIZE = 1024
THRESHOLD = 0.6
BEST_MODEL_PATH = "/home/rohitg/xview_challenge/models/baseline/turing_finetune_tier1/10.pth"

# μ and σ for xview dataset
MEANS = [0.309, 0.340, 0.255]
STDDEVS = [0.162, 0.144, 0.135]
pil_to_tensor = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(MEANS, STDDEVS)
                                    ])

# Load model
state_dict = torch.load(BEST_MODEL_PATH, map_location=torch.device(device)) 
state_dict = clean_distributed_state_dict(state_dict)
semseg_model.load_state_dict(state_dict)
semseg_model.eval()

# I/O paths
pre_img, post_img = sys.argv[1], sys.argv[2]
loc_out, cls_out = sys.argv[3], sys.argv[4]

# Image Tensors
pre_img = pil_to_tensor(Image.open(pre_img)).to(device)
post_img = pil_to_tensor(Image.open(post_img)).to(device)

img_stacked = torch.stack([pre_img, post_img], dim=0)

with torch.set_grad_enabled(False):
    outputs = semseg_model(img_stacked)
    preds = outputs['out']
    pre_preds = preds[0]
    post_preds = preds[1]
    pre_probs = logits_to_probs(pre_preds)
    post_probs = logits_to_probs(post_preds)

# Write to disk for scoring
# Save results for leaderboard

r = postprocess_segmap_tensor_to_pil_img(pre_probs, apply_color=False, binarize=True, threshold=THRESHOLD)
r.save(loc_out)
r = postprocess_combined_predictions(pre_probs, post_probs, apply_color=False, threshold=THRESHOLD)
r.save(cls_out)

# Save visual results for examination
r = postprocess_segmap_tensor_to_pil_img(pre_probs, binarize=True, threshold=THRESHOLD)
r.save("pre_pred.png")
r = postprocess_segmap_tensor_to_pil_img(post_probs)
r.save("post_pred.png", threshold=THRESHOLD)
r = postprocess_combined_predictions(pre_probs, post_probs, threshold=THRESHOLD)
r.save("combo_pred.png")

