
import pathlib
from PIL import Image
import numpy as np
from glob import glob
import os

colors = [[  0,   0, 200],  # Blue:   Background
          [  0, 200,   0],  # Green:  No Damage
          [250, 125,   0],  # Orange: Minor Damage
          [250,  25, 150],  # Pink:   Major Damage
          [250,   0,   0]]  # Red:    Destroyed
NUM_CLASSES = len(colors)
colors = np.array(colors).astype(np.uint8)


def open_image_as_nparray(img_path, dtype):
    return np.array(Image.open(img_path), dtype=dtype)


XVIEW_CONFIG = os.environ["XVIEW_CONFIG"] + "/"
EPOCHS = int(os.environ["XVIEW_EPOCHS"])
print(XVIEW_CONFIG, EPOCHS)
prefix = "val_results/" + XVIEW_CONFIG
samples = glob(prefix + "*/")

output_dir = "val_scoring/" + XVIEW_CONFIG
# Create Required Directories
targets_dir = output_dir + "targets/"
pathlib.Path(targets_dir).mkdir(parents=True, exist_ok=True)
for epoch in range(EPOCHS):
    predictions_dir = output_dir + str(epoch) + "/predictions/"
    predictions_color_dir = output_dir + str(epoch) + "/predictions_color/"
    pathlib.Path(predictions_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(predictions_color_dir).mkdir(parents=True, exist_ok=True)

for idx, sample in enumerate(samples):

    # Ground truth is same for all epochs
    pre_gt_path = sample + "pre_gt"
    post_gt_path = sample + "post_gt"
    pre_gt = open_image_as_nparray(pre_gt_path + ".png", dtype=np.uint8)
    post_gt = open_image_as_nparray(post_gt_path + ".png", dtype=np.uint8)

    num_id = str(idx)
    num_id = "".join(["0"] * (5 - len(num_id)) + list(num_id))

    # Localization Binarization
    pre_gt[pre_gt > 1] = 1

    # Write groundtruth
    Image.fromarray(pre_gt).save(targets_dir + "test_localization_" + num_id + "_target.png")
    Image.fromarray(post_gt).save(targets_dir + "test_damage_" + num_id + "_target.png")

    for epoch in range(EPOCHS):
        # Read Results from DeepLab
        pre_pred_path = sample + "pre_pred_epoch_" + str(epoch)
        post_pred_path = sample + "post_pred_epoch_" + str(epoch)
        combined_pred_path = sample + "post_pred_epoch_" + str(epoch)
        if os.path.isfile(combined_pred_path + ".png"):
            post_pred_path = combined_pred_path

        pre_pred = open_image_as_nparray(pre_pred_path + ".png", dtype=np.uint8)
        post_pred = open_image_as_nparray(post_pred_path + ".png", dtype=np.uint8)

        # Output dirs
        predictions_dir = output_dir + str(epoch) + "/predictions/"
        predictions_color_dir = output_dir + str(epoch) + "/predictions_color/"

        # Localization Binarization
        pre_pred[pre_pred > 1] = 1
        post_pred = post_pred * pre_pred

        # Write predictions output
        r = Image.fromarray(pre_pred)
        r.putpalette(colors)
        r.save(predictions_color_dir + str(idx) + "_localization.png")
        r = Image.fromarray(post_pred)
        r.putpalette(colors)
        r.save(predictions_color_dir + str(idx) + "_classification.png")

        Image.fromarray(pre_pred).save(predictions_dir + "test_localization_" + num_id + "_prediction.png")
        Image.fromarray(post_pred).save(predictions_dir + "test_damage_" + num_id + "_prediction.png")
