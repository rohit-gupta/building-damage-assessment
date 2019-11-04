
# from utils import convert_color_segmap_to_int
# from utils import open_image_as_nparray
import pathlib
from PIL import Image
import numpy as np
from glob import glob

colors = [[  0,   0, 200],  # Blue:   Background
          [  0, 200,   0],  # Green:  No Damage
          [250, 125,   0],  # Orange: Minor Damage
          [250,  25, 150],  # Pink:   Major Damage
          [250,   0,   0]]  # Red:    Destroyed
NUM_CLASSES = len(colors)
colors = np.array(colors).astype(np.uint8)


def open_image_as_nparray(img_path, dtype):
    return np.array(Image.open(img_path), dtype=dtype)


# prefix = "samples/"
prefix = "val_results/default_lb/"
samples = glob(prefix + "*/")

output_dir = "val_scoring/default_lb/"
# Create Required Directories
targets_dir = output_dir + "targets/"
pathlib.Path(targets_dir).mkdir(parents=True, exist_ok=True)
for epoch in range(200):
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

    for epoch in range(200):
        # Read Results from DeepLab
        pre_pred_path = sample + "pre_pred_epoch_" + str(epoch)
        post_pred_path = sample + "post_pred_epoch_" + str(epoch)
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
        r.save(predictions_color_dir + str(idx) + "localization.png")
        r = Image.fromarray(post_pred)
        r.putpalette(colors)
        r.save(predictions_color_dir + str(idx) + "classification.png")

        Image.fromarray(pre_pred).save(predictions_dir + "test_localization_" + num_id + "_prediction.png")
        Image.fromarray(post_pred).save(predictions_dir + "test_damage_" + num_id + "_prediction.png")

# print(np.max(pre_gt)), print(np.min(pre_gt))
# print(np.max(post_gt)), print(np.min(post_gt))
#
# print(np.max(pre_pred)), print(np.min(pre_pred))
# print(np.max(post_pred)), print(np.min(post_pred))

# pre_gt_class = convert_color_segmap_to_int(pre_gt)
# pre_gt_class[pre_gt_class > 1] = 1
# post_gt_class = convert_color_segmap_to_int(post_gt)
# pre_pred_class = convert_color_segmap_to_int(pre_pred)
# pre_pred_class[pre_pred_class > 1] = 1
# post_pred_class = convert_color_segmap_to_int(post_pred)


# for i in range(H):
#     for j in range(W):
#         for k in range(colors.size[]):
#         np.array_equal(pre_gt[i,j,:],


# Hack, we know all colors have unique R+G values
