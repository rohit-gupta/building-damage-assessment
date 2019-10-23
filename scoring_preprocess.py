

# from utils import convert_color_segmap_to_int
# from utils import open_image_as_nparray
import pathlib
from PIL import Image
import numpy as np


def open_image_as_nparray(img_path, dtype):
    return np.array(Image.open(img_path), dtype=dtype)

EPOCH = 69

prefix = "samples/200/"
pre_pred_path = prefix + "pre_pred_epoch_" + str(EPOCH)
post_pred_path = prefix + "post_pred_epoch_" + str(EPOCH)
pre_gt_path = prefix + "pre_gt"
post_gt_path = prefix + "post_gt"

pre_gt = open_image_as_nparray(pre_gt_path + ".png", dtype=np.int64)
post_gt = open_image_as_nparray(post_gt_path + ".png", dtype=np.int64)
pre_pred = open_image_as_nparray(pre_pred_path + ".png", dtype=np.int64)
post_pred = open_image_as_nparray(post_pred_path + ".png", dtype=np.int64)

# pre_gt_class = convert_color_segmap_to_int(pre_gt)
# pre_gt_class[pre_gt_class > 1] = 1
# post_gt_class = convert_color_segmap_to_int(post_gt)
# pre_pred_class = convert_color_segmap_to_int(pre_pred)
# pre_pred_class[pre_pred_class > 1] = 1
# post_pred_class = convert_color_segmap_to_int(post_pred)

pre_gt[pre_gt > 1] = 1
pre_pred[pre_pred > 1] = 1


# for i in range(H):
#     for j in range(W):
#         for k in range(colors.size[]):
#         np.array_equal(pre_gt[i,j,:],


# Hack, we know all colors have unique R+G values


pathlib.Path("scoring/targets/").mkdir(parents=True, exist_ok=True)
pathlib.Path("scoring/predictions/").mkdir(parents=True, exist_ok=True)

print(np.max(pre_gt)), print(np.min(pre_gt))
print(np.max(post_gt)), print(np.min(post_gt))

print(np.max(pre_pred)), print(np.min(pre_pred))
print(np.max(post_pred)), print(np.min(post_pred))

Image.fromarray(pre_gt).save("scoring/targets/test_localization_00000_target.png")
Image.fromarray(post_gt).save("scoring/targets/test_damage_00000_target.png")
Image.fromarray(pre_pred).save("scoring/predictions/test_localization_00000_prediction.png")
Image.fromarray(post_pred).save("scoring/predictions/test_damage_00000_prediction.png")






