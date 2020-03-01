#!/usr/bin/env python3
"""
Miscellaneous utility functions
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None

import inspect
from collections import OrderedDict

# from skimage.io import imsave
import torch
# import orjson as json

import numpy as np
from PIL import Image
# import scipy.ndimage
# import matplotlib.pyplot as plt


# import glob
# import random



def open_image_as_nparray(img_path, dtype):
    return np.array(Image.open(img_path), dtype=dtype)


def EMSG(errtype="ERROR"):
    if errtype in ["E", "ERR", "ERROR"]:
        errtype = "[ERROR] "
    elif errtype in ["W", "WARN", "WARNING"]:
        errtype = "[WARN] "
    elif errtype in ["I", "INFO", "INFORMATION"]:
        errtype = "[INFO] "

    previous_frame = inspect.currentframe().f_back
    _, _, fname, _, _ = inspect.getframeinfo(previous_frame)

    return errtype + " in function " + fname + " :"


def clean_distributed_state_dict(distributed_state_dict):
    new_state_dict = OrderedDict()
    for k, v in distributed_state_dict.items():
        name = k.replace("module.", "")  # remove `module.`
        new_state_dict[name] = v

    return new_state_dict


def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.reduce_op.SUM)
    rt /= world_size
    return rt


def main():
    """ Dummy main module for Library """
    pass
    # xview_root = "/home/rohitg/data/xview/"
    # train_data, test_data = load_xview_metadata(xview_root)

    # # random_key = random.sample(list(train_data), 1)[0]
    # # print(random_key)

    # random_key = "hurricane-michael_00000083"
    # random_key = "palu-tsunami_00000097"
    # # random_file = train_data[random_key]["post_label_file"]

    # # Pre Disaster Image
    # random_file = train_data[random_key]["pre_label_file"]
    # labels_data = read_labels_file(random_file)
    # image_file = train_data[random_key]["pre_image_file"]
    # segmap = labels_to_segmentation_map(labels_data)
    # bboxes, labels = labels_to_bboxes(labels_data)
    # save_segmentation_map(segmap, "pre_map_seg.png")
    # save_bboxes(image_file, bboxes, labels, "pre_map_boxes.png")
    # im = Image.open(image_file)
    # im.save("pre_map_orig.jpg")

    # # Post Disaster Image
    # random_file = train_data[random_key]["post_label_file"]
    # labels_data = read_labels_file(random_file)
    # image_file = train_data[random_key]["post_image_file"]
    # segmap = labels_to_segmentation_map(labels_data)
    # bboxes, labels = labels_to_bboxes(labels_data)
    # save_segmentation_map(segmap, "post_map_seg.png")
    # save_bboxes(image_file, bboxes, labels, "post_map_boxes.png")
    # im = Image.open(train_data[random_key]["post_image_file"])
    # im.save("post_map_orig.jpg")

    # print("Saving sample", random_key, "from xview2 dataset.")


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
