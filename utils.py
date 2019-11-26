#!/usr/bin/env python3
"""
Utility functions
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None


import shapely
import shapely.wkt
from skimage.draw import polygon, polygon_perimeter
# from skimage.io import imsave
import torch
import orjson as json

import numpy as np
import scipy.ndimage
# import matplotlib.pyplot as plt
from PIL import Image

import glob
import random
import inspect

colors = [[  0,   0, 200], # Blue:   Background
          [  0, 200,   0], # Green:  No Damage
          [250, 125,   0], # Orange: Minor Damage
          [250,  25, 150], # Pink:   Major Damage
          [250,   0,   0]] # Red:    Destroyed
NUM_CLASSES = len(colors)
colors = np.array(colors).astype(np.uint8)



H, W = 1024, 1024
ACTUAL_SIZE = 1024


def convert_color_segmap_to_int(segmap):
    segmap_rg = segmap[:, :, 0] + segmap[:, :, 1]
    segmap_class = np.zeros((H, W))
    for i in range(NUM_CLASSES):
        segmap_class[segmap_rg == colors[i][0] + colors[i][1]] = i

    return segmap_class

def open_image_as_nparray(img_path, dtype):
    return np.array(Image.open(img_path), dtype=dtype)

def reconstruct_from_tiles(tiles, CHANNELS, CROP_SIZE):
    num_tiles = ACTUAL_SIZE // CROP_SIZE

    reconstructed = torch.zeros([CHANNELS, ACTUAL_SIZE, ACTUAL_SIZE], dtype=torch.float32)
    for x in range(num_tiles):
        for y in range(num_tiles):
            x0, x1 = x * CROP_SIZE, (x + 1) * CROP_SIZE
            y0, y1 = y * CROP_SIZE, (y + 1) * CROP_SIZE
            reconstructed[:, x0:x1, y0:y1] = tiles[x * num_tiles + y]

    return reconstructed


def input_tensor_to_pil_img(inp_tensor):
    # μ and σ for xview dataset
    MEAN = [0.309, 0.340, 0.255]
    STDDEV = [0.162, 0.144, 0.135]

    r = (255 * ((inp_tensor[0] * STDDEV[0]) + MEAN[0])).round().byte().cpu().numpy()
    g = (255 * ((inp_tensor[1] * STDDEV[1]) + MEAN[1])).round().byte().cpu().numpy()
    b = (255 * ((inp_tensor[2] * STDDEV[2]) + MEAN[2])).round().byte().cpu().numpy()

    img = np.stack([r, g, b], axis=-1)

    return Image.fromarray(img)


def segmap_tensor_to_pil_img(segmap_tensor):
    r = Image.fromarray(torch.max(segmap_tensor, 0).indices.byte().cpu().numpy())
    r.putpalette(colors)

    return r


def logits_to_probs(segmap_logits):
    bg_logits, class_logits = torch.split(segmap_logits, [1, 4], dim=0)
    bg_probs = torch.nn.functional.sigmoid(bg_logits)
    class_probs = torch.nn.functional.softmax(class_logits, dim=0)
    return torch.cat([bg_probs, class_probs], dim=0)


def postprocess_segmap_tensor_to_pil_img(segmap_tensor, apply_color=True, binarize=False, threshold=0.5):
    segmap = segmap_tensor.cpu().numpy()
    background = segmap[0, :, :] > threshold
    foreground = ~background
    damage_class = np.argmax(segmap[1:, :, :], axis=0)
    damage_class += 1
    processed_segmap = foreground * damage_class
    if binarize:
        processed_segmap[processed_segmap > 1] = 1
    r = Image.fromarray(processed_segmap.astype(np.uint8))
    if apply_color:
        r.putpalette(colors)
    
    return r


def EMSG(errtype="ERROR"):
    if errtype in ["E", "ERR", "ERROR"]:
        errtype = "[ERROR] "
    elif errtype in ["W", "WARN", "WARNING"]:
        errtype = "[WARN] "
    elif errtype in ["I", "INFO", "INFORMATION"]:
        errtype = "[INFO] "

    previous_frame = inspect.currentframe().f_back
    _, _, fname, _, _ = inspect.getframeinfo(previous_frame)

    return errtype + " in function " + fname 


def read_labels_file(labels_file):
    with open(labels_file, "r") as f:
        contents = f.read()
    data = json.loads(contents)

    return data['features']['xy']


def get_object_label(object_properties):
    if 'feature_type' in object_properties.keys():
        objclass = object_properties['feature_type']
    else:
        objclass = "building"

    if 'subtype' in object_properties.keys():
        objstate = object_properties['subtype']
    else:
        objstate = "no-damage"

    return objclass, objstate


def labels_to_segmentation_map(labels_data, scale=None):
    """
    Parse a label file into a segmentation matrix (pytorch format)
    """
    if scale:
        scaled_H = int(H * scale)
        scaled_W = int(W * scale)
    else:
        scaled_H = H
        scaled_W = W
    segmap = np.zeros((5, scaled_H, scaled_W), dtype=np.uint8)
    for annotated_object in labels_data:
        shp = shapely.wkt.loads(annotated_object['wkt'])
        pts = list(shp.exterior.coords)
        if scale:
            pts = [(int(round(scale*x[0])), int(round(scale*x[1]))) for x in pts]
        else:
            pts = [(int(round(x[0])), int(round(x[1]))) for x in pts]

        # Draw Polygon
        poly = np.array(pts)
        # rr, cc = polygon(poly[:, 0], poly[:, 1], segmap.shape[1:])
        cc, rr = polygon(poly[:, 0], poly[:, 1], segmap.shape[1:])

        objclass, objstate = get_object_label(annotated_object["properties"])

        if objclass == "building":
            if objstate == "destroyed":
                segmap[4, rr, cc] = 1
            elif "major" in objstate:
                segmap[3, rr, cc] = 1
            elif "minor" in objstate:
                segmap[2, rr, cc] = 1
            else:
                segmap[1, rr, cc] = 1
        else:
            segmap[0, rr, cc] = 1
    segmap[0, :, :] += np.sum(segmap, axis=0) == 0

    # assert np.sum(np.sum(segmap, axis=0)) == H * W, EMSG("Labels != Pixels")

    return segmap


def labels_to_bboxes(labels_data):
    """
    Parse a label file into a segmentation matrix (pytorch format)
    """
    bboxes = []
    labels = []
    for annotated_object in labels_data:
        shp = shapely.wkt.loads(annotated_object['wkt'])
        pts = list(shp.exterior.coords)
        pts = [(int(round(x[0])), int(round(x[1]))) for x in pts]

        # Draw Polygon
        poly = np.array(pts)
        # rr, cc = polygon(poly[:, 0], poly[:, 1], (1024, 1024))
        cc, rr = polygon(poly[:, 0], poly[:, 1], (H, W))

        objclass, objstate = get_object_label(annotated_object["properties"])

        if objclass == "building":
            bboxes.append((min(cc), min(rr), max(cc), max(rr)))
            if objstate == "destroyed":
                labels.append(4)
            elif "major" in objstate:
                labels.append(3)
            elif "minor" in objstate:
                labels.append(2)
            else:
                labels.append(1)
        else:
            labels.append(0)

    return bboxes, labels


def save_segmentation_map(segmap, filename):

    # get class ID from predictions array
    segmap = np.argmax(segmap, axis=0).astype("uint8")

    # plot the semantic segmentation predictions of 5 classes in each color
    # r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
    r = Image.fromarray(segmap)
    r.putpalette(colors)

    r.save(filename)


def bbox_to_poly(bbox):
    x0, y0, x1, y1 = bbox

    r = []
    c = []
    c.append(x0); r.append(y0)
    c.append(x1); r.append(y0)
    c.append(x1); r.append(y1)
    c.append(x0); r.append(y1)
    c.append(x0); r.append(y0)
    c = np.array(c)
    r = np.array(r)

    return r, c


def save_bboxes(image_file, bboxes, labels, filename):
    img = np.array(Image.open(image_file), dtype=np.uint8)
    # img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    for idx, bbox in enumerate(bboxes):

        label = labels[idx]
        r, c = bbox_to_poly(bbox)

        # Draw Bbox
        rr, cc = polygon_perimeter(r, c, shape=img.shape)

        img[rr, cc, :] = colors[label]

    pil_image = Image.fromarray(img)
    pil_image.save(filename)


def smoothing_filter_factory(n):
    return (1.0 / n * n) * np.ones((n, n))


def spatial_label_smoothing(segmap, n):

    # Only allow odd kernel sizes
    assert n % 2 == 1, "Kernel size for smoothing must be odd"
    kernel = smoothing_filter_factory(n)

    for channel in range(segmap.shape[0]):
        segmap[channel] = scipy.ndimage.convolve(segmap[channel],
                                                 kernel, mode='nearest')

    # L1 Normalize, sum of probablities across classes = 1.0
    sums = np.sum(segmap, axis=0)
    norm_segmap = segmap / sums

    return norm_segmap


def load_xview_metadata(xview_root, data_version, use_tier3):
    """
    Load data file paths
    """
    data_dir = xview_root + data_version
    train_label_files = glob.glob(data_dir + "train/labels/*")
    train_image_files = glob.glob(data_dir + "train/images/*")

    if use_tier3:
        tier3_label_files = glob.glob(data_dir + "tier3/labels/*")
        tier3_image_files = glob.glob(data_dir + "tier3/images/*")
        train_label_files += tier3_label_files
        train_image_files += tier3_image_files
    test_image_files = glob.glob(data_dir + "test/images/*")
    trainval_data = {}
    for file_name in train_label_files:
        disaster, image_num, pre_or_post, _ = file_name.split("_")
        # disaster = disaster.replace(data_dir + "train/labels/", "")
        disaster = disaster.split("/")[-1]
        input_id = disaster + "_" + image_num
        if input_id not in trainval_data.keys():
            trainval_data[input_id] = {}
        trainval_data[input_id][pre_or_post + "_label_file"] = file_name

    for file_name in train_image_files:
        disaster, image_num, pre_or_post, _ = file_name.split("_")
        # disaster = disaster.replace(data_dir + "train/images/", "")
        disaster = disaster.split("/")[-1]
        input_id = disaster + "_" + image_num
        if input_id not in trainval_data.keys():
            trainval_data[input_id] = {}
        trainval_data[input_id][pre_or_post + "_image_file"] = file_name

    test_data = {}
    for file_name in test_image_files:
        _, pre_or_post, input_id = file_name.split("_")
        if input_id not in test_data.keys():
            test_data[input_id] = {}
        test_data[input_id][pre_or_post + "_image_file"] = file_name

    # Carry out train/val split
    with open(data_dir + "val-split.txt", "r") as f:
        val_keys = [x.strip() for x in f.readlines()]
    train_keys = list(set(list(trainval_data.keys())) - set(val_keys))
    val_keys = sorted(val_keys)
    train_keys = sorted(train_keys)

    train_data = {key: val for key, val in trainval_data.items() if key in train_keys}
    val_data = {key: val for key, val in trainval_data.items() if key in val_keys}

    return train_data, val_data, test_data

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.reduce_op.SUM)
    rt /= world_size
    return rt

def main():
    """ Dummy main module for Library """
    xview_root = "/home/rohitg/data/xview/"
    train_data, test_data = load_xview_metadata(xview_root)

    # random_key = random.sample(list(train_data), 1)[0]
    # print(random_key)

    random_key = "hurricane-michael_00000083"
    random_key = "palu-tsunami_00000097"
    # random_file = train_data[random_key]["post_label_file"]

    # Pre Disaster Image
    random_file = train_data[random_key]["pre_label_file"]
    labels_data = read_labels_file(random_file)
    image_file = train_data[random_key]["pre_image_file"]
    segmap = labels_to_segmentation_map(labels_data)
    bboxes, labels = labels_to_bboxes(labels_data)
    save_segmentation_map(segmap, "pre_map_seg.png")
    save_bboxes(image_file, bboxes, labels, "pre_map_boxes.png")
    im = Image.open(image_file)
    im.save("pre_map_orig.jpg")

    # Post Disaster Image
    random_file = train_data[random_key]["post_label_file"]
    labels_data = read_labels_file(random_file)
    image_file = train_data[random_key]["post_image_file"]
    segmap = labels_to_segmentation_map(labels_data)
    bboxes, labels = labels_to_bboxes(labels_data)
    save_segmentation_map(segmap, "post_map_seg.png")
    save_bboxes(image_file, bboxes, labels, "post_map_boxes.png")
    im = Image.open(train_data[random_key]["post_image_file"])
    im.save("post_map_orig.jpg")

    print("Saving sample", random_key, "from xview2 dataset.")


if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()
