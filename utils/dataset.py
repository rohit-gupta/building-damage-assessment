#!/usr/bin/env python3
"""
Utility functions specific to xview dataset
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None


import glob
import orjson as json
import scipy.ndimage
import numpy as np
import shapely
import shapely.wkt
from skimage.draw import polygon, polygon_perimeter

H, W = 1024, 1024

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


