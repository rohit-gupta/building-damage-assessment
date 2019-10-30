#!/usr/bin/env python3
"""
Dataloaders for xview challenge
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None


import random
from PIL import Image
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from utils import load_xview_metadata
from utils import read_labels_file, labels_to_segmentation_map
from utils import spatial_label_smoothing

# μ and σ for xview dataset
MEANS = [0.309, 0.340, 0.255]
STDDEVS = [0.162, 0.144, 0.135]
pil_to_tensor = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(MEANS, STDDEVS)
                                    ])


def get_rand_crops(actual_size, crop_size):
    x0 = random.randint(0, actual_size - crop_size)
    y0 = random.randint(0, actual_size - crop_size)
    x1, y1 = x0 + crop_size, y0 + crop_size

    return x0, y0, x1, y1


def get_rand_flips():
    flip_options = [None, 'rot90', 'rot180', 'rot270', 'hflip', 'vflip']
    return random.sample(flip_options, 1)[0]


class xviewDataset(Dataset):
    '''
    mode: should support train (aug + labels), val (noaug + labels), test(noaug + nolabels)
    flips: h, v, combo
    noise: salt, gaussian, other
    erase: random erasing to mean
    distort: small elastic distortion
    '''
    def __init__(self, data, mode='train', actual_size=1024, crop_size=384,
                 flips=True, erase=False, noise=False, distort=False,
                 spatial_label_smoothing=1):
        self.data = data
        self.mode = mode
        self.actual_size = actual_size
        self.crop_size = crop_size
        self.flips = flips
        self.erase = erase
        self.noise = noise
        self.distort = distort
        self.ls_size = spatial_label_smoothing  # 1 = No Smoothing

        data_keys = list(data.keys())
        if mode == "train":
            self.index = random.sample(data_keys, len(data_keys))
        else:
            self.index = sorted(data_keys)

    def __getitem__(self, index):

        x = self.data[self.index[index]]
        pre_image = self.__readimg(x["pre_image_file"])
        post_image = self.__readimg(x["post_image_file"])

        if self.mode == "train" or self.mode == "val":
            pre_segmap = self.__readlabels(x["pre_label_file"])
            post_segmap = self.__readlabels(x["post_label_file"])

            # Label smoothing with 1x1 Kernel = no smoothing
            # if self.ls_size > 1:
            # "if" was removed because smoothing function corrects multi
            pre_segmap = spatial_label_smoothing(pre_segmap.astype(np.float64),
                                                 self.ls_size)
            post_segmap = spatial_label_smoothing(post_segmap.astype(np.float64),
                                                  self.ls_size)

            pre_segmap = torch.from_numpy(pre_segmap).to(torch.float32)
            post_segmap = torch.from_numpy(post_segmap).to(torch.float32)

        if self.mode == "train":
            img1, segmap1, img2, segmap2 = self.__augment(pre_image,
                                                          pre_segmap,
                                                          post_image,
                                                          post_segmap)
            return ([img1, img2], [segmap1, segmap2])

        elif self.mode == "val":
            preims, presegs, postims, postsegs = self.__augment(pre_image,
                                                                pre_segmap,
                                                                post_image,
                                                                post_segmap)

            return (preims, postims, presegs, postsegs)

        elif self.mode == "test":
            preimgs, _, postimgs, _ = self.__augment(pre_image,
                                                     None,
                                                     post_image,
                                                     None)
            return (preimgs, postimgs)

    def __readlabels(self, label_file):

        labels_data = read_labels_file(label_file)
        segmap = labels_to_segmentation_map(labels_data)

        return segmap

    def __readimg(self, image_file):

        img = Image.open(image_file)
        img_tensor = pil_to_tensor(img)

        return img_tensor

    def __augment(self, pre_img, pre_segmap, post_img, post_segmap):

        # No Test Time Augmentation (for now)
        # In Val and Test, we need all tiles of the image
        # During Training, we need 1 random crop that's further flipped/rotated
        if self.mode == "val" or self.mode == "test":
            assert self.actual_size % self.crop_size == 0, "bad tile size"

            num_tiles = self.actual_size // self.crop_size

            pre_img_crops = []
            post_img_crops = []

            if self.mode == "val":
                pre_seg_crops = []
                post_seg_crops = []

            for x in range(num_tiles):
                for y in range(num_tiles):

                    x0 = x * self.crop_size
                    x1 = (x + 1) * self.crop_size
                    y0 = y * self.crop_size
                    y1 = (y + 1) * self.crop_size

                    pre_img_crops.append(pre_img[:, x0: x1, y0: y1])
                    post_img_crops.append(post_img[:, x0: x1, y0: y1])

                    if self.mode == "val":
                        pre_seg_crops.append(pre_segmap[:, x0: x1, y0: y1])
                        post_seg_crops.append(post_segmap[:, x0: x1, y0: y1])

            if self.mode == "val":
                return (pre_img_crops, pre_seg_crops,
                        post_img_crops, post_seg_crops)
            elif self.mode == "test":
                return (pre_img_crops, post_img_crops)

        x0, y0, x1, y1 = get_rand_crops(self.actual_size, self.crop_size)

        if self.mode == "train":
            # Tensors are NCHW, x is column number in numpy notation
            cropped_pre_img = pre_img[:, x0:x1, y0:y1]
            cropped_post_img = post_img[:, x0:x1, y0:y1]
            cropped_pre_segmap = pre_segmap[:, x0:x1, y0:y1]
            cropped_post_segmap = post_segmap[:, x0:x1, y0:y1]

        if self.flips is False:
            return (cropped_pre_img, cropped_pre_segmap,
                    cropped_post_img, cropped_post_segmap)

        if self.mode == "train" and self.flips is not False:
            # There are 6 possible flips: None, rot90, rot180, rot270, hflip & vflip
            flip = get_rand_flips()
            if flip is None:
                flipped_pre_img = cropped_pre_img
                flipped_post_img = cropped_post_img
                flipped_pre_segmap = cropped_pre_segmap
                flipped_post_segmap = cropped_post_segmap
            elif "rot" in flip:
                angle = int(int(flip.replace("rot", "")) // 90)
                flipped_pre_img = torch.rot90(cropped_pre_img, angle, [1, 2])
                flipped_post_img = torch.rot90(cropped_post_img, angle, [1, 2])
                flipped_pre_segmap = torch.rot90(cropped_pre_segmap, angle, [1, 2])
                flipped_post_segmap = torch.rot90(cropped_post_segmap, angle, [1, 2])
            elif "flip" in flip:
                if flip == "hflip":
                    flipped_pre_img = torch.flip(cropped_pre_img, [2])
                    flipped_post_img = torch.flip(cropped_post_img, [2])
                    flipped_pre_segmap = torch.flip(cropped_pre_segmap, [2])
                    flipped_post_segmap = torch.flip(cropped_post_segmap, [2])
                elif flip == "vflip":
                    flipped_pre_img = torch.flip(cropped_pre_img, [1])
                    flipped_post_img = torch.flip(cropped_post_img, [1])
                    flipped_pre_segmap = torch.flip(cropped_pre_segmap, [1])
                    flipped_post_segmap = torch.flip(cropped_post_segmap, [1])

            return (flipped_pre_img, flipped_pre_segmap,
                    flipped_post_img, flipped_post_segmap)

    def __len__(self):
        return len(self.index)


def train_collate_fn(batch):
    images = [x[0] for x in batch]
    labels = [x[1] for x in batch]

    flat_images = [item.type(torch.float32) for sublist in images for item in sublist]
    flat_labels = [item.type(torch.float32) for sublist in labels for item in sublist]

    # print("Num images in batch", len(flat_images))
    # print(flat_images[0].size)

    images_batch_tensor = torch.stack(flat_images)
    labels_batch_tensor = torch.stack(flat_labels)

    return (images_batch_tensor, labels_batch_tensor)


def val_collate_fn(batch):
    pretiles = [x[0] for x in batch]
    posttiles = [x[1] for x in batch]
    prelabels = [x[2] for x in batch]
    postlabels = [x[3] for x in batch]

    pretiles_instance_tensors = [torch.stack(x) for x in pretiles]
    posttiles_instance_tensors = [torch.stack(x) for x in posttiles]

    prelabels_instance_tensors = [torch.stack(x) for x in prelabels]
    postlabels_instance_tensors = [torch.stack(x) for x in postlabels]

    return (pretiles_instance_tensors, posttiles_instance_tensors,
            prelabels_instance_tensors, postlabels_instance_tensors)


def test_collate_fn(batch):
    pretiles = [x[0] for x in batch]
    posttiles = [x[1] for x in batch]

    pretiles_instance_tensors = [torch.stack(x) for x in pretiles]
    posttiles_instance_tensors = [torch.stack(x) for x in posttiles]

    return (pretiles_instance_tensors, posttiles_instance_tensors)


def xview_train_loader_factory(xview_root, crop_size, batch_size, num_workers):
    # print(xview_root)
    # Read metadata
    trainval_data, test_data = load_xview_metadata(xview_root)

    # print("TrainVal images:", len(trainval_data))
    # print("Test images:", len(test_data))

    with open("train-split.txt", "r") as f:
        train_keys = [x.strip() for x in f.readlines()]
    with open("val-split.txt", "r") as f:
        val_keys = [x.strip() for x in f.readlines()]

    # train_keys = train_keys[:32]
    # val_keys = val_keys[:16]

    val_keys = sorted(val_keys)
    train_keys = sorted(train_keys)

    train_data = {key:val for key, val in trainval_data.items() if key in train_keys}
    val_data = {key: val for key, val in trainval_data.items() if key in val_keys}

    # print("Validation images:", len(val_data))
    # print("Train images:", len(train_data))

    trainset = xviewDataset(train_data, mode="train",
                            actual_size=1024, crop_size=crop_size,
                            flips=True,
                            erase=False, noise=False, distort=False)

    valset = xviewDataset(val_data, mode="val",
                          actual_size=1024, crop_size=crop_size)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                            collate_fn=train_collate_fn, num_workers=num_workers,
                            pin_memory=True)

    valloader = DataLoader(valset, batch_size=1, shuffle=False,
                           collate_fn=val_collate_fn, num_workers=1,
                           pin_memory=True)

    return trainloader, valloader
