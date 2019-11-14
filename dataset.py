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

pixel_space_augmentations = transforms.Compose(
    [transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02)], p=0.2)]
)


def get_rand_crops(actual_size, crop_size):
    # print(actual_size, crop_size)
    if actual_size == crop_size:
        return 0, 0, actual_size, actual_size
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
    color jitter: brightness, contrast, hue, saturation
    noise: salt, gaussian, other
    erase: random erasing to mean
    distort: small elastic distortion
    '''

    def __init__(self, data, mode='tiles', load_segmaps=False, actual_size=1024, crop_size=256, tile_size=256,
                 flips=False, scale_jitter=False, color_jitter=False,
                 erase=False, noise=False, distort=False,
                 spatial_label_smoothing=1):
        self.data = data
        self.mode = mode
        self.load_segmaps = load_segmaps
        self.actual_size = actual_size
        self.crop_size = crop_size
        self.tile_size = tile_size
        self.flips = flips
        self.color_jitter = color_jitter
        self.scale_jitter = scale_jitter
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

        if self.scale_jitter:
            chosen_scale = random.choice([0.5, None, None, None, None, None, None, 2.0]) # Use 1x images 75% of the time
        else:
            chosen_scale = None
        pre_image = self.__readimg(x["pre_image_file"], chosen_scale)
        post_image = self.__readimg(x["post_image_file"], chosen_scale)

        if self.load_segmaps:
            pre_segmap = self.__readlabels(x["pre_label_file"], chosen_scale)
            post_segmap = self.__readlabels(x["post_label_file"], chosen_scale)

            # Label smoothing with 1x1 Kernel = no smoothing
            # if self.ls_size > 1:
            # "if" was removed because smoothing function corrects multi
            pre_segmap = spatial_label_smoothing(pre_segmap.astype(np.float64),
                                                 self.ls_size)
            post_segmap = spatial_label_smoothing(post_segmap.astype(np.float64),
                                                  self.ls_size)

            pre_segmap = torch.from_numpy(pre_segmap).to(torch.float32)
            post_segmap = torch.from_numpy(post_segmap).to(torch.float32)

        if self.mode == "batch" and self.load_segmaps:  # train-seg mode
            img1, segmap1, img2, segmap2 = self.__augment(pre_image,
                                                          pre_segmap,
                                                          post_image,
                                                          post_segmap)
            return ([img1, img2], [segmap1, segmap2])

        elif self.mode == "tiles" and self.load_segmaps:  # val or train-change mode
            preimgs, presegs, postimgs, postsegs = self.__augment(pre_image,
                                                                pre_segmap,
                                                                post_image,
                                                                post_segmap)

            return (preimgs, postimgs, presegs, postsegs)

        elif self.mode == "tiles" and not self.load_segmaps:  # test mode
            preimgs, postimgs = self.__augment(pre_image, None,
                                               post_image, None)
            return (preimgs, postimgs)

    def __readlabels(self, label_file, scale):

        labels_data = read_labels_file(label_file)
        segmap = labels_to_segmentation_map(labels_data, scale)

        return segmap

    def __readimg(self, image_file, scale):

        img = Image.open(image_file)
        if scale:
            img = transforms.Resize(int(self.actual_size * scale))(img)
        if self.color_jitter:
            img = pixel_space_augmentations(img)
        img_tensor = pil_to_tensor(img)

        return img_tensor

    def __augment(self, pre_img, pre_segmap, post_img, post_segmap):

        # Assuming everything is square
        x0, y0, x1, y1 = get_rand_crops(pre_img.shape[-1], self.crop_size)
        # Tensors are NCHW, x is column number in numpy notation
        cropped_pre_img = pre_img[:, x0:x1, y0:y1]
        cropped_post_img = post_img[:, x0:x1, y0:y1]

        if self.load_segmaps:
            cropped_pre_segmap = pre_segmap[:, x0:x1, y0:y1]
            cropped_post_segmap = post_segmap[:, x0:x1, y0:y1]

        # print(x0, y0, x1, y1)
        # print(cropped_pre_img.shape, cropped_post_img.shape, cropped_pre_segmap.shape, cropped_post_segmap.shape)
        # There are 6 possible flips: None, rot90, rot180, rot270, hflip & vflip
        flip = get_rand_flips()
        if flip is None or self.flips is False:
            flipped_pre_img = cropped_pre_img
            flipped_post_img = cropped_post_img
            flipped_pre_segmap = cropped_pre_segmap
            flipped_post_segmap = cropped_post_segmap
        else:
            # print(flip)
            if "rot" in flip:
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

        if self.mode == "batch":
            return (flipped_pre_img, flipped_pre_segmap,
                    flipped_post_img, flipped_post_segmap)

        # We need all tiles of the image
        if self.mode == "tiles":
            assert self.crop_size % self.tile_size == 0, "bad tile size"

            num_tiles = self.crop_size // self.tile_size

            pre_img_crops = []
            post_img_crops = []

            if self.load_segmaps:
                pre_seg_crops = []
                post_seg_crops = []

            for x in range(num_tiles):
                for y in range(num_tiles):

                    x0 = x * self.tile_size
                    x1 = (x + 1) * self.tile_size
                    y0 = y * self.tile_size
                    y1 = (y + 1) * self.tile_size

                    pre_img_crops.append(flipped_pre_img[:, x0: x1, y0: y1])
                    post_img_crops.append(flipped_post_img[:, x0: x1, y0: y1])

                    if self.load_segmaps:
                        pre_seg_crops.append(flipped_pre_segmap[:, x0: x1, y0: y1])
                        post_seg_crops.append(flipped_post_segmap[:, x0: x1, y0: y1])

            if self.load_segmaps:
                return (pre_img_crops, pre_seg_crops,
                        post_img_crops, post_seg_crops)
            else:
                return (pre_img_crops, post_img_crops)

    def __len__(self):
        return len(self.index)


def batch_collate_fn(batch):
    images = [x[0] for x in batch]
    labels = [x[1] for x in batch]

    flat_images = [item.type(torch.float32) for sublist in images for item in sublist]
    flat_labels = [item.type(torch.float32) for sublist in labels for item in sublist]

    # print("Num images in batch", len(flat_images))
    # print(flat_images[0].size)

    images_batch_tensor = torch.stack(flat_images)
    labels_batch_tensor = torch.stack(flat_labels)

    return (images_batch_tensor, labels_batch_tensor)


def tiles_collate_fn(batch):
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


def xview_train_loader_factory(mode, xview_root, data_version, use_tier3, crop_size, tile_size, batch_size, num_workers):
    # Read metadata
    train_data, val_data, _ = load_xview_metadata(xview_root, data_version, use_tier3)

    # print("Train images:", len(train_data))
    # print("Validation images:", len(val_data))
    # print("Test images:", len(test_data))

    if mode == "segmentation":
        train_mode = "batch"
        train_tile_size = -1
        train_crop_size = crop_size
        train_collate_fn = batch_collate_fn
        val_tile_size = tile_size
        val_crop_size = 1024
        val_batch_size = 1
    elif mode == "change":
        train_mode = "tiles"
        train_tile_size = tile_size
        train_crop_size = crop_size
        train_collate_fn = tiles_collate_fn
        val_tile_size = tile_size
        val_crop_size = 1024
        val_batch_size = batch_size

    train_set = xviewDataset(train_data, mode=train_mode, load_segmaps=True,
                             actual_size=1024, crop_size=train_crop_size, tile_size=train_tile_size,
                             flips=True, scale_jitter=True, color_jitter=True,
                             erase=False, noise=False, distort=False)

    val_set = xviewDataset(val_data, mode="tiles", load_segmaps=True,
                           actual_size=1024, crop_size=val_crop_size, tile_size=val_tile_size)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              collate_fn=train_collate_fn, num_workers=num_workers,
                              pin_memory=True)

    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False,
                            collate_fn=tiles_collate_fn, num_workers=1,
                            pin_memory=True)

    return train_loader, val_loader


def xview_test_loader_factory(xview_root, tile_size):
    # Read metadata
    _, _, test_data = load_xview_metadata(xview_root)

    test_set = xviewDataset(test_data, mode="tiles", actual_size=1024, crop_size=1024, tile_size=tile_size)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             collate_fn=test_collate_fn, num_workers=1, pin_memory=True)

    return test_loader
