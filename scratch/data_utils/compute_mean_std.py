#!/usr/bin/env python3
"""
Dataloaders for xview challenge
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None

# import torchvision
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

from PIL import Image

from utils import load_xview_metadata


class xviewDataset(Dataset):
    def __init__(self, file_list):
        self.data = file_list

    def __getitem__(self, index):
        x = self.data[index]
        img = Image.open(x).convert('RGB')
        return transforms.ToTensor()(img)

    def __len__(self):
        return len(self.data)


def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for data in loader:

        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


def mean_std(file_list):
    dataset = xviewDataset(file_list)
    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=4,
        shuffle=False
    )

    mean, std = online_mean_and_sd(loader)
    return (mean.numpy(), std.numpy())


def main():
    # Read metadata
    xview_root = "/home/rohitg/data/xview/"
    train, test = load_xview_metadata(xview_root)

    train_pre_files = [x["pre_image_file"] for x in list(train.values())]
    train_post_files = [x["post_image_file"] for x in list(train.values())]
    train_files = train_pre_files + train_post_files

    test_pre_files = [x["pre_image_file"] for x in list(test.values())]
    test_post_files = [x["post_image_file"] for x in list(test.values())]
    test_files = test_pre_files + test_post_files

    print("train_pre_files", mean_std(train_pre_files))
    print("train_post_files", mean_std(train_post_files))
    print("train_files", mean_std(train_files))

    print("test_pre_files", mean_std(test_pre_files))
    print("test_post_files", mean_std(test_post_files))
    print("test_files", mean_std(test_files))


if __name__ == '__main__':
    main()
