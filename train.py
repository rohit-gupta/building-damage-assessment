#!/usr/bin/env python3
"""
Training script for xview challenge
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None




import numpy as np
from PIL import Image
import pathlib
import tqdm

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn



from dsilva_metrics.iou import IoU
from logger import MetricLog
from metrics import AverageMeter
from dataset import xview_train_loader_factory
from utils import input_tensor_to_pil_img
from utils import segmap_tensor_to_pil_img
from utils import reconstruct_from_tiles

# Configuration
import configparser
import os

config_name = os.environ["XVIEW_CONFIG"]
config_file = config_name + ".ini"
config = configparser.ConfigParser()
config.read(config_file)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

semseg_model = deeplabv3_resnet50(pretrained=False,
                                  num_classes=5,
                                  aux_loss=None)
semseg_model = semseg_model.to(device)
# print(semseg_model)
# create dataloader
trainloader, valloader = xview_train_loader_factory(config["paths"]["XVIEW_ROOT"],
                                                    int(config["dataloader"]["CROP_SIZE"]),
                                                    int(config["dataloader"]["BATCH_SIZE"]),
                                                    int(config["dataloader"]["THREADS"]))

# For starters, train a constant LR Model
optimizer = optim.SGD(semseg_model.parameters(),
                      lr=float(config["hyperparameters"]["MAX_LR"]),
                      momentum=float(config["hyperparameters"]["MOMENTUM"]))


# print("Entering Training Loop")

train_loss = AverageMeter("train_loss")
val_loss = AverageMeter("val_loss")
# iou_mean = AverageMeter("mIoU")
# iou_localization = AverageMeter("localization_iou")
train_iou = IoU(5)
val_pre_iou = IoU(5)
val_post_iou = IoU(5)


train_loss_log = MetricLog("train_loss")
train_mIoU_log = MetricLog("train_mIoU")
train_localization_IoU_log = MetricLog("train_localization_IoU")

val_loss_log = MetricLog("val_loss")
val_mIoU_log = MetricLog("val_mIoU")
val_localization_IoU_log = MetricLog("val_localization_IoU")


logsoftmax = nn.LogSoftmax(dim=1)

for epoch in range(int(config["hyperparameters"]["NUM_EPOCHS"])):

    print("Beginning Epoch #" +str(epoch) + ":" )
    # Reset Loss & metric tracking at beginning of epoch
    train_loss.reset()
    val_loss.reset()
    train_iou.reset()
    val_pre_iou.reset()
    val_post_iou.reset()

    # Put model in training mode
    semseg_model.train()

    train_pbar = tqdm.tqdm(total=len(trainloader))
    for images, segmaps in trainloader:
        # Send tensors to GPU
        images = images.to(device)
        segmaps = segmaps.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = semseg_model(images)
            pred_segmaps = outputs['out']
            # print("First forward pass done")
            # TODO Hacky Categorical CE alternative using Binary CE
            # pred_probas = nn.Softmax(dim=1)(pred_segmaps)
            # loss = nn.BCELoss()(pred_probas, segmaps)

            # Cross Entropy Loss
            loss = torch.mean(torch.sum(-segmaps * logsoftmax(pred_segmaps),
                                        dim=1))

            loss.backward()
            optimizer.step()

        train_loss.update(val=loss.item(), n=images.size(0))
        segmaps_classid = segmaps.argmax(1)
        train_iou.add(pred_segmaps.detach(), segmaps_classid.detach())
        train_pbar.update(1)

    # End of an epoch

    (train_iou_list, train_miou) = train_iou.value()
    train_localization_iou = train_iou_list[0]

    # Log train metrics
    train_loss_log.update(train_loss.value())
    train_mIoU_log.update(train_miou)
    train_localization_IoU_log.update(train_localization_iou)

    train_pbar.close()

    val_pbar = tqdm.tqdm(total=len(valloader))
    # Validation Phase of epoch
    # Assume batch_size = 1 (higher sizes are impractical)
    for idx, (pretiles, posttiles, prelabels, postlabels) in enumerate(valloader):
        semseg_model.eval()
        n_val = len(pretiles)

        pretiles[0] = pretiles[0].to(device)
        posttiles[0] = posttiles[0].to(device)

        prelabels[0] = prelabels[0].to(device)
        postlabels[0] = postlabels[0].to(device)

        with torch.set_grad_enabled(False):
            preoutputs = semseg_model(pretiles[0])
            pre_preds = preoutputs['out']
            postoutputs = semseg_model(posttiles[0])
            post_preds = postoutputs['out']

            # Compute metrics
            val_loss_val = torch.mean(torch.sum(-prelabels[0] * logsoftmax(pre_preds), dim=1))
            val_loss_val += torch.mean(torch.sum(-prelabels[0] * logsoftmax(pre_preds), dim=1))
            val_loss_val /= 2
            val_loss.update(val=val_loss_val.item(), n=2*prelabels[0].size(0))

            pre_gt_classid = prelabels[0].argmax(1)
            post_gt_classid = postlabels[0].argmax(1)
            val_pre_iou.add(pre_preds.detach(), pre_gt_classid.detach())
            val_post_iou.add(post_preds.detach(), post_gt_classid.detach())

        # Write small sample to disk for visually tracking training progress
        if idx in [0, 5, 50, 55, 100, 105, 150, 155, 200, 205, 250, 255, 300, 303]:

            save_path = "samples/" + str(idx) + "/"
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            pre_pred = reconstruct_from_tiles(pre_preds, 5, int(config["dataloader"]["CROP_SIZE"]))
            post_pred = reconstruct_from_tiles(post_preds, 5, int(config["dataloader"]["CROP_SIZE"]))

            r = segmap_tensor_to_pil_img(pre_pred)
            r.save(save_path + "pre_pred_epoch_" + str(epoch) + ".png")
            r = segmap_tensor_to_pil_img(post_pred)
            r.save(save_path + "post_pred_epoch_" + str(epoch) + ".png")

            # Groundtruth only needs to be saved once
            if epoch == 0:
                pre_img = reconstruct_from_tiles(pretiles[0], 3, int(config["dataloader"]["CROP_SIZE"]))
                post_img = reconstruct_from_tiles(posttiles[0], 3, int(config["dataloader"]["CROP_SIZE"]))

                pre_gt = reconstruct_from_tiles(prelabels[0], 5, int(config["dataloader"]["CROP_SIZE"]))
                post_gt = reconstruct_from_tiles(postlabels[0], 5, int(config["dataloader"]["CROP_SIZE"]))

                pilimg = input_tensor_to_pil_img(pre_img)
                pilimg.save(save_path + "pre_image.png")
                pilimg = input_tensor_to_pil_img(post_img)
                pilimg.save(save_path + "post_image.png")

                r = segmap_tensor_to_pil_img(pre_gt)
                r.save(save_path + "pre_gt.png")
                r = segmap_tensor_to_pil_img(post_gt)
                r.save(save_path + "post_gt.png")
        val_pbar.update(1)

    # End of val phase
    (pre_iou_list, pre_miou) = val_pre_iou.value()
    (post_iou_list, post_miou) = val_post_iou.value()
    val_localization_iou = pre_iou_list[0]
    val_miou = post_miou

    val_loss_log.update(val_loss.value())
    val_localization_IoU_log.update(val_localization_iou)
    val_mIoU_log.update(val_miou)

    val_pbar.close()

    if epoch % int(config["misc"]["SAVE_FREQ"]) == 0:
        models_folder = str(config["paths"]["MODELS"]) + config_name + "/"
        pathlib.Path(models_folder).mkdir(parents=True, exist_ok=True)
        torch.save(semseg_model.state_dict(), models_folder + str(epoch) + ".pth")




    # TODO Write validation code and compute metrics
    # semseg_model.eval()
    # segmaps_classid = segmaps.argmax(1)
    # metric.add(pred_segmaps.detach(), segmaps_classid.detach())
    # (iou, miou) = metric.value()
    #
    #

            
            
        # image = images[-1]
        # segmap = segmaps[-1]

        # pilimg = input_tensor_to_pil_img(image)
        # pilimg.save("post_sample_image_tensor" + str(idx) + ".png")

        # r = segmap_tensor_to_pil_img(segmap)
        # r.save("post_sample_segmap_tensor" + str(idx) + ".png")

        # print(images.shape, segmaps.shape)

    # if idx > 10:
    #     import sys
    #     sys.exit(0)

    # idx += 1


# image_file = train_data[random_key]["pre_image_file"]
# im_tensor = preprocess(Image.open(image_file))
# input_batch = im_tensor.unsqueeze(0)


# input_batch = input_batch.to(device)
# semseg_model.eval()

# with torch.no_grad():
#     output = semseg_model(input_batch)['out'][0]
#     output_predictions = output.argmax(0)

