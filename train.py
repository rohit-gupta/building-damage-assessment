#!/usr/bin/env python3
"""
Training script for xview challenge
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None

import numpy as np
from PIL import Image
import argparse
import pathlib
import tqdm

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn

from apex import amp
from apex.parallel import DistributedDataParallel as DDP, convert_syncbn_model

from dsilva_metrics.iou import IoU
from logger import MetricLog
from metrics import AverageMeter
from dataset import xview_train_loader_factory

from utils import input_tensor_to_pil_img
from utils import segmap_tensor_to_pil_img
from utils import reconstruct_from_tiles
from utils import postprocess_segmap_tensor_to_pil_img, logits_to_probs

from utils import reduce_tensor

from losses import cross_entropy, localization_aware_cross_entropy


# Configuration
import os
from config_parser import read_config

config_name = os.environ["XVIEW_CONFIG"]
config = read_config(config_name)


parser = argparse.ArgumentParser()
# local_rank argument is supplied automatically by torch.distributed.launch
parser.add_argument("--local_rank", default=1, type=int)
args = parser.parse_args()

args.distributed = False

if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.world_size = int(os.environ['WORLD_SIZE'])

if args.distributed:
    # FOR DISTRIBUTED:  Set the device according to local_rank.
    torch.cuda.set_device(args.local_rank)

    # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
    # environment variables, and requires that you use init_method=`env://`.
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
else:
    if torch.cuda.is_available():
        torch.cuda.set_device("cuda:0")
    else:
        torch.cuda.set_device("cpu")
semseg_model = deeplabv3_resnet50(pretrained=False,
                                  num_classes=5,
                                  aux_loss=None)
semseg_model = semseg_model.cuda()


# print(semseg_model)
# create dataloader
trainloader, valloader, train_sampler = xview_train_loader_factory("segmentation",
                                                                   config["paths"]["XVIEW_ROOT"],
                                                                   config["dataloader"]["DATA_VERSION"],
                                                                   config["dataloader"]["USE_TIER3_TRAIN"],
                                                                   config["dataloader"]["CROP_SIZE"],
                                                                   config["dataloader"]["TILE_SIZE"],
                                                                   config["dataloader"]["BATCH_SIZE"],
                                                                   config["dataloader"]["THREADS"],
                                                                   args.distributed)

if args.distributed:
    semseg_model = convert_syncbn_model(semseg_model)

if config["hyperparameters"]["OPTIMIZER"] == "ADAMW":
    optimizer = optim.AdamW(semseg_model.parameters(),
                            lr=config["hyperparameters"]["INITIAL_LR"],
                            weight_decay=config["hyperparameters"]["WEIGHT_DECAY"])
elif config["hyperparameters"]["OPTIMIZER"] == "SGD":
    optimizer = optim.SGD(semseg_model.parameters(),
                          lr=config["hyperparameters"]["INITIAL_LR"],
                          momentum=config["hyperparameters"]["MOMENTUM"],
                          weight_decay=config["hyperparameters"]["WEIGHT_DECAY"])
elif config["hyperparameters"]["OPTIMIZER"] == "ASGD":
    optimizer = optim.SGD(semseg_model.parameters(),
                          lr=config["hyperparameters"]["INITIAL_LR"],
                          to=1e5,
                          weight_decay=config["hyperparameters"]["WEIGHT_DECAY"])

# initialize mixed precision training
#print(config["misc"]["APEX_OPT_LEVEL"])
if config["misc"]["APEX_OPT_LEVEL"] != "None":
    semseg_model, optimizer = amp.initialize(semseg_model, optimizer, opt_level=config["misc"]["APEX_OPT_LEVEL"])

if args.distributed:
    semseg_model = DDP(semseg_model)

if "finetune" in config_name:
    model_checkpoint = config["paths"]["MODELS"] + config["paths"]["BEST_MODEL"]
    semseg_model.load_state_dict(torch.load(model_checkpoint))
    print("Loading weights from", model_checkpoint)

# print("Entering Training Loop")
if args.local_rank == 1:
    train_loss = AverageMeter("train_loss")
    train_loc_loss = AverageMeter("train_loc_loss")
    train_cls_loss = AverageMeter("train_cls_loss")
    val_loss = AverageMeter("val_loss")
    val_loc_loss = AverageMeter("val_loc_loss")
    val_cls_loss = AverageMeter("val_cls_loss")
    # iou_mean = AverageMeter("mIoU")
    # iou_localization = AverageMeter("localization_iou")
    # train_iou = IoU(5)
    val_pre_iou = IoU(5)
    val_post_iou = IoU(5)

    train_loss_log = MetricLog("train_loss")
    train_loc_loss_log = MetricLog("train_loc_loss")
    train_cls_loss_log = MetricLog("train_cls_loss")
    # train_mIoU_log = MetricLog("train_mIoU")
    # train_localization_IoU_log = MetricLog("train_localization_IoU")

    val_loss_log = MetricLog("val_loss")
    val_loc_loss_log = MetricLog("val_loc_loss")
    val_cls_loss_log = MetricLog("val_cls_loss")
    val_mIoU_log = MetricLog("val_mIoU")
    val_localization_IoU_log = MetricLog("val_localization_IoU")

for epoch in range(int(config["hyperparameters"]["NUM_EPOCHS"])):

    print("Beginning Epoch #" + str(epoch) + ":\n")

    if args.distributed:
        train_sampler.set_epoch(epoch)

    if args.local_rank == 1:
        # Reset Loss & metric tracking at beginning of epoch
        train_loss.reset()
        train_loc_loss.reset()
        train_cls_loss.reset()
        val_loss.reset()
        val_loc_loss.reset()
        val_cls_loss.reset()
        # train_iou.reset()
        val_pre_iou.reset()
        val_post_iou.reset()

    # Put model in training mode
    semseg_model.train()

    if args.local_rank == 1:
        train_pbar = tqdm.tqdm(total=len(trainloader))
    for images, segmaps in trainloader:
        # Send tensors to GPU
        images = images.cuda()
        segmaps = segmaps.cuda()

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
            if config["hyperparameters"]["LOSS"] == "crossentropy":
                loss = cross_entropy(segmaps, pred_segmaps)
            elif config["hyperparameters"]["LOSS"] == "locaware":
                loc_loss, cls_loss, loss = localization_aware_cross_entropy(segmaps, pred_segmaps,
                                                                            config["hyperparameters"]["LOCALIZATION_WEIGHT"],
                                                                            config["hyperparameters"]["CLASSIFICATION_WEIGHT"])

            if config["misc"]["APEX_OPT_LEVEL"] != "None":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()


        if args.distributed:
            reduced_loss = reduce_tensor(loss, args.world_size)
            reduced_loc_loss = reduce_tensor(loc_loss, args.world_size)
            reduced_cls_loss = reduce_tensor(cls_loss, args.world_size)
            loss_val = reduced_loss.item()
            loc_loss_val = reduced_loc_loss.item()
            cls_loss_val = reduced_cls_loss.item()
            count = args.world_size
        else:
            loss_val = loss.item()
            loc_loss_val = loc_loss.item()
            cls_loss_val = cls_loss.item()
            count = 1.

        if args.local_rank == 1:
            train_loss.update(val=loss_val, n=1./count)
            train_loc_loss.update(val=loc_loss_val, n=1./count)
            train_cls_loss.update(val=cls_loss_val, n=1./count)
        # gt_segmaps_classid = segmaps.argmax(1)
        # train_iou.add(pred_segmaps.detach(), gt_segmaps_classid.detach())
        train_pbar.update(1)

    if args.local_rank == 1:

        # End of an epoch
        # (train_iou_list, train_miou) = train_iou.value()
        # train_localization_iou = train_iou_list[0]

        # Log train metrics
        train_loss_log.update(train_loss.value())
        train_loc_loss_log.update(train_loc_loss.value())
        train_cls_loss_log.update(train_cls_loss.value())
        # train_mIoU_log.update(train_miou)
        # train_localization_IoU_log.update(train_localization_iou)

        train_pbar.close()

        # Validation Phase of epoch
        # Assume batch_size = 1 (higher sizes are impractical)

        val_pbar = tqdm.tqdm(total=len(valloader))
        semseg_model.eval()
        for idx, (pretiles, posttiles, prelabels, postlabels) in enumerate(valloader):
            n_val = len(pretiles)

            pretiles[0] = pretiles[0].cuda()
            posttiles[0] = posttiles[0].cuda()

            prelabels[0] = prelabels[0].cuda()
            postlabels[0] = postlabels[0].cuda()

            with torch.set_grad_enabled(False):
                preoutputs = semseg_model(pretiles[0])
                pre_preds = preoutputs['out']
                postoutputs = semseg_model(posttiles[0])
                post_preds = postoutputs['out']

                # Compute metrics (for now we always log locaware val loss)
                val_pre_loc_loss,\
                val_pre_cls_loss,\
                val_pre_loss_val = localization_aware_cross_entropy(prelabels[0], pre_preds,
                                                                    config["hyperparameters"]["LOCALIZATION_WEIGHT"],
                                                                    config["hyperparameters"]["CLASSIFICATION_WEIGHT"])
                val_post_loc_loss,\
                val_post_cls_loss,\
                val_post_loss_val = localization_aware_cross_entropy(postlabels[0], post_preds,
                                                                     config["hyperparameters"]["LOCALIZATION_WEIGHT"],
                                                                     config["hyperparameters"]["CLASSIFICATION_WEIGHT"])
                val_loss_val = (val_pre_loss_val + val_post_loss_val)/2
                val_loss.update(val=val_loss_val.item(), n=1)
                val_loc_loss.update(val=val_pre_loc_loss.item(), n=1)
                val_cls_loss.update(val=val_post_cls_loss.item(), n=1)

                pre_gt_classid = prelabels[0].argmax(1)
                post_gt_classid = postlabels[0].argmax(1)
                val_pre_iou.add(pre_preds.detach(), pre_gt_classid.detach())
                val_post_iou.add(post_preds.detach(), post_gt_classid.detach())

            # Write to disk for visually tracking training progress
            save_path = "val_results/" + config_name + "/" + str(idx) + "/"
            pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
            pre_pred = reconstruct_from_tiles(pre_preds, 5, config["dataloader"]["TILE_SIZE"])
            post_pred = reconstruct_from_tiles(post_preds, 5, config["dataloader"]["TILE_SIZE"])

            r = postprocess_segmap_tensor_to_pil_img(logits_to_probs(pre_pred), binarize=True)
            r.save(save_path + "pre_pred_epoch_" + str(epoch) + ".png")
            r = postprocess_segmap_tensor_to_pil_img(logits_to_probs(post_pred))
            r.save(save_path + "post_pred_epoch_" + str(epoch) + ".png")

            # Groundtruth only needs to be saved once
            if epoch == 0:
                pre_img = reconstruct_from_tiles(pretiles[0], 3, config["dataloader"]["TILE_SIZE"])
                post_img = reconstruct_from_tiles(posttiles[0], 3, config["dataloader"]["TILE_SIZE"])

                pre_gt = reconstruct_from_tiles(prelabels[0], 5, config["dataloader"]["TILE_SIZE"])
                post_gt = reconstruct_from_tiles(postlabels[0], 5, config["dataloader"]["TILE_SIZE"])

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
        val_loc_loss_log.update(val_loc_loss.value())
        val_cls_loss_log.update(val_cls_loss.value())
        val_localization_IoU_log.update(val_localization_iou)
        val_mIoU_log.update(val_miou)

        val_pbar.close()

        if epoch % config["misc"]["SAVE_FREQ"] == 0:
            models_folder = config["paths"]["MODELS"] + config_name + "/"
            pathlib.Path(models_folder).mkdir(parents=True, exist_ok=True)
            torch.save(semseg_model.state_dict(), models_folder + str(epoch) + ".pth")
    torch.cuda.synchronize()

