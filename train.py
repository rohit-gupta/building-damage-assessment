#!/usr/bin/env python3
"""
Training script for xview challenge
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None

import argparse
import pathlib
import tqdm

from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import torch.optim as optim

from apex import amp
from apex.parallel import DistributedDataParallel as DDP, convert_syncbn_model

from dsilva_metrics.iou import IoU
from logger import MetricLog
from metrics import AverageMeter
from dataset import xview_train_loader_factory, xview_val_loader_factory

from utils.train import save_val_results, save_val_gt
from utils.misc import reduce_tensor, clean_distributed_state_dict
from losses import cross_entropy, localization_aware_loss
from seg_models import DeepLabv3PlusModel


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

torch.backends.cudnn.benchmark = True

if config["arch"]["META_ARCH"] == "DeepLabv3":
    semseg_model = deeplabv3_resnet50(pretrained=False,
                                      num_classes=5,
                                      aux_loss=None)
elif config["arch"]["META_ARCH"] == "DeepLabv3+":
    semseg_model = DeepLabv3PlusModel(arch="resnet50", output_stride=8, pretrained_backbone=True, num_classes=5)

semseg_model = semseg_model.cuda()

# print(semseg_model)

# print(semseg_model)
ACTUAL_SIZE = 1024
NUM_TILES = ACTUAL_SIZE // config["dataloader"]["TILE_SIZE"]
NUM_TILES *= NUM_TILES
VAL_BATCH_SIZE = 16
# create dataloader
trainloader, train_sampler = xview_train_loader_factory("segmentation",
                                                        config["paths"]["XVIEW_ROOT"],
                                                        config["dataloader"]["DATA_VERSION"],
                                                        config["dataloader"]["USE_TIER3_TRAIN"],
                                                        config["dataloader"]["CROP_SIZE"],
                                                        -1,
                                                        config["dataloader"]["BATCH_SIZE"],
                                                        config["dataloader"]["THREADS"],
                                                        args.distributed)
valloader = xview_val_loader_factory(config["paths"]["XVIEW_ROOT"],
                                     config["dataloader"]["DATA_VERSION"],
                                     VAL_BATCH_SIZE)
INITIAL_EPOCH = 0
if "finetune" in config_name:
    model_checkpoint = config["paths"]["MODELS"] + config["paths"]["BEST_MODEL"]
    checkpoint_config = config["paths"]["BEST_MODEL"].split("/")[0]
    state_dict = torch.load(model_checkpoint)
    # Clean distributed state dict is idempotent
    semseg_model.load_state_dict(clean_distributed_state_dict(state_dict))
    print("Loading weights from", model_checkpoint)
    if checkpoint_config == config_name:
        INITIAL_EPOCH = int(config["paths"]["BEST_MODEL"].split("/")[-1].split(".")[0]) + 1

if args.distributed:
    semseg_model = convert_syncbn_model(semseg_model)

if config["hyperparams"]["OPTIMIZER"] == "ADAMW":
    optimizer = optim.AdamW(semseg_model.parameters(),
                            lr=config["hyperparams"]["INITIAL_LR"],
                            weight_decay=config["hyperparams"]["WEIGHT_DECAY"])
elif config["hyperparams"]["OPTIMIZER"] == "SGD":
    optimizer = optim.SGD(semseg_model.parameters(),
                          lr=config["hyperparams"]["INITIAL_LR"],
                          momentum=config["hyperparams"]["MOMENTUM"],
                          weight_decay=config["hyperparams"]["WEIGHT_DECAY"])
elif config["hyperparams"]["OPTIMIZER"] == "ASGD":
    optimizer = optim.ASGD(semseg_model.parameters(),
                          lr=config["hyperparams"]["INITIAL_LR"],
                          to=1e5,
                          weight_decay=config["hyperparams"]["WEIGHT_DECAY"])

# initialize mixed precision training
if config["misc"]["APEX_OPT_LEVEL"] != "None":
    semseg_model, optimizer = amp.initialize(semseg_model, optimizer, opt_level=config["misc"]["APEX_OPT_LEVEL"])

if args.distributed:
    semseg_model = DDP(semseg_model)

if args.local_rank == 1:
    train_loss = AverageMeter("train_loss")
    if config["hyperparams"]["LOSS"] == "locaware":
        train_loc_loss = AverageMeter("train_loc_loss")
        train_loc_ce_loss = AverageMeter("train_loc_ce_loss")
        train_cls_loss = AverageMeter("train_cls_loss")
    val_loss = AverageMeter("val_loss")
    val_iou = IoU(5)
    train_loss_log = MetricLog("train_loss")
    val_loss_log = MetricLog("val_loss")
    if config["hyperparams"]["LOSS"] == "locaware":
        train_loc_loss_log = MetricLog("train_loc_loss")
        train_loc_ce_loss_log = MetricLog("train_loc_ce_loss")
        train_cls_loss_log = MetricLog("train_cls_loss")
    val_mIoU_log = MetricLog("val_mIoU")
    val_localization_IoU_log = MetricLog("val_localization_IoU")

for epoch in range(INITIAL_EPOCH, INITIAL_EPOCH + int(config["hyperparams"]["NUM_EPOCHS"])):

    print("Beginning Epoch #" + str(epoch) + ":\n")

    if args.distributed:
        train_sampler.set_epoch(epoch)

    if args.local_rank == 1:
        # Reset Loss & metric tracking at beginning of epoch
        train_loss.reset()
        val_loss.reset()
        if config["hyperparams"]["LOSS"] == "locaware":
            train_loc_loss.reset()
            train_loc_ce_loss.reset()
            train_cls_loss.reset()
        val_iou.reset()

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
            # Hacky Categorical CE alternative using Binary CE
            # pred_probas = nn.Softmax(dim=1)(pred_segmaps)
            # loss = nn.BCELoss()(pred_probas, segmaps)

            # Cross Entropy Loss
            if config["hyperparams"]["LOSS"] == "crossentropy":
                loss = cross_entropy(segmaps, pred_segmaps)
            elif config["hyperparams"]["LOSS"] == "locaware":
                loc_loss, loc_ce_loss, cls_loss, loss = localization_aware_loss(segmaps, pred_segmaps,
                                                                   config["hyperparams"]["LOC_WEIGHT"],
                                                                   config["hyperparams"]["CLS_WEIGHT"],
                                                                   loc_loss_type=config["hyperparams"]["LOC_LOSS"],
                                                                   loc_ce_wt=config["hyperparams"]["LOC_CE_WEIGHT"],
                                                                   loc_oth_wt=config["hyperparams"]["LOC_OTH_WEIGHT"],
                                                                   gamma=config["hyperparams"]["FOCAL_GAMMA"])

            if config["misc"]["APEX_OPT_LEVEL"] != "None":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()


        if args.distributed:
            reduced_loss = reduce_tensor(loss, args.world_size)
            loss_val = reduced_loss.item()
            if config["hyperparams"]["LOSS"] == "locaware":
                reduced_loc_loss = reduce_tensor(loc_loss, args.world_size)
                reduced_loc_ce_loss = reduce_tensor(loc_ce_loss, args.world_size)
                reduced_cls_loss = reduce_tensor(cls_loss, args.world_size)
                loc_loss_val = reduced_loc_loss.item()
                loc_ce_loss_val = reduced_loc_ce_loss.item()
                cls_loss_val = reduced_cls_loss.item()
            count = args.world_size
        else:
            loss_val = loss.item()
            if config["hyperparams"]["LOSS"] == "locaware":
                loc_loss_val = loc_loss.item()
                loc_ce_loss_val = loc_ce_loss.item()
                cls_loss_val = cls_loss.item()
            count = 1.

        if args.local_rank == 1:
            train_loss.update(val=loss_val, n=1./count)
            if config["hyperparams"]["LOSS"] == "locaware":
                train_loc_loss.update(val=loc_loss_val, n=1./count)
                train_loc_ce_loss.update(val=loc_ce_loss_val, n=1./count)
                train_cls_loss.update(val=cls_loss_val, n=1./count)
            train_pbar.update(1)

    if args.local_rank == 1:

        # End of an epoch
        # Log train metrics
        train_loss_log.update(train_loss.value())
        if config["hyperparams"]["LOSS"] == "locaware":
            train_loc_loss_log.update(train_loc_loss.value())
            train_loc_ce_loss_log.update(train_loc_ce_loss.value())
            train_cls_loss_log.update(train_cls_loss.value())

        train_pbar.close()

        # Validation Phase of epoch
        val_pbar = tqdm.tqdm(total=len(valloader))
        semseg_model.eval()
        for idx, (pretiles, posttiles, prelabels, postlabels) in enumerate(valloader):
            n_val = len(pretiles)

            pretiles = torch.cat(pretiles, dim=0).cuda()
            posttiles = torch.cat(posttiles, dim=0).cuda()

            prelabels = torch.cat(prelabels, dim=0).cuda()
            postlabels = torch.cat(postlabels, dim=0).cuda()

            with torch.set_grad_enabled(False):
                pred_segmentations = semseg_model(torch.cat((pretiles, posttiles), dim=0))['out']
                gt_segmentations = torch.cat((prelabels, postlabels), dim=0)
                pre_preds = pred_segmentations[:VAL_BATCH_SIZE//2, :, :, :]
                post_preds = pred_segmentations[VAL_BATCH_SIZE//2:, :, :, :]

                # Compute metrics
                if config["hyperparams"]["LOSS"] == "locaware":
                    val_loc_loss,\
                    val_loc_ce_loss,\
                    val_cls_loss,\
                    val_loss_val = localization_aware_loss(gt_segmentations, pred_segmentations,
                                                               config["hyperparams"]["LOC_WEIGHT"],
                                                               config["hyperparams"]["CLS_WEIGHT"], 
                                                               loc_loss_type=config["hyperparams"]["LOC_LOSS"],
                                                               loc_ce_wt=config["hyperparams"]["LOC_CE_WEIGHT"],
                                                               loc_oth_wt=config["hyperparams"]["LOC_OTH_WEIGHT"],
                                                               gamma=config["hyperparams"]["FOCAL_GAMMA"])
                elif config["hyperparams"]["LOSS"] == "crossentropy":
                    val_loss_val = cross_entropy(gt_segmentations, pred_segmentations)

                val_loss.update(val=val_loss_val.item(), n=VAL_BATCH_SIZE)

                # gt_classid = gt_segmentations.argmax(1)
                # val_iou.add(pred_segmentations.detach(), gt_classid.detach())

            # Write to disk for visually tracking training progress
            for i in range(VAL_BATCH_SIZE):
                save_path = "val_results/" + config_name + "/" + str(idx + i) + "/"
                pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
                save_val_results(save_path, epoch, config["hyperparams"]["LOSS"], pre_preds[i], post_preds[i], tiled=False)
                # Groundtruth only needs to be saved once
                if epoch == 0:
                    save_val_gt(save_path, pretiles[i], posttiles[i], prelabels[i], postlabels[i], 1024)
            
            val_pbar.update(1)

        # End of val phase
        # (val_iou_list, val_miou) = val_iou.value()
        # val_localization_iou = val_iou_list[0]
        # val_miou = val_miou

        val_loss_log.update(val_loss.value())
        # val_localization_IoU_log.update(val_localization_iou)
        # val_mIoU_log.update(val_miou)

        val_pbar.close()
        
        del pretiles[0], posttiles[0], prelabels[0], postlabels[0]
        del pred_segmentations, gt_segmentations, pre_preds, post_preds



        if epoch % config["misc"]["SAVE_FREQ"] == 0:
            models_folder = config["paths"]["MODELS"] + config_name + "/"
            pathlib.Path(models_folder).mkdir(parents=True, exist_ok=True)
            torch.save(semseg_model.state_dict(), models_folder + str(epoch) + ".pth")
    torch.cuda.synchronize()
