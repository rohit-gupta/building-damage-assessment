
import os
import sys

import torch
from torchvision.models.segmentation import deeplabv3_resnet50

from change_detection_model import ChangeDetectionNet
from config_parser import read_config
from dataset import xview_train_loader_factory

config_name = os.environ["XVIEW_CONFIG"]
config = read_config(config_name)

if config_name == "default_lb":
    BEST_EPOCH = 169
elif config_name == "locaware_local_ssd":
    BEST_EPOCH = -1  # TODO
elif config_name == "locaware_local_full_finetune":
    BEST_EPOCH = 51  # TODO
if torch.cuda.is_available():
    gpu0 = torch.device('cuda:0')
    gpu1 = torch.device('cuda:1')
else:
    sys.exit("Cannot run with less than 2 GPUs")

semseg_model = deeplabv3_resnet50(pretrained=False,
                                  num_classes=5,
                                  aux_loss=None)

changenet = ChangeDetectionNet(classes=5, num_layers=3, feature_channels=15,
                               kernel_scales=[3, 11, 19], dilation_scales=[2, 4, 8],
                               use_bn=True, padding_type="replication")

semseg_model = semseg_model.to(gpu0)
changenet = changenet.to(gpu1)

print(changenet)

trainloader, valloader = xview_train_loader_factory("change",
                                                    config["paths"]["XVIEW_ROOT"],
                                                    config["dataloader"]["DATA_VERSION"],
                                                    config["dataloader"]["USE_TIER3_TRAIN"],
                                                    config["dataloader"]["CROP_SIZE"],
                                                    config["dataloader"]["TILE_SIZE"],
                                                    config["dataloader"]["BATCH_SIZE"],
                                                    config["dataloader"]["THREADS"])

print("Beginning Test Inference using model from Epoch #" + str(BEST_EPOCH) + ":")
models_folder = str(config["paths"]["MODELS"]) + config_name + "/"
semseg_model.load_state_dict(torch.load(models_folder + str(BEST_EPOCH) + ".pth"))
semseg_model.eval()
