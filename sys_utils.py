#!/usr/bin/env python3
"""
Dataloaders for xview challenge
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None


import torch
import yaml
import os 

CUDA_AVAIL = torch.cuda.is_available()
CUDA_DEVICES = {}



if CUDA_AVAIL:
    CUDA_COUNT = torch.cuda.device_count()
    for dev_id in range(CUDA_COUNT):
        CUDA_DEVICES[dev_id] = torch.cuda.get_device_name(dev_id)

with open('cuda_' +os.environ['GPU_ARCH']+ '.yml', 'w') as yaml_file:
    yaml.dump(CUDA_DEVICES, yaml_file, default_flow_style=False)
