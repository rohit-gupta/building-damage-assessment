#!/usr/bin/env python3
"""
Functions to parse config files for training semantic segmentation models for xview challenge
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None

import configparser


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


def str2intarr(v):
  return [int(x) for x in v.split(" ")]

type_map = {
    "hyperparams": {
        "LOSS": str,
        "LOC_LOSS": str,
        "LOC_WEIGHT": float,
        "CLS_WEIGHT": float,
        "FOCAL_GAMMA": float,
        "NUM_EPOCHS": int,
        "OPTIMIZER": str,
        "INITIAL_LR": float,
        "WEIGHT_DECAY": float,
        "MOMENTUM": float,
        "WARMUP_EPOCHS": int,
        "MAX_LR": float,
        "STEP_PERIOD": int,
        "STEP_RATIO": float
    },
    "dataloader": {
        "DATA_VERSION": str,
        "USE_TIER3_TRAIN": str2bool,
        "CROP_SIZE": int,
        "TILE_SIZE": int,
        "THREADS": int,
        "BATCH_SIZE": int
    },
    "paths": {
        "XVIEW_ROOT": str,
        "LOGS": str,
        "MODELS": str,
        "BEST_MODEL": str
    },
    "misc": {
        "SAVE_FREQ": int,
        "APEX_OPT_LEVEL": str
    }
}


change_type_map = {
    "change": {
        "SEG_MODEL" : str,
        "POST_CROP": int,
        "KERNEL_SIZES": str2intarr,
        "KERNEL_DILATIONS": str2intarr,
        "NUM_LAYERS": int
    }
}


def read_config(config_name, config_type="segmentation"):
    config_file = "configs/" + config_name + ".ini"
    config = configparser.ConfigParser()
    config.optionxform = lambda option: option
    config.read(config_file)

    if config_type == "segmentation":
        return parse_config(config._sections, type_map)
    elif config_type == "change":
        full_type_map = {}
        full_type_map.update(type_map)
        full_type_map.update(change_type_map)
        return parse_config(config._sections, full_type_map)
    else:
        raise TypeError("config_type must be either 'segmentation' or 'change'")


def parse_config(config, template):
    print(config)
    for section, params in template.items():
        for param_name, param_type in params.items():
            config[section][param_name] = param_type(config[section][param_name])

    return config


if __name__ == '__main__':
    import os
    import pprint
    config_name = os.environ["XVIEW_CONFIG"]
    config = read_config(config_name)
    pprint.pprint(config)

