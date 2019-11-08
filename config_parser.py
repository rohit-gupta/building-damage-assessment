#!/usr/bin/env python3
"""
Functions to parse config files for training semantic segmentation models for xview challenge
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None

import configparser

type_map = {
    "hyperparameters": {
        "LOSS": str,
        "LOCALIZATION_WEIGHT": float,
        "CLASSIFICATION_WEIGHT": float,
        "NUM_EPOCHS": int,
        "INITIAL_LR": float,
        "MOMENTUM": float,
        "WARMUP_EPOCHS": int,
        "MAX_LR": float,
        "STEP_PERIOD": int,
        "STEP_RATIO": float
    },
    "dataloader": {
        "DATA_VERSION": str,
        "CROP_SIZE": int,
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


def read_config(config_name):
    config_file = config_name + ".ini"
    config = configparser.ConfigParser()
    config.read(config_file)
    return parse_config(config, type_map)


def parse_config(config, template):
    for section, params in template.items():
        for param_name, param_type in params:
            config[section][param_name] = param_type(config[section][param_name])

    return config
