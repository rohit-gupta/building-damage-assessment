#!/usr/bin/env python3
"""
Class for logging metrics
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None

import os
import pathlib

# TODO Log to Tensorboard etc
# TODO for now just log to text file
config_name = os.environ["XVIEW_CONFIG"]

class MetricLog(object):
    """Stores the value of a metric over training"""
    def __init__(self, name):
        self.name = name
        log_dir = "logs/" + config_name + "/"
        self.log_file = log_dir + name + ".txt"
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "w") as f:
            print("epoch," + name, file=f)
        self.reset()

    def reset(self):
        self.values = []

    def update(self, val):
        self.values.append(val)
        with open(self.log_file, "a") as f:
            print(str(len(self.values)) + "," + str(val), file=f)

    def value(self):
        return self.values[-1]
        # return {"values" : self.values[-1], "average" : sum(values)/len(values)}

    def __str__(self):
        fmtstr = '{name}'
        return fmtstr.format(**self.__dict__)
