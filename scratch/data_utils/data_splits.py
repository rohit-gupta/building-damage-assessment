#!/usr/bin/env python3
"""
Generate train and val splits for xview challenge
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None

from loader_utils import load_xview_metadata

from collections import Counter
import random

# Read metadata
xview_root = "/home/rohitg/data/xview/"
train_data, _ = load_xview_metadata(xview_root)


scenes = list(train_data.keys())
num_total_scenes = len(scenes)
print("Total training scenes:", num_total_scenes)

# num_val_scenes = 399
# num_train_scenes = num_total_scenes - num_val_scenes


# val_scenes = list(random.sample(scenes, num_val_scenes))
# train_scenes = list(set(scenes) - set(val_scenes))

# for labclass, count in Counter([x.split("_")[0] for x in scenes]).most_common():
#     print(labclass, round(100*float(count)/len(scenes), 2))
# print(Counter([x.split("_")[0] for x in val_scenes]).most_common())
# print(Counter([x.split("_")[0] for x in train_scenes]).most_common())


val_selections = {"socal-fire": 88,
                  "hurricane-michael": 37,
                  "hurricane-florence": 35,
                  "hurricane-harvey": 35,
                  "midwest-flooding": 30,
                  "hurricane-matthew": 26,
                  "santa-rosa-wildfire": 25,
                  "mexico-earthquake": 13,
                  "palu-tsunami": 13,
                  "guatemala-volcano": 2}

random.shuffle(scenes)

val = {x: [] for x in val_selections.keys()}

# print(val)

for disaster_type, count in val_selections.items():
    for scene in scenes:
        scene_type = scene.split("_")[0]
        if len(val[disaster_type]) < count:
            if scene_type == disaster_type:
                val[disaster_type].append(scene)

# print(val_selections)
val_list = []
for disaster_type, selected_scenes in val.items():
    # print(disaster_type, len(selected_scenes))
    val_list += selected_scenes

train_list = list(set(scenes) - set(val_list))
# print(val_list)
# print(len(val_list))

print("Split train set size:", num_total_scenes - len(val_list))
print("Split val set size:", len(val_list))


with open("val-split.txt", "w") as f:
    print("\n".join(val_list), file=f)

with open("train-split.txt", "w") as f:
    print("\n".join(train_list), file=f)
