
from .dataset import load_xview_metadata, read_labels_file
import json

train_data, val_data, _ = load_xview_metadata("/home/c3-0/rohitg/xviewdata/xview/", data_version="v2/", use_tier3=True)

# print(val_data.keys())


val_loc_split = open("loc_val.txt", "w")
train_loc_split = open("loc_train.txt", "w")

val_cls_split = open("cls_val.csv", "w")
train_cls_split = open("cls_train.csv", "w")

print("id,uuid,labels", file=val_cls_split, flush=True)
print("id,uuid,labels", file=train_cls_split, flush=True)

label_mapping_dict = {"no-damage": 0,
                      "minor-damage": 1,
                      "major-damage": 2,
                      "destroyed": 3}

i = 0

for scene_id, scene_data in val_data.items():

    building_polys = read_labels_file(scene_data["post_label_file"])

    if len(building_polys) != 0:
        print(scene_data["pre_image_file"].split("/")[-1], file=val_loc_split, flush=True)
        
        for building_poly in building_polys:
            uuid = building_poly['properties']['uid']
            damage_type = building_poly['properties']['subtype']
            if damage_type != "un-classified":
                print(i, uuid, label_mapping_dict[damage_type], sep=",", file=val_cls_split, flush=True)
                i += 1


i = 0

for scene_id, scene_data in train_data.items():

    building_polys = read_labels_file(scene_data["post_label_file"])

    if len(building_polys) != 0:
        print(scene_data["pre_image_file"].split("/")[-1], file=train_loc_split, flush=True)
        
        for building_poly in building_polys:
            uuid = building_poly['properties']['uid']
            damage_type = building_poly['properties']['subtype']
            if damage_type != "un-classified":
                print(i, uuid, label_mapping_dict[damage_type], sep=",", file=train_cls_split, flush=True)
                i += 1
