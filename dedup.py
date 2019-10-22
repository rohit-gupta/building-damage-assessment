
from utils import load_xview_metadata
from utils import read_labels_file


# Read metadata
xview_root = "/home/rohitg/data/xview/"
train_data, test_data = load_xview_metadata(xview_root)


for scene_key, scene_files in train_data.items():
    pre_labels_data = read_labels_file(scene_files["pre_label_file"])
    post_labels_data = read_labels_file(scene_files["post_label_file"])
    if len(pre_labels_data) != len(post_labels_data) or "palu-tsunami_00000097" in scene_key:
        print(scene_key, len(pre_labels_data), len(post_labels_data))
        poly_texts = []
        for x in pre_labels_data:
            poly_texts.append(x["wkt"])
            # if x["properties"]["feature_type"] != "building":
            #     print(x["properties"])

        print(len(set(poly_texts)), len(poly_texts))

        poly_texts = []
        for x in post_labels_data:
            poly_texts.append(x["wkt"])

        print(len(set(poly_texts)), len(poly_texts))
