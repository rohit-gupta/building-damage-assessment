

from loader_utils import load_xview_metadata, read_labels_file, labels_to_segmentation_map
import torch
import numpy as np


# Read metadata
xview_root = "/home/rohitg/data/xview/"
train_data, test_data = load_xview_metadata(xview_root)

# Random example
random_key = "hurricane-michael_00000083"

# Pre Disaster Image
random_file = train_data[random_key]["pre_label_file"]
labels_data = read_labels_file(random_file) # np.uint8

segmap_np = labels_to_segmentation_map(labels_data)

print("Segmap Shape:", segmap_np.shape)
print(np.argmax(segmap_np, axis=0))
# print(np.argmax(segmap_np, axis=0).shape)
segmap = torch.from_numpy(segmap_np)
print(torch.argmax(segmap, 0))
# print(torch.argmax(segmap, 0))
# print(torch.argmax(segmap, 0).shape)
print(np.argmax(segmap.byte().cpu().numpy(), axis=0))


print(segmap.data[:,0,0])
print(segmap_np[:,0,0])

print( "Equal Elements:", np.sum(np.equal(segmap_np,segmap.byte().cpu().numpy())))
tot_elements = 1
for x in segmap_np.shape:
    tot_elements *= x

print( "Total Elements:", tot_elements)