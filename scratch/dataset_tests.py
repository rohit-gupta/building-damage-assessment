#!/usr/bin/env python3
"""
Training script for xview challenge
"""

__author__ = "Rohit Gupta"
__version__ = "dev"
__license__ = None


from utils import load_xview_metadata
from utils import read_labels_file
from utils import labels_to_segmentation_map, labels_to_bboxes
from utils import colors

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
import torch

from PIL import Image
import numpy as np


# μ and σ for xview dataset
MEANS = [0.309, 0.340, 0.255]
STDDEVS = [0.162, 0.144, 0.135]
pil_to_tensor = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(MEANS, STDDEVS)
                                    ])


preprocess = transforms.Compose([transforms.ToTensor()])

semseg_model = deeplabv3_resnet50(pretrained=False,
                                  progress=True,
                                  num_classes=5,
                                  aux_loss=None)

print(semseg_model)

# Read metadata
xview_root = "/home/rohitg/data/xview/"
train_data, test_data = load_xview_metadata(xview_root)





# Random example
# random_key = "hurricane-michael_00000083"
# random_key = "palu-tsunami_00000097"

for key, metadata in train_data.items():
    # Pre Disaster Image
    file = train_data[key]["pre_label_file"]
    labels_data = read_labels_file(file)

    segmap_np = labels_to_segmentation_map(labels_data)
    segmap = torch.from_numpy(segmap_np)

    image_file = train_data[key]["pre_image_file"]
    im_tensor = pil_to_tensor(Image.open(image_file))

# print(np.argmax(segmap_np, axis=0))

# bboxes, labels = labels_to_bboxes(labels_data)
# bboxes = torch.from_numpy(np.array(bboxes))
# labels = torch.from_numpy(np.array(labels))


input_batch = im_tensor.unsqueeze(0)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    semseg_model.to('cuda')


semseg_model.eval()

with torch.no_grad():
    output = semseg_model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

# print("labels shape:", labels.shape)
# print("bboxes shape:", bboxes.shape)
# print("im shape:", im.shape)

print("input shape:", input_batch.shape)
print("segmap shape:", segmap.shape)

# print("labels dtype:", labels.dtype)
# print("bboxes dtype:", bboxes.dtype)
# print("im dtype:", im.dtype)
print("input dtype:", input_batch.dtype)
print("segmap dtype:", segmap.dtype)


print(np.sum(np.equal(segmap_np,segmap.byte().cpu().numpy())))

print(segmap.argmax(0))
print(segmap.argmax(0).byte().cpu().numpy())

r = Image.fromarray(torch.max(segmap, 0).indices.byte().cpu().numpy())
r.putpalette(colors)
r.save("saved_tensor.png")


r = Image.fromarray(im_tensor.byte().cpu().numpy())
r.save("saved_tensor_img.png")


# r = Image.fromarray(4 - segmap.argmax(0).byte().cpu().numpy())
# r.putpalette(colors)
# r.save("saved_tensor_flipped.png")





# Post Disaster Image
# random_file = train_data[random_key]["post_label_file"]
# labels_data = read_labels_file(random_file)
# image_file = train_data[random_key]["post_image_file"]
# segmap = labels_to_segmentation_map(labels_data)
# bboxes, labels = labels_to_bboxes(labels_data)
# bboxes, labels = np.array(bboxes), np.array(labels)
# im = Image.open(image_file)
