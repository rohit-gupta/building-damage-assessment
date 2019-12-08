

import numpy as np
from imantics import Polygons, Mask
from PIL import Image
from shapely.geometry import Polygon

def open_image_as_nparray(img_path, dtype):
    return np.array(Image.open(img_path), dtype=dtype)


location = "/home/c3-0/rohitg/xview_results/val_results/turing_finetune_tier1/"
preds = "epoch10_val_results/pred_"
gt = "gt_val_results/post_"

count = 304
selected = "70.png"
pred = open_image_as_nparray(location + preds + selected, dtype=np.uint8)
pred[pred > 1] = 1
polygons = Mask(pred).polygons()
# print(len(polygons.points))
# print(polygons.segmentation)

simp_polys = []
for x in polygons.points:
    if x < 5:
        print(len(x))
    shp_poly = Polygon(x)
    shp_poly_simp = shp_poly.simplify(0.05, preserve_topology=True)
    simp_polys.append(shp_poly_simp.exterior.coords.xy)
i_poly_simp = Polygons(simp_polys)

img = np.zeros((1024,1024,3)).astype(np.uint8)
polygons.draw(img)
Image.fromarray(img).save("masks_to_polygons/" + selected)


img = np.zeros((1024,1024,3)).astype(np.uint8)
i_poly_simp.draw(img)
Image.fromarray(img).save("masks_to_polygons/simplified_" + selected)