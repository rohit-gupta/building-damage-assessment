from PIL import Image
from glob import glob
import numpy as np
import sys 

match_path = "test_results/" + sys.argv[1] +"/leaderboard_targets/*.png"
print(match_path)
image_files = glob(match_path)

for image_file in image_files:
    v_image = Image.open(image_file)
#    v_image.verify()
    np_image = np.asarray(v_image)
    shape, dtype = np_image.shape, np_image.dtype
    max = np.max(np_image)
    min = np.min(np_image)
    if max > 4 or min < 0 or (shape != (1024,1024)):
        print(max, min, shape)

