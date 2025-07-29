from noise_filter import apply_correction
from Yan_segmentation import yan_mask
import cv2
import numpy as np

image_path = "vv2.jpg"
image = cv2.imread(image_path)
#      \/ add land segmentation here + remove yan_mask if needed \/

image = yan_mask(image)

print("Starting preprocessing...")

for i in range(1, 2):
    enhanced_image = apply_correction(image)
    image_path ="test_output.png"

cv2.imwrite("test_output.png", image)



