import os
print(os.getcwd())
import sys
sys.path.append("preprocessing")
from noise_filter import apply_correction
from Land_masking import process_image, compare_images
import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess(image):
    img, mask = process_image(image, visualize=False)
    return apply_correction(img,times=3), mask
    


image_path = "preprocessing/vv2.jpg"

print("Starting preprocessing...")
#      noise reduction and enhancement
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
# image_new = apply_correction(image, times=0)



# #land masking
# image_mask, mask = process_image(image_new, visualize=False)
# compare_images(image_new, image_mask)


# image_final = apply_correction(image_mask, times=3)

# compare_images(image_final, image)
# cv2.imwrite("final.jpg", image_final)
# #visualize mask
image, mask = preprocess(image)
compare_images(image, mask)
