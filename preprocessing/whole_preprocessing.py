from noise_filter import apply_correction
from Yan_segmentation import yan_mask
from Land_masking import process_image
import cv2
import numpy as np

image_path = "hi.png"

print("Starting preprocessing...")
#      \/ add land segmentation here \/

image = process_image(image_path, visualize=True)



for i in range(1, 2):
    enhanced_image = apply_correction(image)
    image_path ="test_output.png"




