from .noise_filter import apply_correction
from .Land_masking import process_image, compare_images
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "wwww.jpg"

print("Starting preprocessing...")
#      noise reduction and enhancement
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image_new = apply_correction(image, image_path=image_path)

#compare_images(image, image_new)

#land masking
image_mask, mask = process_image(image_new, visualize=False)
#compare_images(image_new, image_mask)


#visualize mask
#compare_images(image, mask)
