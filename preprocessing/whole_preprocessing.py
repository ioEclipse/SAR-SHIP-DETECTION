import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.noise_filter import apply_correction
from preprocessing.Land_masking import process_image, compare_images
import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess(image,return_steps=False):
    
    if return_steps:
        step_1, step_2, step_3, step_4,step_5,img, mask = process_image(image, visualize=False,return_steps=True)
        return step_1, step_2, step_3, step_4, step_5,apply_correction(img,times=3,return_allsteps=True), mask

    img, mask = process_image(image, visualize=False)
    return apply_correction(img,times=3), mask
    


image_path = "FullApp/Test_image.png"

print("Starting preprocessing...")
if __name__ == "__main__":
    image= cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image at {image_path}")
    else:
        enhanced_image, mask = preprocess(image, return_steps=False)
        print("Preprocessing completed.")
        # Save or display the enhanced image
        cv2.imwrite("enhanced_image.png", enhanced_image)
        cv2.imwrite("mask.png", mask)
        print("Enhanced image and mask saved.")