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
    


image_path = "Test_image.png"
#Noisy image path
#image_path = "../preprocessing/test.jpg"

print("Starting preprocessing...")
#      noise reduction and enhancement
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

lee_filter, enhance, thresholding, morphing, apply_mask, masked_image, mask_fin = process_image(image, visualize=False, return_steps=True)

# De-noising step
Img_for_inference = apply_correction(masked_image, times=3)


# Save images to preprocessing folder
cv2.imwrite("../preprocessing/Step1_Lee_Filter.png", lee_filter)
cv2.imwrite("../preprocessing/Step2_Enhance.png", enhance)
cv2.imwrite("../preprocessing/Step3_Thresholding.png", thresholding)
cv2.imwrite("../preprocessing/Step4_Morphing.png", morphing)
cv2.imwrite("../preprocessing/Step5_Apply_Mask.png", apply_mask)
cv2.imwrite("../preprocessing/Masked_Image.png", masked_image)
cv2.imwrite("../preprocessing/Completed_Mask.png", mask_fin)

cv2.imwrite("../preprocessing/Final_Image.png", Img_for_inference)