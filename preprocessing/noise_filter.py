import cv2
import numpy as np

def gamma_correction(image, gamma=1.0):
    # Build lookup table
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255
                      for i in np.arange(256)]).astype("uint8")
    # Apply gamma correction using LUT
    return cv2.LUT(image, table)

def apply_correction(image,times=1):
    
    
    if image is None:
        return
    
    # Apply gamma correction (adjust gamma value as needed)
    for _ in range(times):
        enhanced = gamma_correction(image, gamma=0.6)  # More moderate gamma
        # Apply contrast adjustment (more moderate parameters)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=10/7, beta=0)
    
    # Apply Gaussian blur to reduce noise
    
    
    # Save and show the result
    
    
    cv2.imshow('Enhanced Image', enhanced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return enhanced

# Load an image (use a relative path or make sure the path exists)

#remove the comment below to test the function:
import os

path = "ts.png"
print("Exists?", os.path.exists(path))
img = cv2.imread(path)
print("Loaded?", img is not None)
image=cv2.imread("ts.png")
apply_correction(image, times=2)

