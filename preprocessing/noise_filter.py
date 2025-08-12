import cv2
import numpy as np

def gamma_correction(image, gamma=1.0):
    # Build lookup table
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255
                      for i in np.arange(256)]).astype("uint8")
    # Apply gamma correction using LUT
    return cv2.LUT(image, table)

def apply_correction(image,times=1,return_allsteps=False):
    
    
    if image is None:
        return
    
    # Apply gamma correction (adjust gamma value as needed)
    enhanced = image
    for i in range(times):
        enhanced = gamma_correction(enhanced, gamma=0.5)  # More moderate gamma
        if i == 0 : darkened = enhanced.copy() 
        # Apply contrast adjustment (more moderate parameters)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=10/7, beta=0)
        if i == 0 : enlightened = enhanced.copy() 
        
    
    # Apply Gaussian blur to reduce noise
    
    
    # Save and show the result
    
    
    # cv2.imshow('Enhanced Image', enhanced)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if return_allsteps:
        return enhanced, darkened, enlightened
    return enhanced

# Load an image (use a relative path or make sure the path exists)

#remove the comment below to test the function:
import os



# path = "preprocessing/test.jpg"
# print("Exists?", os.path.exists(path))
# img = cv2.imread(path)
# print("Loaded?", img is not None)
# image=cv2.imread(path)
# apply_correction(image, times=3)

