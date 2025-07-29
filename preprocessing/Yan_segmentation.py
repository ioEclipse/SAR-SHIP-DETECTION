import cv2
import numpy as np

def gamma_correction(image, gamma=1.0):
    # Build lookup table
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255
                      for i in np.arange(256)]).astype("uint8")
    # Apply gamma correction using LUT
    return cv2.LUT(image, table)



sharpened = cv2.imread("vv.jpg")
img=sharpened

sharpened = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
for i in range(0,9):
 sharpened = cv2.bilateralFilter(sharpened, d=9, sigmaColor=20, sigmaSpace=75)
sharpened = cv2.convertScaleAbs(sharpened, alpha=14, beta=0)




_, sharpened = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY)
sharpened = cv2.bitwise_not(sharpened) 

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
sharpened = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel,iterations=3)
sharpened = cv2.morphologyEx(sharpened, cv2.MORPH_OPEN, kernel, iterations=2)


land_mask = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
result = cv2.bitwise_and(img, img,mask = land_mask)



