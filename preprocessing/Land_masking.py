import numpy as np
import cv2
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
from noise_filter import gamma_correction

def refined_lee_filter(image, window_size=5, k=1.0):

    # Convert to float for calculations
    img = image.astype(np.float32)

    # Step 1: Compute local mean
    mean = uniform_filter(img, size=window_size)

    # Step 2: Compute local variance
    mean_square = uniform_filter(img**2, size=window_size)
    variance = mean_square - mean**2

    # Avoid division by zero
    variance[variance <= 0] = 1e-10

    # Step 3: Compute coefficient of variation
    cv = np.sqrt(variance) / mean

    # Step 4: Compute weighting factor
    weight = 1.0 / (1.0 + k * cv**2)

    # Step 5: Apply filter
    filtered = mean + weight * (img - mean)

    # Clip values to valid range
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)

    return filtered

def compare_images(original, filtered):
    """Display original and filtered images side by side."""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Image 1')
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Image 2')
    plt.imshow(filtered, cmap='gray')
    plt.axis('off')

    plt.show()

def compute_mask(image,combined_masks=0, invert_mask=False,bull=False,return_steps=False):
    # 1. Load and preprocess
    img = image.copy()
    img = cv2.bilateralFilter( img, d=10, sigmaColor=256, sigmaSpace=75) 
    img = cv2.bilateralFilter( img, d=10, sigmaColor=256, sigmaSpace=75) 
    img = cv2.bilateralFilter( img, d=10, sigmaColor=256, sigmaSpace=75)
    if bull or return_steps: step_1=img
    # 2. Multi-stage denoising
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    if bull or return_steps: step_2=enhanced

    # 3. Combined thresholding
    
    _, otsu_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_thresh = cv2.adaptiveThreshold(enhanced, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 21, 5)
    
    # 4. Fusion of thresholding methods
    combined = cv2.bitwise_or(otsu_thresh, adaptive_thresh)
    combined = cv2.bilateralFilter( combined, d=9, sigmaColor=256, sigmaSpace=75) 
    combined = cv2.bilateralFilter( combined, d=9, sigmaColor=256, sigmaSpace=75)
    combined = cv2.bilateralFilter( combined, d=9, sigmaColor=256, sigmaSpace=75)
    _, combined = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bull or return_steps: step_3=combined
    # 5. Advanced morphological processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel, iterations=2)
    if bull or return_steps: step_4=morphed
    # 6. Edge-aware flood filling
    h, w = img.shape[:2]
    # mask = np.zeros((h+2, w+2), np.uint8)
    # cv2.floodFill(morphed, mask, (0, 0), 255)  # Fill background
    # morphed = cv2.bitwise_not(morphed)

    # cv2.floodFill(combined, mask, (0, 0), 255)  # Fill background
    
    morphed = cv2.bitwise_or(morphed, combined_masks)
    # compare_images(morphed,combined_masks)
    morphed = cv2.bitwise_not(morphed)
    # 7. Contour filtering (remove small islands)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 2000  # 1% of image area
    land_mask = np.zeros_like(img)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_contour_area:
            cv2.drawContours(land_mask, [cnt], -1, 255, -1)

    # 8. Final refinement
   # land_mask = cv2.morphologyEx(land_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    if invert_mask:
        land_mask = cv2.bitwise_not(land_mask)
    if bull or return_steps: step_5=land_mask
    
    
    if return_steps:
        return step_1, step_2, step_3, step_4,step_5, land_mask
    
    return land_mask

def calculate_land_percentage(mask):
    """Calculate the percentage of land in the mask"""
    total_pixels = mask.size 
    land_pixels = cv2.countNonZero(mask)
    return (land_pixels / total_pixels) * 100

def remove_land_areas(image, mask):
    """Remove land areas using the provided mask"""
    inverted_mask = cv2.bitwise_not(mask)
    return cv2.bitwise_and(image, image, mask=inverted_mask)

def create_black_image_like(original_image):
    """Create a black image with the same shape and type as the original."""
    return np.zeros_like(original_image)

def process_image(image, visualize=True, return_steps=False):
    original_image = image.copy()
    
    # Step 1: Apply Lee filter
    print("Applying Lee filter...")
    filtered_image = refined_lee_filter(original_image, window_size=35, k=15)

    # Initialize variables for all steps
    step_1 = step_2 = step_3 = step_4 = step_5 = None
    masked_image = original_image.copy()
    mask_fin = np.zeros_like(original_image, dtype=np.uint8)
    
    try:
        if return_steps:
            step_1, step_2, step_3, step_4, step_5, land_mask = compute_mask(
                filtered_image, invert_mask=True, bull=True, return_steps=True
            )
        else:
            land_mask = compute_mask(filtered_image, invert_mask=True)
            
        # Calculate land percentage using the mask
        land_percentage = calculate_land_percentage(land_mask)
        
        # Process the mask
        masked_image = remove_land_areas(masked_image, land_mask)
        mask_fin = cv2.bitwise_or(mask_fin, land_mask)
        
        if return_steps:
            return (step_1, step_2, step_3, step_4, step_5, 
                    masked_image, mask_fin)
        return masked_image, mask_fin
        
    except Exception as e:
        print(f"Error in process_image: {e}")
        if return_steps:
            return (None, None, None, None, None, 
                    original_image, np.zeros_like(original_image))
        return original_image, np.zeros_like(original_image)
    
'''Original_image_path="/content/fullPNG1.png"
Final_image_path = "/content/Final_image.png"

Final_image = process_image(Original_image_path, visualize=True)
if Final_image is not None:
    cv2.imwrite(Final_image_path, Final_image)
    print(f"Final image saved to: {Final_image_path}")

compare_images(cv2.imread(Original_image_path, cv2.IMREAD_GRAYSCALE), Final_image)'''
