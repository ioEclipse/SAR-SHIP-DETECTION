import numpy as np
from skimage.transform import radon
from scipy.stats import entropy
from scipy.ndimage import gaussian_filter, maximum_filter
import cv2

# Step 1: Mean shift filtering (approximated with Gaussian blur for demo)
def mean_shift_filtering_approx(image, spatial_radius=21):
    filtered = gaussian_filter(image, sigma=spatial_radius/5)
    return filtered

# Step 2: Normalization
def normalize_image(img):
    J_t = np.mean(img)
    u = np.std(img)
    f = (img - J_t) / u
    return f

# Step 3: Localized Radon transform
def localized_radon(image, window_size=100, overlap=50, theta=None):
    if theta is None:
        theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    h, w = image.shape
    radon_coeffs = []
    positions = []
    step = window_size - overlap
    for y in range(0, h - window_size + 1, step):
        for x in range(0, w - window_size + 1, step):
            window = image[y:y + window_size, x:x + window_size]
            radon_trans = radon(window, theta=theta, circle=False)
            radon_coeffs.append(radon_trans)
            positions.append((x, y))
    return radon_coeffs, positions, theta

# Step 4: Locally adaptive peak detection
def detect_peaks_local(radon_coeffs, M=1.5, N=1.0, window_size=15):
    detected_peaks = []
    for rt in radon_coeffs:
        J_t = np.mean(rt)
        u = np.std(rt)
        thresh = M * J_t + N * u
        local_max = maximum_filter(rt, size=window_size) == rt
        peaks = np.where((rt > thresh) & local_max)
        detected_peaks.append(peaks)
    return detected_peaks

# Step 5: Shannon entropy computation
def shannon_entropy(gray_values):
    hist, _ = np.histogram(gray_values, bins=256, range=(0, 255), density=True)
    hist = hist[hist > 0]
    ent = -np.sum(hist * np.log2(hist))
    return ent

# Main pipeline
def enhance_ship_wakes_sar(image, M=1.5, N=1.0):
    filtered = mean_shift_filtering_approx(image)
    norm_img = normalize_image(filtered.astype(np.float32))
    radon_coeffs, positions, theta = localized_radon(norm_img)
    detected_peaks = detect_peaks_local(radon_coeffs, M=M, N=N)
    true_wakes = []
    for idx, peaks in enumerate(detected_peaks):
        x, y = positions[idx]
        window = image[y:y + 100, x:x + 100]
        peaks_filtered = []
        for r, c in zip(peaks[0], peaks[1]):
            if r < window.shape[0] and c < window.shape[1]:
                gray_vals = window[:, c]
                ent = shannon_entropy(gray_vals)
                if ent < 4.0:  # Threshold to be tuned
                    peaks_filtered.append((r, c))
        true_wakes.append(peaks_filtered)
    return true_wakes

# Usage:
img= cv2.imread('44.jpg', cv2.IMREAD_GRAYSCALE)
result = enhance_ship_wakes_sar(img)
#save results 
cv2.imwrite('detected_wakes.jpg', np.array(result, dtype=np.uint8))