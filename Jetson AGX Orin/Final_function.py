from ultralytics import YOLO
import json
from PIL import Image, ImageDraw, ImageFont
from tempfile import NamedTemporaryFile
import cv2
import numpy as np
import rasterio
import os
from scipy.ndimage import uniform_filter


# === Chargement du modèle local ===
LOCAL_MODEL = YOLO("best1.onnx", task="detect")


def gamma_correction(image, gamma=0.5):
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
        enhanced = gamma_correction(enhanced, gamma=1.0)  # More moderate gamma
        if i == 0 : darkened = enhanced.copy() 
        # Apply contrast adjustment (more moderate parameters)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=10/7, beta=0)
        if i == 0 : enlightened = enhanced.copy() 
        
    if return_allsteps:
        return enhanced, darkened, enlightened
    return enhanced

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

def compute_mask(image, invert_mask=False,bull=False,return_steps=False):
    # 1. Load and preprocess
    img = image.copy()
    img = cv2.bilateralFilter( img, d=10, sigmaColor=256, sigmaSpace=75) 
    img = cv2.bilateralFilter( img, d=10, sigmaColor=256, sigmaSpace=75) 
    img = cv2.bilateralFilter( img, d=10, sigmaColor=256, sigmaSpace=75)
    if bull: step_1=img
    # 2. Multi-stage denoising
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    if bull: step_2=enhanced

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
    if bull: step_3=combined
    # 5. Advanced morphological processing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel, iterations=2)
    if bull: step_4=morphed
    # 6. Edge-aware flood filling
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(morphed, mask, (0, 0), 255)  # Fill background
    morphed = cv2.bitwise_not(morphed)

    cv2.floodFill(combined, mask, (0, 0), 255)  # Fill background
    morphed = cv2.bitwise_not(combined)
    
    # 7. Contour filtering (remove small islands)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = h * w * 0.05  # 1% of image area
    land_mask = np.zeros_like(img)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_contour_area:
            cv2.drawContours(land_mask, [cnt], -1, 255, -1)

    # 8. Final refinement
    land_mask = cv2.morphologyEx(land_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    if invert_mask:
        land_mask = cv2.bitwise_not(land_mask)
    if bull: step_5=land_mask
    
    
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

def process_image(image, visualize=True,return_steps=False):
    original_image=image.copy()
    # Load image
    #print(f"Loading image from: {image_path}")
    #original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #if original_image is None:
    #    raise FileNotFoundError(f"Could not load image at {image_path}")

    # Step 1: Apply Lee filter for noise reduction
    print("Applying Lee filter...")
    filtered_image = refined_lee_filter(original_image, window_size=35, k=15)

    # Step 2: Process iteratively to remove land
    current_image = filtered_image.copy()
    masked_image = original_image.copy()
    iteration = 0
    max_iterations = 4

    #create void mask
    mask_fin = np.zeros_like(original_image, dtype=np.uint8)

    while iteration < max_iterations:
        iteration += 1
        print(f"\nIteration {iteration}:")
        
        if iteration == 1: bull=True
        else: bull=False

        # Create land mask
        if return_steps:
            step_1, step_2, step_3, step_4,step_5,land_mask = compute_mask(current_image, invert_mask=True, bull=bull,return_steps=return_steps)
        land_mask = compute_mask(current_image, invert_mask=True, bull=bull,return_steps=return_steps)
        land_percentage = calculate_land_percentage(land_mask)

        print(f"Land percentage detected: {land_percentage:.2f}%")

        # Check if image is mostly land
        if land_percentage > 85:
            print("⚠️  WARNING: Image is almost completely land (>90%)!")
            print("   This image is not suitable for water body analysis.")
            print("   Consider using a different image.")

            black_img = create_black_image_like(original_image)
            return black_img

        # Remove land areas
        
        masked_image = remove_land_areas(masked_image, land_mask)
        current_image = remove_land_areas(current_image, land_mask)
        current_image= refined_lee_filter(current_image, window_size=15, k=35)
        mask_fin = cv2.bitwise_or(mask_fin, land_mask)
        if visualize:
            compare_images(original_image, masked_image)


        # After first iteration, check if we need to continue
        if land_percentage > 10:
            print(f"Land percentage > 15% ({land_percentage:.2f}%), continuing cleanup...")
            #print("Performing second iteration (minimum requirement)...")
            continue
        elif iteration <= max_iterations and land_percentage > 15:
            print(f"Land percentage still > 15% ({land_percentage:.2f}%), continuing cleanup...")
            continue
        else:
            print(f"Land percentage acceptable ({land_percentage:.2f}%), stopping cleanup.")
            break
    buffer_radius = 10  # pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_radius, buffer_radius))
    mask_fin = cv2.dilate(mask_fin, kernel, iterations=1)
    masked_image=remove_land_areas(masked_image, mask_fin)
    if return_steps:
        return step_1, step_2, step_3, step_4,step_5, masked_image, mask_fin
    
    return masked_image, mask_fin



def pixel_to_lonlat(tif_path, x_pixel, y_pixel):
    """Fallback si pixel_to_lonlat non fourni"""
    with rasterio.open(tif_path) as src:
        lon, lat = rasterio.transform.xy(src.transform, y_pixel, x_pixel)
    return lon, lat   

def convert_radar_tif_to_jpg(tif_path, jpg_path):
    """Convertit une image radar TIFF en JPEG avec normalisation adaptée"""
    with rasterio.open(tif_path) as src:
        img_array = src.read(1)
    
    # Normalisation des valeurs radar
    p2, p98 = np.percentile(img_array, (2, 98))
    img_normalized = np.clip((img_array - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
    
    # Conversion en JPEG avec qualité maximale
    Image.fromarray(img_normalized, mode='L').convert("RGB").save(jpg_path, 'JPEG', quality=100)
    return jpg_path

def is_on_land(mask, x1, y1, x2, y2, threshold=0.5):
    """Vérifie si une bounding box est principalement sur terre"""
    # S'assurer que les coordonnées sont dans les limites de l'image
    x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(mask.shape[1], x2)), int(min(mask.shape[0], y2))
    
    if x2 <= x1 or y2 <= y1:
        return False
    
    # Extraire la région correspondant à la bounding box
    region = mask[y1:y2, x1:x2]
    
    if region.size == 0:
        return False
    
    # Calculer le ratio de pixels de terre dans la région
    land_pixels = np.sum(region)
    total_pixels = region.size
    land_ratio = land_pixels / total_pixels
    
    return land_ratio > threshold

def run_inference_with_crops(uploaded_image, tile_size=640, resolution_m=10, filter_abnormal=True):
    """
    Exécute l'inférence avec découpage en tuiles
    
    Args:
        uploaded_image: Chemin ou objet image
        tile_size: Taille des tuiles pour le découpage
        resolution_m: Résolution spatiale en mètres par pixel
        filter_abnormal: Si True, filtre les détections avec pixel_area <= 110 ou > 2000
    """
    # Keep path to original tif if provided (used later for geolocation)
    original_tif_path = None

    # Conversion et sauvegarde si l'image est un TIFF
    if isinstance(uploaded_image, str) and uploaded_image.lower().endswith('.tif'):
        original_tif_path = uploaded_image
        converted_path = "converted_from_tif.jpg"
        convert_radar_tif_to_jpg(uploaded_image, converted_path)
        image = Image.open(converted_path).convert("RGB")
    else:
        image = Image.open(uploaded_image).convert("RGB")
    
    # === Pré-traitement de l'image avant l'inférence ===
    # Convertir l'image PIL en array OpenCV
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Appliquer uniquement le débruitage pour l'inférence et l'image finale
    denoised_image = apply_correction(gray_image, times=3)
    
    # Obtenir le land mask séparément pour le filtrage
    _, land_mask = process_image(gray_image, visualize=False)
    
    # Convertir l'image débruitée en RGB pour le modèle et l'annotation finale
    denoised_rgb = cv2.cvtColor(denoised_image, cv2.COLOR_GRAY2RGB)
    denoised_image_pil = Image.fromarray(denoised_rgb)
    
    w, h = image.size
    
    # Utiliser l'image DÉBRUITÉE pour l'annotation finale
    annotated = denoised_image_pil.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    crops = []
    metadata = []
    ship_counter = 0
    filtered_land = 0
    filtered_abnormal = 0

    # Découpage en tuiles et traitement
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            # Utilise l'image DÉBRUITÉE pour l'inférence
            tile = denoised_image_pil.crop((x, y, min(x+tile_size, w), min(y+tile_size, h)))
            with NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                tile.save(temp_file.name, quality=95)
                temp_path = temp_file.name

            try:
                # === INFÉRENCE AVEC MODÈLE LOCAL ===
                results = LOCAL_MODEL.predict(
                    temp_path,
                    conf=0.25,
                    imgsz=tile_size,
                    device="cpu",
                    verbose=False
                )
                
                # Traitement des résultats
                predictions = []
                for result in results:
                    for box in result.boxes:
                        x_center, y_center, width, height = box.xywh[0].tolist()
                        predictions.append({
                            "x": x_center,
                            "y": y_center,
                            "width": width,
                            "height": height,
                            "confidence": box.conf.item()
                        })
                
                for pred in predictions:
                    x_center_tile, y_center_tile = pred["x"], pred["y"]
                    w_box, h_box = pred["width"], pred["height"]

                    # Conversion des coordonnées relatives en absolues
                    x_center_abs = int(x + x_center_tile)
                    y_center_abs = int(y + y_center_tile)

                    x1 = int(x + x_center_tile - w_box / 2)
                    y1 = int(y + y_center_tile - h_box / 2)
                    x2 = int(x + x_center_tile + w_box / 2)
                    y2 = int(y + y_center_tile + h_box / 2)
                    
                    # Calcul de la surface en pixels
                    pixel_area = (x2 - x1) * (y2 - y1)

                    # Vérifier si la bounding box est sur terre
                    if is_on_land(land_mask, x1, y1, x2, y2):
                        filtered_land += 1
                        continue  # Ignorer cette détection
                    
                    # Filtrage des valeurs aberrantes si activé
                    if filter_abnormal:
                        if pixel_area <= 110 or pixel_area > 2000:
                            filtered_abnormal += 1
                            continue  # Ignorer cette détection

                    ship_counter += 1

                    # Annotation sur l'image DÉBRUITÉE
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    draw.text((x1 + 5, y1 + 5), f"#{ship_counter}", fill="yellow", font=font)

                    # Extraction de la zone détectée depuis l'image DÉBRUITÉE
                    margin = int(max(w_box, h_box) * 0.3)
                    crop_x1 = max(x1 - margin, 0)
                    crop_y1 = max(y1 - margin, 0)
                    crop_x2 = min(x2 + margin, w)
                    crop_y2 = min(y2 + margin, h)
                    crop_img = denoised_image_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    crops.append((f"#{ship_counter}", crop_img))

                    # Calcul de la surface en m²
                    area_m2 = pixel_area * (resolution_m ** 2)

                    # Géolocalisation
                    pixel_coord = {"pixel_x": x_center_abs, "pixel_y": y_center_abs}
                    geoloc = None
                    
                    if original_tif_path:
                        try:
                            lon, lat = pixel_to_lonlat(original_tif_path, x_center_abs, y_center_abs)
                            geoloc = {"lon": float(lon), "lat": float(lat)}
                        except Exception as e:
                            geoloc = None
                            print(f"[WARN] pixel_to_lonlat failed for ship {ship_counter}: {e}")

                    # Metadata
                    metadata.append({
                        "ship_id": f"Ship #{ship_counter}",
                        "pixel_area": pixel_area,
                        "surface_m2": round(area_m2, 2),
                        "bounding_box": pixel_coord,
                        "geolocation": geoloc
                    })

            except Exception as e:
                print(f"Erreur sur la tuile {x},{y}: {str(e)}")
                continue
            finally:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    # Annotation finale sur l'image DÉBRUITÉE
    draw.text((10, 10), f"Total Ships Detected: {ship_counter}", fill="cyan", font=font)
    draw.text((10, 40), f"Filtered (land): {filtered_land}", fill="cyan", font=font)
    draw.text((10, 70), f"Filtered (abnormal): {filtered_abnormal}", fill="cyan", font=font)
    draw.text((10, 100), f"Résolution: {resolution_m}m/pixel", fill="cyan", font=font)

    # Write metadata JSON
    with open("ship_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return annotated, crops, ship_counter, metadata

if __name__ == "__main__":
    # appel identique qu'avant — si tu passes un TIFF, la geolocation sera ajoutée
    annotated_img, crops, count, metadata = run_inference_with_crops("Test_image.png", tile_size=640, resolution_m=10)
    print("Detected ships:", count)
    # affiche les 1ères métadonnées
    for m in metadata[:1]:
        print(m)

        # Convertir PIL → NumPy (BGR pour OpenCV)
    annotated_img_cv = cv2.cvtColor(np.array(annotated_img), cv2.COLOR_RGB2BGR)

    # Sauvegarder en PNG
    cv2.imwrite("annotated_image2.png", annotated_img_cv)
    