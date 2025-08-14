from ultralytics import YOLO
import json
from PIL import Image, ImageDraw, ImageFont
from tempfile import NamedTemporaryFile
import cv2
import numpy as np
import rasterio
import os


# === Chargement du modèle local ===
LOCAL_MODEL = YOLO("best.onnx",task="detect")  # Modèle local
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

def run_inference_with_crops(uploaded_image, tile_size=640, resolution_m=10):
    
    # Keep path to original tif if provided (used later for geolocation)
    original_tif_path = None

    # Conversion et sauvegarde si l'image est un TIFF
    if isinstance(uploaded_image, str) and uploaded_image.lower().endswith('.tif'):
        # preserve original tif path for pixel->lonlat mapping
        original_tif_path = uploaded_image
        converted_path = "converted_from_tif.jpg"
        convert_radar_tif_to_jpg(uploaded_image, converted_path)
        image = Image.open(converted_path).convert("RGB")
    else:
        image = Image.open(uploaded_image).convert("RGB")
    
    w, h = image.size
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    crops = []
    metadata = []
    ship_counter = 0

    # Découpage en tuiles et traitement
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image.crop((x, y, min(x+tile_size, w), min(y+tile_size, h)))
            with NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                tile.save(temp_file.name, quality=95)
                temp_path = temp_file.name

            try:
                # === INFÉRENCE AVEC MODÈLE LOCAL ===
                results = LOCAL_MODEL.predict(
                    temp_path,
                    conf=0.25,          # Seuil de confiance
                    imgsz=tile_size,     # Taille d'inférence
                    device="cpu",        # Utiliser "cuda" pour GPU
                    verbose=False        # Désactiver les logs
                )
                
                # Traitement des résultats
                predictions = []
                for result in results:
                    for box in result.boxes:
                        # Extraction des coordonnées (format XYWH)
                        x_center, y_center, width, height = box.xywh[0].tolist()
                        predictions.append({
                            "x": x_center,
                            "y": y_center,
                            "width": width,
                            "height": height,
                            "confidence": box.conf.item()
                        })
                
                # === FIN DE LA SECTION MODIFIÉE ===
                
                for pred in predictions:
                    ship_counter += 1
                    x_center_tile, y_center_tile = pred["x"], pred["y"]
                    w_box, h_box = pred["width"], pred["height"]

                    # Conversion des coordonnées relatives en absolues
                    x_center_abs = int(x + x_center_tile)
                    y_center_abs = int(y + y_center_tile)

                    x1 = int(x + x_center_tile - w_box / 2)
                    y1 = int(y + y_center_tile - h_box / 2)
                    x2 = int(x + x_center_tile + w_box / 2)
                    y2 = int(y + y_center_tile + h_box / 2)

                    # Annotation
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    draw.text((x1 + 5, y1 + 5), f"Ship #{ship_counter}", fill="yellow", font=font)

                    # Extraction de la zone détectée
                    margin = int(max(w_box, h_box) * 0.3)
                    crop_x1 = max(x1 - margin, 0)
                    crop_y1 = max(y1 - margin, 0)
                    crop_x2 = min(x2 + margin, w)
                    crop_y2 = min(y2 + margin, h)
                    crop_img = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    crops.append((f"Ship #{ship_counter}", crop_img))

                    # Calcul de la surface
                    pixel_area = (x2 - x1) * (y2 - y1)
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

    # Annotation finale
    draw.text((10, 10), f"Total Ships Detected: {ship_counter}", fill="cyan", font=font)
    draw.text((10, 40), f"Résolution: {resolution_m}m/pixel", fill="cyan", font=font)

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
    cv2.imwrite("annotated_image.png", annotated_img_cv)
    
    