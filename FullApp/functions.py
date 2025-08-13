import json
from PIL import Image, ImageDraw, ImageFont
from tempfile import NamedTemporaryFile
import numpy as np
from local_inference import get_local_client
from preprocessing.noise_filter import apply_correction
from preprocessing.Land_masking import process_image, compare_images
import cv2
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from inference_sdk import InferenceHTTPClient

# === Local YOLO model setup ===
CLIENT = get_local_client()


# === Roboflow setup ===
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="e33E5OuqhaAIPQiyMqLt"
)
MODEL_ID = "sar-ship-hbhns/1"

import rasterio
import numpy as np
import os


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
        
    # Conversion et sauvegarde si l'image est un TIFF
    if isinstance(uploaded_image, str) and uploaded_image.lower().endswith('.tif'):
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
                tile.save(temp_file.name, quality=95)  # Qualité légèrement réduite pour les tuiles
                temp_path = temp_file.name

            try:
                result = CLIENT.infer(temp_path, model_id=MODEL_ID)
                for pred in result["predictions"]:
                    ship_counter += 1
                    x_center, y_center = pred["x"], pred["y"]
                    w_box, h_box = pred["width"], pred["height"]

                    # Conversion des coordonnées relatives en absolues
                    x1 = int(x + x_center - w_box / 2)
                    y1 = int(y + y_center - h_box / 2)
                    x2 = int(x + x_center + w_box / 2)
                    y2 = int(y + y_center + h_box / 2)

                    # Annotation
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    draw.text((x1 + 5, y1 + 5), f"Ship #{ship_counter}", fill="yellow", font=font)

                    # Extraction de la zone détectée
                    margin = int(max(w_box, h_box) * 0.3)  # Marge proportionnelle
                    crop_x1 = max(x1 - margin, 0)
                    crop_y1 = max(y1 - margin, 0)
                    crop_x2 = min(x2 + margin, w)
                    crop_y2 = min(y2 + margin, h)
                    crop_img = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    crops.append((f"Ship #{ship_counter}", crop_img))

                    # Calcul de la surface
                    pixel_area = (x2 - x1) * (y2 - y1)
                    area_m2 = pixel_area * (resolution_m ** 2)

                    # Metadata
                    metadata.append({
                        "ship_id": f"Ship #{ship_counter}",
                        "pixel_area": pixel_area,
                        "surface_m2": round(area_m2, 2),
                        "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    })


            except Exception as e:
                print(f"Erreur sur la tuile {x},{y}: {str(e)}")
                continue
            finally:
                os.unlink(temp_path)  # Nettoyage obligatoire

    # Annotation finale
    draw.text((10, 10), f"Total Ships Detected: {ship_counter}", fill="cyan", font=font)
    draw.text((10, 40), f"Résolution: {resolution_m}m/pixel", fill="cyan", font=font)

   # Write metadata JSON
    with open("ship_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return annotated, crops, ship_counter, metadata

def get_Cords_of_ship(bounding_box, resolution_m_per_px,img_longitude,img_latitude):
    x,y,wx,wy,= bounding_box
    center_x = (x + wx) / 2
    center_y = (y + wy) / 2
    mx= center_x * resolution_m_per_px
    my = center_y * resolution_m_per_px
    latitude = img_latitude + (my / 111133)  # Approximation for latitude
    longtitude = img_longitude + (mx / (111188 * np.cos(np.radians(latitude))))  # Approximation for longitude
    
    return latitude, longtitude

def get_nearest_ship_from_ais(ships, ais_data,min_distance=1000):
    nearest_ship = np.zeros(len(ships), dtype=int)
    
    print(len(ships), len(ais_data))
    for s in range(0,len(ships)):
        nearest_ais = 10000
        ship_coords = (ships[s][0], ships[s][1])
        for i in range(0,len(ais_data)):
            ais_coords = (ais_data[i][0], ais_data[i][1])
            distance = np.linalg.norm(np.array(ship_coords) - np.array(ais_coords))
            print(distance)
            if distance < nearest_ais:
                if distance < min_distance:
                    nearest_ais = distance
                    nearest_ship[s] = i
                else: 
                    nearest_ship[s] = None
    
    return nearest_ship

# =======================
# Preprocessing functions
# =======================
def process_single_image(self, image_path: str, 
                           session_id: str) -> Dict[str, Any]:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")

        image_new = apply_correction(image, image_path=image_path)
        # Preprocess image (land masking)
        masked_image, land_mask = process_image(image, visualize=False)

        return {
            "masked_image": masked_image,
            "land_mask": land_mask,
           }

def process_image_sequence(self, image_paths: List[str], session_id: str) -> Dict[str, Any]:
        results = []
        for image_path in image_paths:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                results.append({
                    "image_path": image_path,
                    "error": f"Could not load image at {image_path}"
                })
                continue

            # Optional: apply noise correction if needed
            image_new = apply_correction(image, image_path=image_path)

            # Preprocess image (land masking)
            masked_image, land_mask = process_image(image, visualize=False)

            results.append({
                "image_path": image_path,
                "masked_image": masked_image,
                "land_mask": land_mask
            })

        return {
            "session_id": session_id,
            "results": results
        }

# =========================
# Downloading AIS data from NOAA
# =========================

import requests
from tqdm import tqdm  # pip install tqdm
import sys
import os
def data_to_str(month,day):
    if(day < 10):
        day = "0"+str(day)
    if(month < 10):
        month = "0"+str(month)
    return str(month) + "_" + str(day)



def get_downloadlist():
    folder_path = "../Ais_data"
    files = os.listdir(folder_path)
    dl=[]
    print(files)
    for file in files:
        if file.endswith(".zip"):
            month,day = file.split("_")
            day = day.split(".")[0]
            month = int(month)
            day = int(day)
            dl.append((month,day))
    return dl
            
def get_storage_for_ais_used():
    total_size = 0
    for file in os.listdir("../Ais_data"):
        total_size +=os.path.getsize("../Ais_data/"+file)
    return total_size / (1024 ** 3)
print("Gb", get_storage_for_ais_used())
####### \/ this somehow needs to be ran in the beginning of the program so that only the oldest files get deleted
download_list=get_downloadlist()
print(download_list)
####### /\ without the print ofc
def get_ais_data(month,day,bar_func=None):
    url = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/AIS_2024_" +data_to_str(month,day)+ ".zip"
    local_filename = "../Ais_data/"+str(month)+ "_" +str(day)+".zip"

    # Send request with streaming enabled
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))  # total size in bytes
        block_size = 1024  # 1 KB chunks
        
        # Create a progress bar
        with open(local_filename, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc=local_filename
        ) as bar:
            for data in r.iter_content(block_size):
                f.write(data)
                
                if bar_func is None: 
                    bar.update(len(data))
                else:
                    bar_func(len(data))
    
    print("Download complete!")

def check_for_Ais_and_create(month,day):
    delete_old_ais_files()
    if os.path.exists("../Ais_data/"+data_to_str(month,day)+".zip"):
        print("File already exists, skipping download.")
    else:
        get_ais_data(month,day)
    download_list.append((month,day))

# check_for_Ais_and_create(11,2)


def delete_old_ais_files():
    if get_storage_for_ais_used() > 4.5:
        os.remove("../Ais_data/"+data_to_str(download_list[0][0],download_list[0][1])+".zip")
        download_list.pop(0)
    return





