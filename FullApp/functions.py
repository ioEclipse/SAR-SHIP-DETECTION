import json
from PIL import Image, ImageDraw, ImageFont
from tempfile import NamedTemporaryFile
from local_inference import get_local_client
from preprocessing.noise_filter import apply_correction
from preprocessing.Land_masking import process_image, compare_images
import cv2
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from inference_sdk import InferenceHTTPClient
import rasterio
import numpy as np
import os
from datetime import datetime,timedelta
from math import radians, sin, cos, sqrt, asin
import pandas as pd
import requests
from tqdm import tqdm



# === Roboflow setup ===
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="e33E5OuqhaAIPQiyMqLt"
)

MODEL_ID = "sar-ship-hbhns/1"




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
    
    # Convert to JPEG with maximum quality
    Image.fromarray(img_normalized, mode='L').convert("RGB").save(jpg_path, 'JPEG', quality=100)
    return jpg_path

def run_inference_with_crops(uploaded_image, tile_size=640, resolution_m=10):
    
    # Keep path to original tif if provided (used later for geolocation)
    original_tif_path = None

    # Convert and save if the image is a TIFF
    if isinstance(uploaded_image, str) and uploaded_image.lower().endswith('.tif'):
        # preserve original tif path for pixel->lonlat mapping
        original_tif_path = uploaded_image  # CHANGE: preserve TIF for geolocation
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

    # Slice into tiles and process
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image.crop((x, y, min(x+tile_size, w), min(y+tile_size, h)))
            with NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                tile.save(temp_file.name, quality=95)  # Slightly reduced quality for tiles
                temp_path = temp_file.name

            try:
                result = CLIENT.infer(temp_path, model_id=MODEL_ID)
                
                if not result or "predictions" not in result:
                    print(f"Warning: Invalid result from Roboflow API for tile at {x},{y}")
                    continue
                        
                predictions = result.get("predictions", [])
                if not predictions:
                    continue  # No ships detected in this tile
                
                for pred in predictions:
                    x_center_tile, y_center_tile = pred["x"], pred["y"]
                    w_box, h_box = pred["width"], pred["height"]

                    # Convert relative coordinates to absolute (relative to full image)
                    x_center_abs = int(x + x_center_tile)   # CHANGE: absolute center pixel_x
                    y_center_abs = int(y + y_center_tile)   # CHANGE: absolute center pixel_y

                    x1 = int(x + x_center_tile - w_box / 2)
                    y1 = int(y + y_center_tile - h_box / 2)
                    x2 = int(x + x_center_tile + w_box / 2)
                    y2 = int(y + y_center_tile + h_box / 2)

                    # Calculate surface area (pixel area) BEFORE drawing
                    pixel_area = (x2 - x1) * (y2 - y1)

                    # FILTER: if pixel area is strictly greater than 2000, skip this detection
                    if pixel_area > 2000:
                        print(f"Info: Skipping detection at tile {x},{y} with pixel_area {pixel_area} (>2000)")
                        continue  # do not draw, do not add to crops or metadata

                    # Only increment and annotate for accepted ships
                    ship_counter += 1

                    # Annotation (keep as before)
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    draw.text((x1 + 5, y1 + 5), f"Ship #{ship_counter}", fill="yellow", font=font)

                    # Extract detected area (crop)
                    margin = int(max(w_box, h_box) * 0.3)  # Proportional margin
                    crop_x1 = max(x1 - margin, 0)
                    crop_y1 = max(y1 - margin, 0)
                    crop_x2 = min(x2 + margin, w)
                    crop_y2 = min(y2 + margin, h)
                    crop_img = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    crops.append((f"Ship #{ship_counter}", crop_img))

                    # Calculate surface area in m^2
                    area_m2 = pixel_area * (resolution_m ** 2)

                    # --- GEOMETRY / PIXEL COORDS CHANGE ---
                    # CHANGE: replace 'bounding_box' key with pixel center coordinates
                    # keep 'bounding_box' name for UI compatibility but put pixels.
                    pixel_coord = {"pixel_x": x_center_abs, "pixel_y": y_center_abs}

                    # CHANGE: if original image is TIF, calculate geolocation (lon/lat)
                    geoloc = None
                    if original_tif_path:
                        try:
                            lon, lat = pixel_to_lonlat(original_tif_path, x_center_abs, y_center_abs)
                            geoloc = {"lon": float(lon), "lat": float(lat)}
                        except Exception as e:
                            # don't crash UI, simply store None in case of error
                            geoloc = None
                            print(f"[WARN] pixel_to_lonlat failed for a ship: {e}")

                    # Metadata (modified)
                    metadata.append({
                        "ship_id": f"Ship #{ship_counter}",
                        "pixel_area": pixel_area,
                        "surface_m2": round(area_m2, 2),
                        # KEEP name but change content to pixel center for UI compatibility
                        "bounding_box": pixel_coord,   # CHANGE : now contains pixel center coords
                        "geolocation": geoloc          # CHANGE : new key with lon/lat (or None)
                    })

            except Exception as e:
                print(f"Error on tile {x},{y}: {str(e)}")
                continue
            finally:
                try:
                    os.unlink(temp_path)  # Mandatory cleanup
                except Exception:
                    pass

    # Final annotation
    draw.text((10, 10), f"Total Ships Detected: {ship_counter}", fill="cyan", font=font)
    draw.text((10, 40), f"Resolution: {resolution_m}m/pixel", fill="cyan", font=font)

    # Write metadata JSON
    with open("ship_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return annotated, crops, ship_counter, metadata



def find_best_ship(lon, lat, date_iso, ais_csv_path,
                   time_window_s=300, search_radius_m=100,
                   time_weight=0.5, chunksize=200_000):
   
    def haversine_m(lat1, lon1, lat2, lon2):
        R = 6371000.0
        lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        return 2*R*asin(sqrt(a))

    # parse date
    try:
        sentinel_time = datetime.fromisoformat(date_iso)
    except Exception:
        sentinel_time = pd.to_datetime(date_iso)

    tmin = sentinel_time - timedelta(seconds=time_window_s)
    tmax = sentinel_time + timedelta(seconds=time_window_s)

    best_per_mmsi = {}  # MMSI -> (score, row_dict)

    # read by chunks for large file
    for chunk in pd.read_csv(ais_csv_path, parse_dates=["BaseDateTime"],
                             infer_datetime_format=True, chunksize=chunksize, low_memory=True):
        # keep only the time window
        chunk = chunk[(chunk["BaseDateTime"] >= tmin) & (chunk["BaseDateTime"] <= tmax)]
        if chunk.empty:
            continue

        # calculate distance and time delta
        # ensure float conversion for lat/lon
        chunk = chunk.copy()
        chunk["LAT"] = pd.to_numeric(chunk["LAT"], errors="coerce")
        chunk["LON"] = pd.to_numeric(chunk["LON"], errors="coerce")
        chunk = chunk.dropna(subset=["LAT", "LON"])
        if chunk.empty:
            continue

        # distance vector (applied row by row)
        chunk["distance_m"] = chunk.apply(
            lambda r: haversine_m(lat, lon, float(r["LAT"]), float(r["LON"])), axis=1
        )
        # absolute time delta in seconds
        chunk["time_diff_s"] = chunk["BaseDateTime"].apply(lambda t: abs((t - sentinel_time).total_seconds()))

        # can restrict to points within radius (otherwise keep for nearest fallback)
        in_radius = chunk[chunk["distance_m"] <= search_radius_m]
        consider = in_radius if not in_radius.empty else chunk

        # update best per MMSI
        for _, row in consider.iterrows():
            mmsi = row.get("MMSI", None)
            if pd.isna(mmsi):
                continue
            score = float(row["distance_m"]) + time_weight * float(row["time_diff_s"])
            # if new MMSI or better score, replace
            prev = best_per_mmsi.get(mmsi)
            if (prev is None) or (score < prev[0]):
                # store score and entire row (converted to dict for memory efficiency)
                best_per_mmsi[mmsi] = (score, row.to_dict())

    # if no candidate found in entire window
    if not best_per_mmsi:
        return None

    # choose MMSI with best score
    best_mmsi = min(best_per_mmsi.items(), key=lambda kv: kv[1][0])[0]
    best_score, best_row = best_per_mmsi[best_mmsi]

    # enrich row with distance_m/time_diff_s/score fields (if missing)
    best_row["distance_m"] = best_row.get("distance_m", None)
    best_row["time_diff_s"] = best_row.get("time_diff_s", None)
    best_row["score"] = best_score

    # return best row as dict (or pd.Series if you prefer)
    return best_row

def _to_json_serializable(obj):
    """Convertit types pandas/numpy/datetime en types standards JSON-serializables."""
    # pandas Timestamp
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    # numpy scalar
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    # lists/tuples -> recursively convert
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    # dict -> recursively convert
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    # pandas/numpy NaNs -> None
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    # fallback
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

def search_ais_for_metadata(metadata_path="ship_metadata.json",
                            ais_csv_path="AIS_2024_01_24.csv",
                            date_iso="2024-01-24T22:51:07.148377",
                            output_path="AIS_search.json",
                            # below params forwarded to find_best_ship if you want to override:
                            time_window_s=300, search_radius_m=100, time_weight=0.5):
        # Load metadata
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"{metadata_path} not found")

    with open(metadata_path, "r", encoding="utf-8") as f:
        ships = json.load(f)

    results = {}

    total = len(ships)
    print(f"[INFO] {total} ships to process from {metadata_path}")

    for idx, ship in enumerate(ships, start=1):
        ship_id = ship.get("ship_id", f"ship_{idx}")
        geoloc = ship.get("geolocation", None)

        print(f"[{idx}/{total}] Processing {ship_id} ...", end=" ")

        if not geoloc:
            print("no geolocation -> writing null")
            results[ship_id] = None
            continue

        try:
            lon = float(geoloc.get("lon"))
            lat = float(geoloc.get("lat"))
        except Exception as e:
            print(f"bad geolocation -> {e} -> writing null")
            results[ship_id] = None
            continue

        # Call existing function (assume defined)
        try:
            best = find_best_ship(lon, lat, date_iso, ais_csv_path,
                                  time_window_s=time_window_s,
                                  search_radius_m=search_radius_m,
                                  time_weight=time_weight)
        except Exception as e:
            print(f"find_best_ship failed: {e} -> writing null")
            results[ship_id] = None
            continue

        if best is None:
            print("no AIS match")
            results[ship_id] = None
        else:
            # ensure everything is JSON serializable
            serial = _to_json_serializable(best)
            # add ship_id and search origin for traceability
            if isinstance(serial, dict):
                serial["_queried_ship_id"] = ship_id
                serial["_queried_geolocation"] = {"lon": lon, "lat": lat}
                serial["_query_date_iso"] = date_iso
            results[ship_id] = serial
            print("found")

    # Write result
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Results saved to {output_path}")
    return results




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
    # Build URL from config
    base_url = config['ais_data']['base_url']
    url_pattern = config['ais_data']['url_pattern']
    date_str = data_to_str(month,day)
    filename = url_pattern.format(date=date_str)
    url = base_url + filename
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





