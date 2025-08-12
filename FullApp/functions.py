import json
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient
from tempfile import NamedTemporaryFile
import numpy as np

# === Roboflow setup ===
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="e33E5OuqhaAIPQiyMqLt"
)
MODEL_ID = "sar-ship-hbhns/1"

def run_inference_with_crops(uploaded_image, tile_size=640, resolution_m=10):
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

    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = image.crop((x, y, min(x+tile_size, w), min(y+tile_size, h)))
            with NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                tile.save(temp_file.name)
                temp_path = temp_file.name

            try:
                result = CLIENT.infer(temp_path, model_id=MODEL_ID)
                for pred in result["predictions"]:
                    ship_counter += 1
                    x_center, y_center = pred["x"], pred["y"]
                    w_box, h_box = pred["width"], pred["height"]

                    # Global image coordinates
                    x1 = int(x + x_center - w_box / 2)
                    y1 = int(y + y_center - h_box / 2)
                    x2 = int(x + x_center + w_box / 2)
                    y2 = int(y + y_center + h_box / 2)

                    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                    draw.text((x1 + 5, y1 + 5), f"Ship #{ship_counter}", fill="yellow", font=font)

                    # Subimage crop
                    margin = 20
                    crop_x1 = max(x1 - margin, 0)
                    crop_y1 = max(y1 - margin, 0)
                    crop_x2 = min(x2 + margin, w)
                    crop_y2 = min(y2 + margin, h)
                    crop_img = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                    crops.append((f"Ship #{ship_counter}", crop_img))

                    # Surface calculation
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
                continue

    draw.text((10, 10), f"Total Ships Detected: {ship_counter}", fill="cyan", font=font)

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


import requests
from tqdm import tqdm  # pip install tqdm
import sys
import os
def get_ais_data(month,day,bar_func=None):
    if(day < 10):
        day = "0"+str(day)
    if(month < 10):
        month = "0"+str(month)
    
    url = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/AIS_2024_" +str(month)+ "_" +str(day)+ ".zip"
    local_filename = "Ais_data/"+str(month)+ "_" +str(day)+".zip"

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
    monthstr = str(month)
    daystr = str(day)
    if(day < 10):
        daystr = "0"+str(day)
    if(month < 10):
        monthstr = "0"+str(month)
    
    if os.path.exists("Ais_data/"+str(monthstr)+ "_" +str(daystr)+".zip"):
        print("File already exists, skipping download.")
    else:
        get_ais_data(month,day)

# check_for_Ais_and_create(11,2)



