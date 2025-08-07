import json
import os
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import tempfile
import shutil
import numpy as np
from geopy.distance import geodesic
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient
from tempfile import NamedTemporaryFile

# Note: docker environment isn't set up to get API keys from environment variables.
# Will be done in the next commit.
# ================================================
# FOR ENVIRONMENT SETUP USE os.getenv('<ENV_VAR>')
# ================================================
# 


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


# ==========================================
# AIS DATA MANAGEMENT FUNCTIONS (NOAA 2024)
# ==========================================

def download_noaa_ais_data(start_date: str, end_date: str, bbox: Tuple[float, float, float, float] = None, 
                          cache_dir: str = "./ais_cache") -> List[str]:
    """
    Download NOAA AIS 2024 data for specified date range.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format  
        bbox: Optional bounding box (min_lat, min_lon, max_lat, max_lon)
        cache_dir: Directory to cache downloaded files
        
    Returns:
        List of paths to downloaded/cached AIS CSV files
        
    TODO: Implement NOAA API integration for downloading daily AIS zip files
    TODO: Add spatial filtering based on bbox
    TODO: Handle zip file extraction and CSV parsing
    """
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate date range
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    date_list = []
    current = start
    while current <= end:
        date_list.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    downloaded_files = []
    
    for date_str in date_list:
        # Check if already cached
        cached_file = os.path.join(cache_dir, f"ais_{date_str}.csv")
        if os.path.exists(cached_file):
            downloaded_files.append(cached_file)
            continue
            
        # TODO: Implement actual NOAA download
        # Example URL format (needs verification):
        # noaa_url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/{date_str.replace('-', '_')}.zip"
        
        print(f"TODO: Download AIS data for {date_str}")
        # For now, create placeholder
        with open(cached_file, 'w') as f:
            f.write("MMSI,BaseDateTime,LAT,LON,SOG,COG,Heading,VesselName,VesselType,Length,Width\n")
        downloaded_files.append(cached_file)
    
    return downloaded_files


def cleanup_ais_cache(cache_dir: str = "./ais_cache", max_files: int = 50) -> None:
    """
    Clean up old AIS cache files when storage limit is reached.
    
    Args:
        cache_dir: Directory containing cached AIS files
        max_files: Maximum number of files to keep
        
    TODO: Implement LRU-based cleanup strategy
    TODO: Add size-based cleanup (e.g., keep last 1GB of data)
    """
    if not os.path.exists(cache_dir):
        return
        
    files = [f for f in os.listdir(cache_dir) if f.endswith('.csv')]
    
    if len(files) <= max_files:
        return
        
    # Sort by modification time, keep most recent
    file_paths = [os.path.join(cache_dir, f) for f in files]
    file_paths.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Remove oldest files
    for file_path in file_paths[max_files:]:
        try:
            os.remove(file_path)
            print(f"Removed old AIS cache file: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"Error removing {file_path}: {e}")


def load_ais_data_from_cache(ais_files: List[str], bbox: Tuple[float, float, float, float] = None) -> pd.DataFrame:
    """
    Load and combine AIS data from multiple cached CSV files.
    
    Args:
        ais_files: List of paths to AIS CSV files
        bbox: Optional spatial filter (min_lat, min_lon, max_lat, max_lon)
        
    Returns:
        Combined pandas DataFrame with AIS records
        
    TODO: Optimize for large files (chunked reading)
    TODO: Add data validation and cleaning
    TODO: Handle different CSV formats from NOAA
    """
    all_data = []
    
    for file_path in ais_files:
        try:
            # TODO: Adjust column names based on actual NOAA format
            df = pd.read_csv(file_path)
            
            # Spatial filtering
            if bbox is not None:
                min_lat, min_lon, max_lat, max_lon = bbox
                df = df[
                    (df['LAT'] >= min_lat) & 
                    (df['LAT'] <= max_lat) &
                    (df['LON'] >= min_lon) & 
                    (df['LON'] <= max_lon)
                ]
            
            all_data.append(df)
            
        except Exception as e:
            print(f"Error loading AIS file {file_path}: {e}")
            continue
    
    if not all_data:
        return pd.DataFrame()
        
    return pd.concat(all_data, ignore_index=True)


def match_sar_detections_with_ais(sar_detections: List[Dict], ais_data: pd.DataFrame, 
                                 sar_timestamp: datetime, spatial_threshold_m: float = 500.0,
                                 temporal_threshold_hours: float = 1.0) -> List[Dict]:
    """
    Match SAR ship detections with AIS records using spatial-temporal correlation.
    
    Args:
        sar_detections: List of SAR detection dictionaries with lat/lon coordinates
        ais_data: DataFrame containing AIS records
        sar_timestamp: Timestamp of SAR image acquisition
        spatial_threshold_m: Maximum distance for spatial matching (meters)
        temporal_threshold_hours: Maximum time difference for temporal matching (hours)
        
    Returns:
        List of matched detection dictionaries with AIS information added
        
    TODO: Implement sophisticated matching algorithm
    TODO: Add confidence scoring for matches
    TODO: Handle vessel movement prediction
    """
    if ais_data.empty:
        # Return detections marked as potential dark vessels
        return [{
            **detection,
            'ais_match': None,
            'is_dark_vessel': True,
            'match_confidence': 0.0
        } for detection in sar_detections]
    
    matched_detections = []
    
    for detection in sar_detections:
        # TODO: Extract lat/lon from detection (needs SAR coordinate conversion)
        # For now, using placeholder coordinates
        sar_lat = detection.get('latitude', 0.0)  # TODO: Convert from pixel to geo coordinates
        sar_lon = detection.get('longitude', 0.0)
        
        best_match = None
        best_distance = float('inf')
        best_time_diff = float('inf')
        
        for _, ais_record in ais_data.iterrows():
            try:
                # Spatial distance calculation
                ais_pos = (ais_record['LAT'], ais_record['LON'])
                sar_pos = (sar_lat, sar_lon)
                distance_m = geodesic(sar_pos, ais_pos).meters
                
                # Temporal distance calculation
                ais_time = pd.to_datetime(ais_record['BaseDateTime'])
                time_diff_hours = abs((sar_timestamp - ais_time).total_seconds()) / 3600
                
                # Check thresholds
                if distance_m <= spatial_threshold_m and time_diff_hours <= temporal_threshold_hours:
                    if distance_m < best_distance:
                        best_match = ais_record
                        best_distance = distance_m
                        best_time_diff = time_diff_hours
                        
            except Exception as e:
                continue
        
        # Calculate match confidence
        if best_match is not None:
            # Confidence based on spatial and temporal proximity
            spatial_confidence = max(0, 1 - (best_distance / spatial_threshold_m))
            temporal_confidence = max(0, 1 - (best_time_diff / temporal_threshold_hours))
            match_confidence = (spatial_confidence + temporal_confidence) / 2
            
            matched_detection = {
                **detection,
                'ais_match': {
                    'mmsi': best_match['MMSI'],
                    'vessel_name': best_match.get('VesselName', 'Unknown'),
                    'vessel_type': best_match.get('VesselType', 'Unknown'),
                    'length': best_match.get('Length', 0),
                    'width': best_match.get('Width', 0),
                    'speed': best_match.get('SOG', 0),
                    'course': best_match.get('COG', 0),
                    'distance_m': round(best_distance, 2),
                    'time_diff_hours': round(best_time_diff, 2)
                },
                'is_dark_vessel': False,
                'match_confidence': round(match_confidence, 3)
            }
        else:
            # No AIS match found - potential dark vessel
            matched_detection = {
                **detection,
                'ais_match': None,
                'is_dark_vessel': True,
                'match_confidence': 0.0
            }
        
        matched_detections.append(matched_detection)
    
    return matched_detections


def process_ais_for_timeframe_and_location(start_date: str, end_date: str, 

                                          bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    """
    Complete AIS processing workflow for a specific timeframe and location.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        bbox: Bounding box (min_lat, min_lon, max_lat, max_lon)
        
    Returns:
        Processed AIS DataFrame ready for matching
        
    TODO: Add progress reporting for long downloads
    TODO: Implement parallel processing for multiple files
    """
    # Download/cache AIS data
    ais_files = download_noaa_ais_data(start_date, end_date, bbox)
    
    # Clean up old cache files
    cleanup_ais_cache()
    
    # Load and filter AIS data
    ais_data = load_ais_data_from_cache(ais_files, bbox)
    
    # TODO: Add data quality checks and cleaning
    # TODO: Remove duplicate records
    # TODO: Filter by vessel types of interest
    
    return ais_data

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
    