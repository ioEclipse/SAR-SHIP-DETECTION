import json
import os
from typing import List, Tuple
import ee
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from geemap import ee_export_image
from noise_filter import apply_correction
from inference_sdk import InferenceHTTPClient
import json
from tempfile import NamedTemporaryFile

# === Roboflow setup ===
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="e33E5OuqhaAIPQiyMqLt"
)
MODEL_ID = "sar-ship-hbhns/1"

def run_inference(uploaded_image, resolution_m=20):
    # Ouvrir l'image
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

    # Envoyer l'image directement au modèle
    result = CLIENT.infer(uploaded_image, model_id=MODEL_ID)

    for pred in result.get("predictions", []):
        ship_counter += 1
        x_center, y_center = pred["x"], pred["y"]
        w_box, h_box = pred["width"], pred["height"]

        # Coords bounding box
        x1 = int(x_center - w_box / 2)
        y1 = int(y_center - h_box / 2)
        x2 = int(x_center + w_box / 2)
        y2 = int(y_center + h_box / 2)

        # Dessiner la box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1 + 5, y1 + 5), f"{ship_counter}", fill="yellow", font=font)

        # Crop du navire
        margin = 20
        crop_x1 = max(x1 - margin, 0)
        crop_y1 = max(y1 - margin, 0)
        crop_x2 = min(x2 + margin, w)
        crop_y2 = min(y2 + margin, h)
        crop_img = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        crops.append((f"Ship #{ship_counter}", crop_img))

        # Surface en m²
        pixel_area = (x2 - x1) * (y2 - y1)
        area_m2 = pixel_area * (resolution_m ** 2)

        metadata.append({
            "ship_id": f"Ship #{ship_counter}",
            "pixel_area": pixel_area,
            "surface_m2": round(area_m2, 2),
            "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })

    # Sauvegarder les métadonnées
    with open("ship_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    return annotated, crops, ship_counter, metadata

def check_zone_size(coordinates: List[Tuple[float, float]]) -> bool:
    
    lons = [coord[0] for coord in coordinates]
    lats = [coord[1] for coord in coordinates]
    
    lon_span = max(lons) - min(lons)
    lat_span = max(lats) - min(lats)
    
    return lon_span <= 0.4 and lat_span <= 0.4

def extract_coordinates_from_geojson(geojson_path: str) -> List[Tuple[float, float]]:
   
    with open(geojson_path) as f:
        data = json.load(f)
    
    coordinates = data['features'][0]['geometry']['coordinates'][0]
    
    if not check_zone_size(coordinates):
        raise ValueError("Zone trop grande. Sélectionnez une zone inférieure à 0.4°×0.4°")
    
    # Calcul du rectangle englobant
    lons = [coord[0] for coord in coordinates]
    lats = [coord[1] for coord in coordinates]
    
    return [
        [min(lons), min(lats)],
        [max(lons), min(lats)],
        [max(lons), max(lats)],
        [min(lons), max(lats)]
    ]

def get_sentinel1_jpg(polygon_coords, year, month, output_dir='images/sar_sentinel1_jpg', resolution_m=10):
    # Authentification et initialisation
    try:
        ee.Initialize()
    except:
        ee.Authenticate(auth_mode='localhost')
        ee.Initialize(project='eendve-bouazizchahine7')
    
    # Créer les dossiers de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Définir la région d'étude
    region = ee.Geometry.Polygon([polygon_coords])
    
    # Période (mois complet)
    start_date = ee.Date.fromYMD(year, month, 1)
    end_date = start_date.advance(1, 'month')
    
    # Chemins des fichiers
    tif_path = 'temp_sar_image.tif'
    jpg_original_path = os.path.join(output_dir, f'SAR_S1_{year}_{month:02d}_original.jpg')
    jpg_corrected_path = os.path.join(output_dir, f'SAR_S1_{year}_{month:02d}_corrected.jpg')
    jpg_detections_path = os.path.join(output_dir, f'SAR_S1_{year}_{month:02d}_detections.jpg')
    crops_dir = os.path.join(output_dir, 'crops')
    os.makedirs(crops_dir, exist_ok=True)
    
    try:
        # Récupération de l'image Sentinel-1
        print(f"🛰 Récupération de l'image Sentinel-1 pour {year}-{month:02d}...")
        sar = ee.ImageCollection("COPERNICUS/S1_GRD") \
            .filterDate(start_date, end_date) \
            .filterBounds(region) \
            .filter(ee.Filter.eq('instrumentMode', 'IW')) \
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
            .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING')) \
            .select(['VV']) \
            .mean().clip(region)
        
        # Téléchargement du GeoTIFF
        print("⬇️ Téléchargement de l'image GeoTIFF...")
        ee_export_image(
            sar,
            filename=tif_path,
            region=region,
            scale=20,
            file_per_band=False
        )
        
        # Conversion et traitement
        print("🖼 Conversion et traitement...")
        with rasterio.open(tif_path) as src:
            img_array = src.read(1)
        
        # Normalisation
        p2, p98 = np.percentile(img_array, (2, 98))
        img_array = np.clip((img_array - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
        
        # Sauvegarde de l'original
        img_original = Image.fromarray(img_array, mode='L')
        img_original.save(jpg_original_path, 'JPEG', quality=100)
        print(f"✅ Original sauvegardé: {jpg_original_path}")
        
        # Application de la correction de bruit
        try:
            # Conversion pour OpenCV
            cv_image = np.array(img_original)
            corrected_img = apply_correction(cv_image)
            
            if isinstance(corrected_img, np.ndarray):
                img_corrected = Image.fromarray(corrected_img, mode='L')
                img_corrected.save(jpg_corrected_path, 'JPEG', quality=100)
                print(f"✅ Corrigé sauvegardé: {jpg_corrected_path}")
                
                # Détection des navires avec la nouvelle fonction
                print("🚢 Lancement de la détection avancée des navires...")
                try:
                    # Conversion en RGB pour l'inférence
                    img_corrected_rgb = img_corrected.convert('RGB')
                    temp_img_path = os.path.join(output_dir, 'temp_inference.jpg')
                    img_corrected_rgb.save(temp_img_path)
                    
                    # Appel de la fonction avancée
                    annotated_img, crops, ship_count, metadata = run_inference(
                        temp_img_path, 
                        resolution_m=resolution_m
                    )
                    
                    # Sauvegarde des résultats
                    annotated_img.save(jpg_detections_path)
                    print(f"✅ Détections sauvegardées: {jpg_detections_path}")
                    print(f"🔍 {ship_count} navires détectés")
                    
                    # Sauvegarde des crops individuels
                    for ship_id, crop_img in crops:
                        crop_path = os.path.join(crops_dir, f"{ship_id.replace('# ', '').replace(' ', '_')}.jpg")
                        crop_img.save(crop_path)
                    
                    # Sauvegarde des métadonnées
                    metadata_path = os.path.join(output_dir, f'SAR_S1_{year}_{month:02d}_metadata.json')
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                    print(f"📊 Métadonnées sauvegardées: {metadata_path}")
                    
                    # Nettoyage
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)
                        
                    return {
                        'original': jpg_original_path,
                        'corrected': jpg_corrected_path,
                        'detections': jpg_detections_path,
                        'crops_dir': crops_dir,
                        'metadata': metadata_path,
                        'ship_count': ship_count
                    }
                    
                except Exception as detection_error:
                    print(f"⚠️ Erreur lors de la détection avancée: {str(detection_error)}")
                    return {
                        'original': jpg_original_path,
                        'corrected': jpg_corrected_path,
                        'error': str(detection_error)
                    }
            else:
                raise ValueError("apply_correction doit retourner un numpy array")
                
        except Exception as correction_error:
            print(f"⚠️ Correction échouée: {str(correction_error)}")
            return {
                'original': jpg_original_path,
                'error': str(correction_error)
            }
        
    except Exception as e:
        print(f"❌ Erreur globale: {str(e)}")
        return {
            'error': str(e)
        }
    finally:
        # Nettoyage systématique du fichier temporaire
        if os.path.exists(tif_path):
            os.remove(tif_path)

def get_sentinel1_jpg_from_geojson(geojson_path: str, year: int, month: int, 
                                  output_dir: str = 'images/sar_sentinel1_jpg') -> str:
   
    polygon_coords = extract_coordinates_from_geojson(geojson_path)
    return get_sentinel1_jpg(polygon_coords, year, month, output_dir)


# Exemple d'utilisation avec VOTRE fichier GeoJSON
if __name__ == "__main__":
    # Remplacez ce chemin par celui de VOTRE fichier GeoJSON
    votre_geojson = "fichier.geojson"
    
    jpg_path = get_sentinel1_jpg_from_geojson(
        geojson_path=votre_geojson,  # Votre fichier ici
        year=2023,
        month=6
    )
    
    if jpg_path:
        print(f"Image générée avec succès: {jpg_path}")
        
        
    else:
        print("Échec de la génération de l'image")