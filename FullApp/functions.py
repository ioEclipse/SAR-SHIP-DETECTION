import json
from PIL import Image, ImageDraw, ImageFont
from inference_sdk import InferenceHTTPClient
from tempfile import NamedTemporaryFile

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

