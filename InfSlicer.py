from PIL import Image, ImageDraw
import numpy as np
import os
from inference_sdk import InferenceHTTPClient
from tqdm import tqdm

# === Roboflow setup ===
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="API_KEY_HERE"  # Replace with your API key
)
model_id = "sar-ship-hbhns/1"  # Roboflow public SAR ship model

# === Image and slicing setup ===
input_path = "fullPNG.png"  # Your input image
tile_size = 640
image = Image.open(input_path).convert("RGB")
w, h = image.size

# Final image to draw on
annotated = image.copy()
draw = ImageDraw.Draw(annotated)

# === 1. Loop over tiles
for y in tqdm(range(0, h, tile_size), desc="🧩 Rows"):
    for x in range(0, w, tile_size):
        # Crop tile (pad if necessary)
        tile = image.crop((x, y, min(x+tile_size, w), min(y+tile_size, h)))

        # Save temp tile for Roboflow
        tile_path = "tile_temp.jpg"
        tile.save(tile_path)

        # === 2. Inference via Roboflow
        try:
            result = CLIENT.infer(tile_path, model_id=model_id)
            for pred in result["predictions"]:
                # Box in local (tile) coordinates
                x_center, y_center = pred["x"], pred["y"]
                w_box, h_box = pred["width"], pred["height"]

                # Convert to global (full image) coordinates
                x1 = int(x + x_center - w_box / 2)
                y1 = int(y + y_center - h_box / 2)
                x2 = int(x + x_center + w_box / 2)
                y2 = int(y + y_center + h_box / 2)

                # Draw box on final image
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

        except Exception as e:
            print(f"⚠️ Error in tile ({x},{y}): {e}")
            continue

# === 3. Save final result
annotated.save("annotated_radar_full.png")
print("✅ Full image with detections saved as: annotated_radar_full.png")
