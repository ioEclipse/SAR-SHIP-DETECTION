"""
Image Slicer for Large SAR Image Inference

This utility processes large SAR images by dividing them into smaller tiles,
running YOLO inference on each tile, and combining the results into a single
annotated image with ship detections.

Usage:
    from utilities.InfSlicer import process_large_image
    
    detections, annotated_image = process_large_image(
        input_path="large_sar_image.png",
        output_path="detections.png",
        tile_size=640
    )
"""

from PIL import Image, ImageDraw
import numpy as np
import os
import sys
import tempfile
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional

# Add FullApp directory to path for local_inference import
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'FullApp'))
from local_inference import get_local_client


def process_large_image(
    input_path: str, 
    output_path: Optional[str] = None,
    tile_size: int = 640,
    show_progress: bool = True
) -> Tuple[List[Dict], Image.Image]:
    """
    Process a large SAR image by slicing it into tiles and running inference.
    
    Args:
        input_path (str): Path to the input SAR image
        output_path (str, optional): Path to save annotated image. If None, doesn't save.
        tile_size (int): Size of each tile for processing (default: 640)
        show_progress (bool): Whether to show progress bar (default: True)
    
    Returns:
        Tuple[List[Dict], Image.Image]: List of detections and annotated image
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")
    
    # Load and prepare image
    image = Image.open(input_path).convert("RGB")
    w, h = image.size
    
    # Create annotated copy
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    
    # Initialize client
    client = get_local_client()
    all_detections = []
    
    # Create temporary file for tiles
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Process tiles
        total_tiles = ((h - 1) // tile_size + 1) * ((w - 1) // tile_size + 1)
        progress_desc = "🧩 Processing tiles" if show_progress else None
        
        with tqdm(total=total_tiles, desc=progress_desc, disable=not show_progress) as pbar:
            for y in range(0, h, tile_size):
                for x in range(0, w, tile_size):
                    # Crop tile
                    tile = image.crop((
                        x, y, 
                        min(x + tile_size, w), 
                        min(y + tile_size, h)
                    ))
                    
                    # Save temporary tile
                    tile.save(temp_path)
                    
                    # Run inference
                    try:
                        result = client.infer(temp_path)
                        
                        for pred in result.get("predictions", []):
                            # Convert tile coordinates to global coordinates
                            x_center, y_center = pred["x"], pred["y"]
                            w_box, h_box = pred["width"], pred["height"]
                            
                            # Global coordinates
                            global_x1 = int(x + x_center - w_box / 2)
                            global_y1 = int(y + y_center - h_box / 2)
                            global_x2 = int(x + x_center + w_box / 2)
                            global_y2 = int(y + y_center + h_box / 2)
                            
                            # Store detection
                            detection = {
                                "x1": global_x1,
                                "y1": global_y1,
                                "x2": global_x2,
                                "y2": global_y2,
                                "confidence": pred.get("confidence", 0.0),
                                "class": pred.get("class", "ship")
                            }
                            all_detections.append(detection)
                            
                            # Draw on annotated image
                            draw.rectangle([global_x1, global_y1, global_x2, global_y2], 
                                         outline="red", width=2)
                    
                    except Exception as e:
                        if show_progress:
                            tqdm.write(f"⚠️ Error in tile ({x},{y}): {e}")
                    
                    pbar.update(1)
    
    finally:
        # Cleanup temporary file
        try:
            os.unlink(temp_path)
        except OSError:
            pass
    
    # Save result if output path provided
    if output_path:
        annotated.save(output_path)
        if show_progress:
            print(f"✅ Annotated image saved: {output_path}")
    
    if show_progress:
        print(f"🔍 Total detections found: {len(all_detections)}")
    
    return all_detections, annotated


def main():
    """Command-line interface for the image slicer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process large SAR images with YOLO detection")
    parser.add_argument("input", help="Input SAR image path")
    parser.add_argument("-o", "--output", help="Output annotated image path", 
                       default="annotated_result.png")
    parser.add_argument("-t", "--tile-size", type=int, default=640,
                       help="Tile size for processing (default: 640)")
    parser.add_argument("-q", "--quiet", action="store_true",
                       help="Suppress progress output")
    
    args = parser.parse_args()
    
    try:
        detections, annotated_img = process_large_image(
            input_path=args.input,
            output_path=args.output,
            tile_size=args.tile_size,
            show_progress=not args.quiet
        )
        
        print(f"\n📊 Results:")
        print(f"   Input image: {args.input}")
        print(f"   Output image: {args.output}")
        print(f"   Ships detected: {len(detections)}")
        print(f"   Tile size: {args.tile_size}x{args.tile_size}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()