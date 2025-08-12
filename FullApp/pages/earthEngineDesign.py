import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import tempfile
import json
import os
import pandas as pd
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random

# Alternative to engineAPI1 import - comment out the real one and use mock
# from engineAPI1 import get_sentinel1_jpg_from_geojson

# Mock function to replace engineAPI1
def get_sentinel1_jpg_from_geojson(geojson_path, year, month):
    """
    Mock function that simulates SAR processing and ship detection
    Returns fake results for UI development
    """
    # Simulate processing time
    time.sleep(2)
    
    # Create a mock SAR detection image with bounding boxes
    img_width, img_height = 800, 600
    img = Image.new('RGB', (img_width, img_height), color=(30, 30, 60))  # Dark blue background
    draw = ImageDraw.Draw(img)
    
    # Add some noise to simulate SAR imagery
    for _ in range(5000):
        x, y = random.randint(0, img_width), random.randint(0, img_height)
        color = (random.randint(40, 80), random.randint(40, 80), random.randint(60, 100))
        draw.point((x, y), fill=color)
    
    # Add mock ship detections with bounding boxes
    ship_count = random.randint(3, 12)
    ship_data = []
    
    try:
        # Try to load a default font, fallback to default if not available
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for i in range(ship_count):
        # Random ship position
        x = random.randint(50, img_width - 100)
        y = random.randint(50, img_height - 80)
        width = random.randint(30, 80)
        height = random.randint(15, 40)
        
        # Draw bounding box
        draw.rectangle([x, y, x + width, y + height], outline=(255, 0, 0), width=3)
        
        # Add ship ID label
        draw.text((x, y - 25), f"Ship {i+1}", fill=(255, 255, 255), font=font)
        
        # Create mock metadata for this ship
        ship_data.append({
            "ship_id": f"SHIP_{i+1:03d}",
            "detection_confidence": round(random.uniform(0.7, 0.99), 3),
            "bbox_x": x,
            "bbox_y": y,
            "bbox_width": width,
            "bbox_height": height,
            "estimated_length_m": round(random.uniform(50, 300), 1),
            "estimated_width_m": round(random.uniform(8, 45), 1),
            "ship_type": random.choice(["Cargo", "Tanker", "Fishing", "Container", "Bulk Carrier"]),
            "heading_degrees": round(random.uniform(0, 360), 1),
            "detection_timestamp": f"{year}-{month:02d}-{random.randint(1, 28):02d}T{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00Z",
            "latitude": round(random.uniform(30.0, 33.0), 6),
            "longitude": round(random.uniform(31.0, 34.0), 6)
        })
    
    # Add title to image
    draw.text((20, 20), f"SAR Ship Detection - {year}/{month:02d}", fill=(255, 255, 255), font=font)
    draw.text((20, 60), f"{ship_count} Ships detected", fill=(0, 255, 0), font=font)
    
    # Save detection image to temp file
    detection_img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    img.save(detection_img_path)
    
    # Save metadata to temp JSON file
    metadata_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w")
    json.dump(ship_data, metadata_path, indent=2)
    metadata_path.close()
    
    return {
        "detections": detection_img_path,
        "metadata": metadata_path.name,
        "ship_count": ship_count,
    }

# --- Page config ---
st.set_page_config(page_title="SAR Map Viewer", layout="wide")

# Add custom CSS for larger metric text
st.markdown("""
<style>
    /* Make the ships detected metric text bigger */
    div[data-testid="metric-container"] {
        padding: 1rem;
    }
    div[data-testid="metric-container"] label {
        font-size: 1.5rem !important;
        font-weight: bold !important;
    }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        font-size: 2.5rem !important;
        font-weight: bold !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Filters + Predict + Reset ---
st.sidebar.title("Filters")

year = st.sidebar.selectbox("Select Year", [2022, 2023, 2024, 2025])
month = st.sidebar.selectbox(
    "Select Month",
    list(range(1, 13)),
    format_func=lambda m: [
        "January","February","March","April","May","June",
        "July","August","September","October","November","December"
    ][m-1]
)

# Predict button in sidebar (under the selectors)
predict_clicked = st.sidebar.button("▶️ Predict SAR & Detect Ships", type="primary")

# Reset button (to return to map)
if st.sidebar.button("🔄 Reset"):
    # Clear stored result if exists
    for k in ("result_out", "tmp_geojson_path"):
        if k in st.session_state:
            try:
                # try to remove tempfile if it exists
                if k == "tmp_geojson_path" and st.session_state.get(k):
                    if os.path.exists(st.session_state[k]):
                        os.remove(st.session_state[k])
            except Exception:
                pass
            st.session_state.pop(k, None)
    st.rerun()

# --- Main area ---
# If we already have a result saved in session_state, show the result UI.
if "result_out" in st.session_state and st.session_state["result_out"]:
    out = st.session_state["result_out"]

    ship_count = out.get("ship_count") if isinstance(out, dict) else None
    st.header(f"🚢 Total Ships Detected {ship_count}")
    
    
    # Create columns for better layout
    col1, col2 = st.columns([10, 1])
    
    with col1:
        # Show detection image if present
        if isinstance(out, dict) and out.get("detections") and os.path.exists(out["detections"]):
            st.image(out["detections"], caption="SAR Ship Detections", use_container_width=True)
        else:
            st.error("No detection image found in the result.")
    
    with col2:
        # Show summary statistics
        
            
        
        # Show processing info if available
        processing_info = out.get("processing_info", {})
        if processing_info:
            st.subheader("📊 Processing Details")
            for key, value in processing_info.items():
                formatted_key = key.replace("_", " ").title()
                st.write(f"**{formatted_key}:** {value}")

    st.markdown("---")
    
    # Load and show metadata table
    metadata_path = out.get("metadata") if isinstance(out, dict) else None
    if metadata_path and os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r") as mf:
                metadata_list = json.load(mf)
            df = pd.DataFrame(metadata_list)
            
            meta_title_col, meta_ctrl_col = st.columns([7, 1])
            with meta_title_col:
                st.markdown("### 🧾 Ship Detection Metadata")
            with meta_ctrl_col:
                show_full = st.checkbox("Show full table", value=False, key="show_full_table")

            if show_full:
                st.dataframe(df, use_container_width=True)
            else:
                st.dataframe(df.head(5), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading metadata: {str(e)}")
    else:
        st.info("No metadata file available to display.")
    
    # Moved show_full checkbox next to the title above
    st.markdown("---")

else:
    # No result yet -> show the map and drawing tools
    # Styled header banner for the drawing section
    st.markdown(
        """
        <div style="
            padding: 20px;
            border-radius: 12px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin-bottom: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        ">
            <span style="font-size: 32px;">🗺️</span>
            <div style="line-height: 1.2; text-align: center;">
                <div style="font-size: 28px; font-weight: 700;">Select Area of Interest</div>
                <div style="font-size: 16px; opacity: 0.9;">Draw a polygon on the map to define your search area for SAR ship detection</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Create Folium map with basic tile layer
    center = [31.2, 32.3]  # Mediterranean Sea area
    m = folium.Map(location=center, zoom_start=6)
    
    # Add satellite imagery option if available
    try:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satellite",
            overlay=False,
            control=True
        ).add_to(m)
    except:
        pass  # Skip if there are issues with custom tiles
    
    # Add drawing tools
    draw = Draw(
        export=False,
        draw_options={
            'polyline': False, 
            'rectangle': True, 
            'circle': False,
            'circlemarker': False, 
            'marker': False,
            'polygon': {
                'allowIntersection': False, 
                'showArea': True, 
                'shapeOptions': {'color': '#ff0000', 'fillColor': '#ffff00', 'fillOpacity': 0.2}
            }
        }
    )
    draw.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)

    map_data = st_folium(m, width=1200, height=500, returned_objects=["last_object_clicked_popup", "all_drawings"])

    # Extract polygon GeoJSON from map_data
    geo = None
    if map_data["all_drawings"]:
        geo = map_data["all_drawings"][-1]  # Get the last drawn shape
    
    if geo:
        st.success("✅ Area selected! Use the sidebar to configure detection parameters and start processing.")
        with st.expander("View Selected Area GeoJSON"):
            st.json(geo)

    # If Predict button clicked in the sidebar, process now
    if predict_clicked:
        if not geo:
            st.sidebar.error("❌ No polygon drawn. Please draw a polygon on the map before predicting.")
        else:
            # Wrap polygon into FeatureCollection
            if isinstance(geo, dict) and geo.get("type") == "FeatureCollection":
                wrapped = geo
            elif isinstance(geo, dict) and geo.get("type") == "Feature":
                wrapped = {"type": "FeatureCollection", "features": [geo]}
            else:
                wrapped = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": geo}]}

            # Save temp geojson file
            tmp_geo = tempfile.NamedTemporaryFile(delete=False, suffix=".geojson", mode="w")
            json.dump(wrapped, tmp_geo)
            tmp_geo.close()
            tmp_geo_path = tmp_geo.name
            st.session_state["tmp_geojson_path"] = tmp_geo_path

            # Show processing steps with progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🛰️ Fetching Sentinel-1 SAR imagery...")
            progress_bar.progress(25)
            time.sleep(0.5)
            
            status_text.text("🔄 Preprocessing SAR data...")
            progress_bar.progress(50)
            time.sleep(0.5)
            
            status_text.text("🤖 Running ship detection model...")
            progress_bar.progress(75)
            time.sleep(0.5)
            
            status_text.text("📊 Generating results...")
            progress_bar.progress(100)

            try:
                out = get_sentinel1_jpg_from_geojson(
                    geojson_path=tmp_geo_path,
                    year=year,
                    month=month
                )
                # store output in session
                st.session_state["result_out"] = out
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                # Rerun so UI switches to result display
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Processing failed: {str(e)}")
                progress_bar.empty()
                status_text.empty()
                # cleanup temp file on failure
                if os.path.exists(tmp_geo_path):
                    try:
                        os.remove(tmp_geo_path)
                    except Exception:
                        pass