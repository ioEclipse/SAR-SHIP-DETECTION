import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import tempfile
import json
import os
import pandas as pd
import time
from engineAPI1 import get_sentinel1_jpg_from_geojson 

# --- Page config ---
st.set_page_config(page_title="SAR Map Viewer", layout="wide")

# Add custom CSS for larger metric text and better styling
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

# Predict button in sidebar (under the selectors) with primary styling
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

    # Enhanced header with ship count
    ship_count = out.get("ship_count") if isinstance(out, dict) else None
    if ship_count is not None:
        st.header(f"🚢 Total Ships Detected: {ship_count}")
    else:
        st.header("📈 SAR Detection Result")
    
    # Create columns for better layout
    col1, col2 = st.columns([10, 1])
    
    with col1:
        # Show detection image if present
        if isinstance(out, dict) and out.get("detections") and os.path.exists(out["detections"]):
            st.image(out["detections"], caption="SAR Ship Detections", use_container_width=True)
        else:
            st.error("No detection image found in the result.")
    
    with col2:
        # Show processing info if available (placeholder for future use)
        processing_info = out.get("processing_info", {})
        if processing_info:
            st.subheader("📊 Processing Details")
            for key, value in processing_info.items():
                formatted_key = key.replace("_", " ").title()
                st.write(f"**{formatted_key}:** {value}")

    st.markdown("---")
    
    # Enhanced metadata table display
    metadata_path = out.get("metadata") if isinstance(out, dict) else None
    if metadata_path and os.path.exists(metadata_path):
        try:
            # Try different loading methods for robustness
            if metadata_path.endswith(".json"):
                with open(metadata_path, "r") as mf:
                    metadata_list = json.load(mf)
                df = pd.DataFrame(metadata_list)
            else:
                df = pd.read_csv(metadata_path)
        except Exception:
            # robust fallback
            with open(metadata_path, "r") as mf:
                metadata_list = json.load(mf)
            df = pd.DataFrame(metadata_list)

        # Enhanced metadata section with better layout
        meta_title_col, meta_ctrl_col = st.columns([7, 1])
        with meta_title_col:
            st.markdown("### 🧾 Ship Detection Metadata")
        with meta_ctrl_col:
            show_full = st.checkbox("Show full table", value=False, key="show_full_table")

        if show_full:
            st.dataframe(df, use_container_width=True)
        else:
            st.dataframe(df.head(5), use_container_width=True)
    else:
        st.info("No metadata file available to display.")

    st.markdown("---")
    st.info("Use the sidebar Reset button to run another query or draw a new polygon.")

else:
    # No result yet -> show the map and drawing tools
    # Enhanced styled header banner for the drawing section
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

    # Enhanced Folium map with satellite imagery option
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
    
    # Enhanced drawing tools with better styling
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

    # Enhanced map display with better dimensions and return objects
    map_data = st_folium(m, width=1200, height=500, returned_objects=["last_object_clicked_popup", "all_drawings"])

    # Enhanced polygon extraction logic
    geo = None
    if map_data.get("all_drawings"):
        geo = map_data["all_drawings"][-1]  # Get the last drawn shape
    else:
        # Fallback to original extraction method
        for key in ("last_drawn_geojson", "last_active_drawing", "features"):
            if isinstance(map_data, dict) and key in map_data and map_data.get(key):
                geo = map_data.get(key)
                break

    if geo:
        st.success("✅ Area selected! Use the sidebar to configure detection parameters and start processing.")
        with st.expander("View Selected Area GeoJSON"):
            st.json(geo)

    # Enhanced prediction processing with progress indicators
    if predict_clicked:
        if not geo:
            st.sidebar.error("❌ No polygon drawn. Please draw a polygon on the map before predicting.")
        else:
            # Wrap polygon into FeatureCollection expected by your function.
            if isinstance(geo, dict) and geo.get("type") == "FeatureCollection":
                wrapped = geo
            elif isinstance(geo, dict) and geo.get("type") == "Feature":
                wrapped = {"type": "FeatureCollection", "features": [geo]}
            else:
                # assume geometry -> wrap into Feature -> FeatureCollection
                wrapped = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": geo}]}

            # Save temp geojson file
            tmp_geo = tempfile.NamedTemporaryFile(delete=False, suffix=".geojson", mode="w")
            json.dump(wrapped, tmp_geo)
            tmp_geo.close()
            tmp_geo_path = tmp_geo.name
            st.session_state["tmp_geojson_path"] = tmp_geo_path

            # Enhanced progress indicators
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

            # Call your existing function (no modification)
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