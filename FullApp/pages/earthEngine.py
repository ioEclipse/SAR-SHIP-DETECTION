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
import base64

# --- Page config ---
st.set_page_config(page_title="SAR Map Viewer", layout="wide", initial_sidebar_state="expanded")

hide_streamlit_style = """
<style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
    [data-testid="stHeader"] {
        display: none;
    }
    [data-testid="stToolbar"] {
        display: none;
    }
    .stApp > header {
        display: none;
    }
    .stDeployButton {
        display: none;
    }
    footer {
        display: none;
    }
    #MainMenu {
        display: none;
    }
    /* Hide sidebar button */
        [data-testid="collapsedControl"] {
            display: none;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Add custom CSS for larger metric text and loading components
st.markdown("""
<style>
    .st-emotion-cache-595tnf{
            height: 0;
            width: 0;
    }
    .stMainBlockContainer{
        padding-top: 20px;
    }
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

    /* Custom styling for the predict button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        padding: 15px 25px !important;
        border-radius: 12px !important;
        width: 100% !important;
        font-size: 10px !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
    }

    /* Custom styling for the reset button */
    .stButton > button:not([kind="primary"]) {
        background: linear-gradient(135deg, #6c757d 0%, #495057 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        padding: 12px 20px !important;
        border-radius: 10px !important;
        width: 100% !important;
        font-size: 14px !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    .stButton > button:not([kind="primary"]):hover {
        background: linear-gradient(135deg, #5a6268 0%, #3d4449 100%) !important;
    }

    /* Loading spinner styling */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 20px 0;
        padding: 30px;
        background: transparent;
        border-radius: 15px;
        border: none;
        box-shadow: none;
    }

    .custom-spinner {
        border: 4px solid #333333;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin-bottom: 15px;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .loading-text {
        color: #667eea;
        font-size: 16px;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
    }

    /* Progress bar custom styling */
    .stProgress > div > div > div > div {
        background-color: #667eea !important;
    }

    /* Custom progress container styling */
    .progress-container {
        background: transparent;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: none;
    }
</style>
""", unsafe_allow_html=True)


def load_logo_base64(path="assets/logo.png"):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

logo_data = load_logo_base64()

with st.sidebar:
    st.markdown(
        f"""
        <div style="
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 10px;
            padding: 10px 0;
            flex-wrap: nowrap;
        ">
            <img src="data:image/png;base64,{logo_data}" style="
                height: 40px; 
                width: auto;
                flex-shrink: 0;
            ">
            <h1 style="
                color: #1e90ff;
                margin: 0;
                font-size: 24px;
                font-weight: bold;
                white-space: nowrap;
            ">BlueGuard</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- Sidebar: Filters + Predict + Reset ---
st.sidebar.title("Filters")

year = st.sidebar.selectbox("Select Year", [2022, 2023, 2024, 2025])
month = st.sidebar.selectbox(
    "Select Month",
    list(range(1, 13)),
    format_func=lambda m: [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ][m - 1]
)

# Predict button in sidebar (under the selectors)
predict_clicked = st.sidebar.button("Predict SAR & Detect Ships", type="primary")

# Reset button (to return to map)
if st.sidebar.button("🔄 Reset Analysis", type="secondary"):
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

if st.sidebar.button("Back to main", key="back_main"):
        st.switch_page("pages/main.py")

# --- Main area ---
# If we already have a result saved in session_state, show the result UI.
if "result_out" in st.session_state and st.session_state["result_out"]:
    out = st.session_state["result_out"]

    ship_count = out.get("ship_count") if isinstance(out, dict) else None
    st.header(f"🚢 Total Ships Detected {ship_count}")

    # Create columns for better layout - MODIFIÉ POUR AFFICHER LES 2 IMAGES
    col_img1, col_img2 = st.columns([5, 5])

    with col_img1:
        # Show original image if present
        if isinstance(out, dict) and out.get("original") and os.path.exists(out["original"]):
            st.image(out["original"], caption="Original SAR Image", use_container_width=True)
        else:
            st.error("Original image not available")

    with col_img2:
        # Show detection image if present
        if isinstance(out, dict) and out.get("detections") and os.path.exists(out["detections"]):
            st.image(out["detections"], caption="SAR Ship Detections", use_container_width=True)
        else:
            st.error("No detection image found in the result.")
    
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

    # Create containers for loading UI - MOVED ABOVE THE MAP
    loading_container = st.empty()
    progress_container = st.empty()

    # Create Folium map with basic tile layer
    center = [40.5, -73]  # Mediterranean Sea area
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
                wrapped = {"type": "FeatureCollection",
                           "features": [{"type": "Feature", "properties": {}, "geometry": geo}]}

            # Save temp geojson file
            tmp_geo = tempfile.NamedTemporaryFile(delete=False, suffix=".geojson", mode="w")
            json.dump(wrapped, tmp_geo)
            tmp_geo.close()
            tmp_geo_path = tmp_geo.name
            st.session_state["tmp_geojson_path"] = tmp_geo_path

            try:
                # Show initial loading state with spinner
                with loading_container.container():
                    st.markdown("""
                    <div class="loading-container">
                        <div class="custom-spinner"></div>
                        <div class="loading-text">🛰️ Initializing SAR processing...</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Initialize progress bar in the progress container
                with progress_container.container():
                    progress_bar = st.progress(0, text="Starting processing...")

                # Step 1: Fetching imagery
                time.sleep(0.03)
                progress_bar.progress(10, text="🛰️ Fetching Sentinel-1 SAR imagery... 10%")
                time.sleep(0.5)

                # Simulate progress while processing
                for percent_complete in range(15, 60, 5):
                    time.sleep(0.03)
                    progress_bar.progress(percent_complete, text=f"🔄 Preprocessing SAR data... {percent_complete}%")

                # Step 2: Model processing
                progress_bar.progress(70, text="🤖 Running ship detection model... 70%")
                time.sleep(0.5)

                # Actual processing
                out = get_sentinel1_jpg_from_geojson(
                    geojson_path=tmp_geo_path,
                    year=year,
                    month=month
                )

                # Final steps
                progress_bar.progress(90, text="📊 Generating results... 90%")
                time.sleep(0.2)

                # Store output in session
                st.session_state["result_out"] = out

                progress_bar.progress(100, text="✅ Processing complete! 100%")
                time.sleep(0.3)

                # Clear loading UI
                loading_container.empty()
                progress_container.empty()

                # Rerun so UI switches to result display
                st.rerun()

            except Exception as e:
                # Clear loading UI on error
                loading_container.empty()
                progress_container.empty()
                
                st.sidebar.error(f"❌ Processing failed: {str(e)}")
                
                # cleanup temp file on failure
                if os.path.exists(tmp_geo_path):
                    try:
                        os.remove(tmp_geo_path)
                    except Exception:
                        pass