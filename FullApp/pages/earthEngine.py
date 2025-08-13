import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import tempfile
import json
import os
import pandas as pd
import time
import io
import base64
from engineAPI1 import get_sentinel1_jpg_from_geojson

# --- Page config ---
st.set_page_config(page_title="SAR Map Viewer", layout="wide")

# Add custom CSS for styling
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

    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        padding: 8px 16px !important;
        border-radius: 8px !important;
        font-size: 12px !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }

    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #218838 0%, #1ea085 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(40, 167, 69, 0.3) !important;
    }

    /* Fixed position for main download button */
    .download-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
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

predict_clicked = st.sidebar.button("Predict SAR & Detect Ships", type="primary")

if st.sidebar.button("🔄 Reset Analysis", type="secondary"):
    for k in ("result_out", "tmp_geojson_path"):
        if k in st.session_state:
            try:
                if k == "tmp_geojson_path" and st.session_state.get(k):
                    if os.path.exists(st.session_state[k]):
                        os.remove(st.session_state[k])
            except Exception:
                pass
            st.session_state.pop(k, None)
    st.rerun()

# --- Main ---
if "result_out" in st.session_state and st.session_state["result_out"]:
    out = st.session_state["result_out"]

    ship_count = out.get("ship_count") if isinstance(out, dict) else None
    if ship_count is not None:
        st.header(f"🚢 Total Ships Detected: {ship_count}")
    else:
        st.header("📈 SAR Detection Result")

    st.markdown("---")

    # Show detection image
    if isinstance(out, dict) and out.get("detections") and os.path.exists(out["detections"]):
        col1, col2, col3 = st.columns([2, 3, 1])
        with col2:
            st.image(out["detections"], caption="SAR Ship Detections", width=400)

        # Floating download button
        with open(out["detections"], "rb") as img_file:
            img_data = img_file.read()

        st.markdown('<div class="download-container">', unsafe_allow_html=True)
        col1, col2 = st.columns([9, 2])
        with col2:
            st.download_button(
                "Download",
                data=img_data,
                file_name=f"SAR_detections_{year}_{month:02d}.jpg",
                mime="image/jpeg",
                key="download_detections"
            )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error("No detection image found in the result.")

    st.markdown("---")

    # Metadata
    metadata_path = out.get("metadata") if isinstance(out, dict) else None
    if metadata_path and os.path.exists(metadata_path):
        try:
            if metadata_path.endswith(".json"):
                with open(metadata_path, "r") as mf:
                    metadata_list = json.load(mf)
                df = pd.DataFrame(metadata_list)
            else:
                df = pd.read_csv(metadata_path)
        except Exception:
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
    else:
        st.info("No metadata file available to display.")

    crops_dir = out.get("crops_dir") if isinstance(out, dict) else None
    if crops_dir and os.path.exists(crops_dir):
        try:
            ship_files = [f for f in os.listdir(crops_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        except Exception as e:
            st.error(f"Error accessing crops directory: {e}")
    else:
        st.info("Ship crops directory not available. Individual ship downloads not possible.")
else:
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

    center = [31.2, 32.3]
    m = folium.Map(location=center, zoom_start=6)

    try:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satellite",
            overlay=False,
            control=True
        ).add_to(m)
    except:
        pass

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

    folium.LayerControl().add_to(m)

    map_data = st_folium(m, width=1200, height=500, returned_objects=["last_object_clicked_popup", "all_drawings"])

    geo = None
    if map_data.get("all_drawings"):
        geo = map_data["all_drawings"][-1]
    else:
        for key in ("last_drawn_geojson", "last_active_drawing", "features"):
            if isinstance(map_data, dict) and key in map_data and map_data.get(key):
                geo = map_data.get(key)
                break

    if geo:
        st.success("✅ Area selected! Use the sidebar to configure detection parameters and start processing.")
        with st.expander("View Selected Area GeoJSON"):
            st.json(geo)

    if predict_clicked:
        if not geo:
            st.sidebar.error("❌ No polygon drawn. Please draw a polygon on the map before predicting.")
        else:
            if isinstance(geo, dict) and geo.get("type") == "FeatureCollection":
                wrapped = geo
            elif isinstance(geo, dict) and geo.get("type") == "Feature":
                wrapped = {"type": "FeatureCollection", "features": [geo]}
            else:
                wrapped = {"type": "FeatureCollection",
                           "features": [{"type": "Feature", "properties": {}, "geometry": geo}]}

            tmp_geo = tempfile.NamedTemporaryFile(delete=False, suffix=".geojson", mode="w")
            json.dump(wrapped, tmp_geo)
            tmp_geo.close()
            tmp_geo_path = tmp_geo.name
            st.session_state["tmp_geojson_path"] = tmp_geo_path

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
                st.session_state["result_out"] = out
                progress_bar.empty()
                status_text.empty()
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"❌ Processing failed: {str(e)}")
                progress_bar.empty()
                status_text.empty()
                if os.path.exists(tmp_geo_path):
                    try:
                        os.remove(tmp_geo_path)
                    except Exception:
                        pass
