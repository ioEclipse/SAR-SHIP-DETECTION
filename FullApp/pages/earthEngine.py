import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import tempfile
import json
import os
import pandas as pd
from engineAPI1 import get_sentinel1_jpg_from_geojson 

# --- Page config ---
st.set_page_config(page_title="SAR Map Viewer", layout="wide")

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
predict_clicked = st.sidebar.button("▶️ Predict SAR & Detect Ships")

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
    st.experimental_rerun()

# --- Main area ---
# If we already have a result saved in session_state, show the result UI.
if "result_out" in st.session_state and st.session_state["result_out"]:
    out = st.session_state["result_out"]

    st.header("📈 SAR Detection Result")
    # Show detection image if present
    if isinstance(out, dict) and out.get("detections") and os.path.exists(out["detections"]):
        st.image(out["detections"], caption="SAR Ship Detections", use_container_width=False)
    else:
        st.error("No detection image found in the result.")

    # Show total ship count (if available)
    ship_count = out.get("ship_count") if isinstance(out, dict) else None
    if ship_count is not None:
        st.subheader(f"Total Ships Detected: {ship_count}")
    else:
        st.info("Ship count not available in the result object.")

    # Load and show metadata table (first 5 rows by default)
    metadata_path = out.get("metadata") if isinstance(out, dict) else None
    if metadata_path and os.path.exists(metadata_path):
        try:
            df = pd.read_json(metadata_path) if metadata_path.endswith(".json") else pd.read_csv(metadata_path)
        except Exception:
            # robust fallback
            with open(metadata_path, "r") as mf:
                metadata_list = json.load(mf)
            df = pd.DataFrame(metadata_list)

        st.markdown("### 🧾 Ship Metadata")
        show_full = st.checkbox("See full table", value=False, key="show_full_table")
        if show_full:
            st.dataframe(df)
        else:
            st.dataframe(df.head(5))
    else:
        st.info("No metadata file available to display.")

    st.markdown("---")
    st.info("Use the sidebar Reset button to run another query or draw a new polygon.")

else:
    # No result yet -> show the map and drawing tools
    # Styled header banner for the drawing section
    st.markdown(
        """
        <div style="
            padding: 14px 18px;
            border-radius: 12px;
            background: transparent;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 12px;
            margin-bottom: 10px;
        ">
            <span style="font-size: 28px;"></span>
            <div style="line-height: 1.2; text-align: center;">
                <div style="font-size: 50px; font-weight: 800;">Select Area</div>
                <div style="font-size: 13px; opacity: 0.95;">Draw a polygon on the map to define your area of interest</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Create Folium map
    center = [31.2, 32.3]  # adjust as needed
    m = folium.Map(location=center, zoom_start=6)
    draw = Draw(
        export=False,
        draw_options={
            'polyline': False, 'rectangle': False, 'circle': False,
            'circlemarker': False, 'marker': False,
            'polygon': {'allowIntersection': False, 'showArea': True, 'shapeOptions': {'color': '#97009c'}}
        }
    )
    draw.add_to(m)

    map_data = st_folium(m, width=1100, height=600)

    # Extract polygon GeoJSON from map_data (support different keys)
    geo = None
    for key in ("last_drawn_geojson", "last_active_drawing", "all_drawings", "features"):
        if isinstance(map_data, dict) and key in map_data and map_data.get(key):
            geo = map_data.get(key)
            break
    # fallback: sometimes 'all_drawings' contains a list - try to pick first
    if not geo and isinstance(map_data, dict) and "all_drawings" in map_data and map_data["all_drawings"]:
        try:
            geo = map_data["all_drawings"][0]
        except Exception:
            geo = None

    if geo:
        st.json(geo)
    

    # If Predict button clicked in the sidebar, process now
    if predict_clicked:
        if not geo:
            st.sidebar.error("No polygon drawn. Please draw a polygon on the map before predicting.")
        else:
            # Wrap polygon into FeatureCollection expected by your function.
            # `geo` may already be a Feature or a geometry. Detect and wrap appropriately.
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

            # Call your existing function (no modification)
            with st.spinner("Fetching Sentinel-1, preprocessing and running detection... This may take several minutes depending on area and processing."):
                try:
                    out = get_sentinel1_jpg_from_geojson(
                        geojson_path=tmp_geo_path,
                        year=year,
                        month=month
                    )
                    # store output in session
                    st.session_state["result_out"] = out
                    # Rerun so UI switches to result display
                    st.experimental_rerun()
                except Exception as e:
                    st.sidebar.error(f"Processing failed: {str(e)}")
                    # cleanup temp file on failure
                    if os.path.exists(tmp_geo_path):
                        try:
                            os.remove(tmp_geo_path)
                        except Exception:
                            pass
