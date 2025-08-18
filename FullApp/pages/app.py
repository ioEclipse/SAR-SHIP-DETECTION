import streamlit as st
import io
import base64
import pandas as pd
import sys
import os
from tempfile import NamedTemporaryFile, NamedTemporaryFile as NTF
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from functions import *   
from streamlit_option_menu import option_menu
import json
import time
import cv2
import numpy as np

# === Fonction pour charger le logo ===
def load_logo_base64(path="assets/logo.png"):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

logo_data = load_logo_base64()

# === Layout ===
st.set_page_config(
    page_title="SAR Ship Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)
if 'ship_counter' not in st.session_state:
    st.session_state.ship_counter = 0
    
# === CSS Design Global ===
st.markdown(f"""
<style>
/* Global dark theme */
.stApp {{
    background-color: #0f0f0f !important;
    color: #ffffff !important;
}}

/* Sidebar styling */
[data-testid="stSidebar"] {{
    background-color: #1a1a1a !important;
    border-right: 1px solid #333333;
}}

/* Main content area */
.main {{
    background-color: #0f0f0f !important;
    color: #ffffff !important;
}}

/* Logo and header styling */
.logo-container {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 30px;
    padding: 10px 0;
}}

.logo-container img {{
    height: 40px;
    width: auto;
}}

.logo-container h1 {{
    color: #1e90ff;
    margin: 0;
    font-size: 24px;
    font-weight: bold;
}}

.upload-title {{
    color: #ffffff;
    font-size: 18px;
    font-weight: bold;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 8px;
}}

/* File uploader styling */
.stFileUploader {{
    background-color: #2a2a2a !important;
    border: 2px dashed #444444 !important;
    border-radius: 8px !important;
    padding: 20px !important;
}}

.stFileUploader:hover {{
    border-color: #1e90ff !important;
}}

/* Button styling */
.stButton > button {{
    background-color: #1e90ff !important;
    color: white !important;
    font-weight: bold !important;
    border: none !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    width: 100% !important;
    font-size: 16px !important;
    transition: all 0.3s ease !important;
}}

.stButton > button:hover {{
    background-color: #0066cc !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(30, 144, 255, 0.3) !important;
}}

/* Main content styling */
.main-content {{
    background-color: #0f0f0f !important;
    color: #ffffff !important;
    padding: 30px !important;
}}

/* Process steps styling */
.process-steps {{
    display: flex;
    justify-content: space-between;
    margin-top: 30px;
    gap: 20px;
}}

.step-item {{
    text-align: center;
    flex: 1;
}}

.step-icon {{
    background-color: #1e90ff;
    border-radius: 50%;
    width: 60px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 10px;
    color: white;
    font-size: 24px;
}}

.step-title {{
    color: #ffffff;
    font-size: 14px;
    font-weight: bold;
    margin-top: 8px;
}}

.ship-counter {{
    background-color: #1e90ff;
    color: white;
    padding: 15px;
    border-radius: 8px;
    font-size: 18px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 20px;
}}

/* Dropdown styling */
.stSelectbox > div {{
    background-color: #2a2a2a !important;
    border: 1px solid #444444 !important;
    border-radius: 8px !important;
}}

.stSelectbox label {{
    color: #ffffff !important;
}}

/* Download button styling */
    .stDownloadButton > button {{
        background: linear-gradient(135deg, #1e90ff, #0066cc) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        padding: 8px 10px !important;
        border-radius: 8px !important;
        font-size: 12px !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }}

    .stDownloadButton > button:hover {{
        background: linear-gradient(135deg, #0066cc, #1e90ff) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(30, 144, 255, 0.3) !important;
    }}


    /* Fixed position for main download button */
    .download-container {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
    }}

/* Status message styling */
.status-message {{
    background-color: #2a2a2a;
    border: 1px solid #444444;
    border-radius: 8px;
    padding: 15px;
    margin-top: 15px;
    color: #1e90ff;
    font-weight: bold;
}}

/* Table styling */
.dataframe {{
    background-color: #1a1a1a !important;
    color: #ffffff !important;
}}

.dataframe th {{
    background-color: #2a2a2a !important;
    color: #ffffff !important;
}}

.dataframe td {{
    background-color: #1a1a1a !important;
    color: #ffffff !important;
}}

/* Hide Streamlit default elements */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}
</style>
""", unsafe_allow_html=True)

# === Sidebar ===
with st.sidebar:
    # Logo and title
    st.markdown(
        f"""
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_data}" style="height: 40px; width: auto;">
            <h1>BlueGuard</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    # SAR image uploader in sidebar
    st.markdown('<div class="upload-title">📤 Upload Image</div>', unsafe_allow_html=True)
    uploaded_image = st.file_uploader(
        'Drag and drop your SAR image here',
        type=["jpg", "png", "jpeg", "tif", "tiff"],
        key="file_uploader",
        help="Supported formats: JPG, PNG, JPEG, TIFF (Max 200MB)"
    )
    if uploaded_image:
        st.success(f"✅ File uploaded: {uploaded_image.name}")
        if uploaded_image.name.lower().endswith(('.tif', '.tiff')):
            st.info("ℹ️ TIFF file detected - Automatic conversion will be applied")
    # AIS uploader in sidebar
    st.markdown('<div class="upload-title">🛰️ Optional: Upload AIS CSV</div>', unsafe_allow_html=True)
    ais_csv_uploader = st.file_uploader(
        'Upload AIS CSV for the image date (optional)', type=["csv"], key="ais_uploader",
        help="Optional: upload the AIS CSV of the corresponding day (ex: AIS_2024_01_24.csv)"
    )
    # Process button in sidebar
    process_clicked = st.button("🚀 Process & Predict", key="predict_button")
    st.markdown('</div>', unsafe_allow_html=True)

# === Main content ===
st.markdown('<div class="main-content" style=height:0;width:0;>', unsafe_allow_html=True)

# Use process_clicked to trigger processing in main area
if process_clicked:
    if uploaded_image:
        tmp_tif_path = None
        tmp_ais_path = None
        progress_placeholder = st.empty()
        percent_text_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0, text="Starting processing...")
        try:
            # Simulate progress while processing
            for percent_complete in range(0, 80, 5):
                time.sleep(0.03)
                progress_bar.progress(percent_complete, text=f"Processing image... {percent_complete}%")
            # Actual processing
            if uploaded_image.name.lower().endswith(('.tif', '.tiff')):
                with NamedTemporaryFile(suffix=".tif", delete=False) as tmp_tif:
                    tmp_tif.write(uploaded_image.getvalue())
                    tmp_tif_path = tmp_tif.name
                annotated, crops, ship_counter, metadata = run_inference_with_crops(tmp_tif_path)
            else:
                annotated, crops, ship_counter, metadata = run_inference_with_crops(uploaded_image)
            progress_bar.progress(90, text="Finalizing results... 90%")
            # Save session state (same names as before)
            st.session_state.annotated_image = annotated
            st.session_state.ship_crops = crops
            st.session_state.ship_counter = ship_counter
            st.session_state.metadata = metadata
            progress_bar.progress(100, text="✅ Processing complete! 100%")
            time.sleep(0.3)
            percent_text_placeholder.empty()
            progress_placeholder.empty()
            # FIX: AIS — ensure the metadata on disk matches the in-memory metadata the UI shows
            st.session_state.ais_results = None
            if uploaded_image.name.lower().endswith(('.tif', '.tiff')):
                meta_tmp_path = "ship_metadata_ui.json"
                try:
                    with open(meta_tmp_path, "w", encoding="utf-8") as mf:
                        json.dump(metadata, mf, indent=2, ensure_ascii=False)
                except Exception as e:
                    st.error(f"❌ Impossible d'écrire le fichier temporaire des métadonnées: {e}")
                    meta_tmp_path = None
                candidates = [
                    os.path.join(os.path.dirname(__file__), "AIS_2024_01_24.csv"),
                    os.path.join(os.path.dirname(__file__), "pages", "AIS_2024_01_24.csv"),
                    os.path.join(os.getcwd(), "AIS_2024_01_24.csv"),
                    os.path.join(os.getcwd(), "pages", "AIS_2024_01_24.csv"),
                    "AIS_2024_01_24.csv"
                ]
                if ais_csv_uploader:
                    with NamedTemporaryFile(suffix=".csv", delete=False) as tmp_ais:
                        tmp_ais.write(ais_csv_uploader.getvalue())
                        tmp_ais_path = tmp_ais.name
                    ais_csv_path = tmp_ais_path
                else:
                    ais_csv_path = next((p for p in candidates if os.path.exists(p)), "AIS_2024_01_24.csv")
                    if not os.path.exists(ais_csv_path):
                        st.warning(f"Le fichier AIS n'a pas été trouvé automatiquement; ensure '{ais_csv_path}' exists or upload it via the sidebar (optional).")
                has_geoloc = any((entry.get("geolocation") is not None) for entry in metadata)
                if has_geoloc and meta_tmp_path:
                    try:
                        ais_results = search_ais_for_metadata(
    metadata_path="ship_metadata_ui.json",
    ais_csv_path="pages/AIS_2024_07_06.csv",
    date_iso="2024-07-06T04:30:22",
    output_path="AIS_search.json",
    time_window_s=300,
    search_radius_m=100,
    time_weight=0.5
)
                        st.session_state.ais_results = ais_results
                    except Exception as e:
                        st.session_state.ais_results = None
                        st.error(f"❌ Error during AIS lookup: {e}")
                else:
                    st.session_state.ais_results = None
                    if not has_geoloc:
                        st.info("No geolocation present in metadata; skipping AIS search.")
                    else:
                        st.error("Temporary metadata file not written; skipping AIS search.")
                if tmp_ais_path and os.path.exists(tmp_ais_path):
                    try:
                        os.unlink(tmp_ais_path)
                    except Exception:
                        pass
            if tmp_tif_path and os.path.exists(tmp_tif_path):
                try:
                    os.unlink(tmp_tif_path)
                except Exception:
                    pass
        except Exception as e:
            st.error(f"❌ Error during inference: {str(e)}")
            if 'tmp_tif_path' in locals() and tmp_tif_path and os.path.exists(tmp_tif_path):
                os.unlink(tmp_tif_path)
            if 'tmp_ais_path' in locals() and tmp_ais_path and os.path.exists(tmp_ais_path):
                os.unlink(tmp_ais_path)
    else:
        st.warning("⚠️ Please upload an image first")

if "annotated_image" not in st.session_state or st.session_state.annotated_image is None:
    # === Default presentation block ===
    st.markdown(
        "<h1 style='color: #ffffff; font-size: 36px; font-weight: bold; margin-bottom: 20px;'>"
        "🚀 Start Your Analysis"
        "</h1>",
        unsafe_allow_html=True
    )
    
    st.markdown(
        "<p style='color: #cccccc; font-size: 18px; line-height: 1.6; margin-bottom: 40px;'>"
        "Upload your SAR file and the system will automatically perform full processing and deliver a detailed, ready-to-export detection report."
        "</p>",
        unsafe_allow_html=True
    )
    
    # Image display area
    col1, col2 = st.columns([4, 1])
    with col1:
        st.image("assets/defaultcontent.png", use_container_width=True)
    with col2:
        st.markdown('<div class="process-steps">', unsafe_allow_html=True)
    
        def get_base64_image(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()

        img_base64_1 = get_base64_image("assets/preprocessing.png")
        img_base64_2 = get_base64_image("assets/boundingboxes.png")
        img_base64_3 = get_base64_image("assets/subimages.png")
        img_base64_4 = get_base64_image("assets/statisticalinsights.png")
        
        steps = [
            {"icon": f'<img src="data:image/png;base64,{img_base64_1}" style="width:32px;height:32px;">', "title": "Preprocessing"},
            {"icon": f'<img src="data:image/png;base64,{img_base64_2}" style="width:32px;height:32px;">', "title": "Bounding Boxes"},
            {"icon": f'<img src="data:image/png;base64,{img_base64_3}" style="width:32px;height:32px;">', "title": "Sub-Images"},
            {"icon": f'<img src="data:image/png;base64,{img_base64_4}" style="width:32px;height:32px;">', "title": "Statistical Insights"}
        ]
        
        for step in steps:
            st.markdown(
                f"""
                <div class="step-item">
                    <div class="step-icon">{step['icon']}</div>
                    <div class="step-title">{step['title']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

else:
    if 'ship_counter' not in st.session_state:
        st.session_state.ship_counter = 0
    # Ship counter
    st.markdown(f'<h1 style="color: #ffffff; font-size: 36px; font-weight: bold; margin-bottom: 20px;">🚢 Total Ships Detected: {st.session_state.ship_counter}</h1>', unsafe_allow_html=True)
    
    # Main image and download
    st.image(st.session_state.annotated_image, use_container_width=True)

    # Align the download button with the right edge of the image
    col1, col2 = st.columns([8, 1])
    with col2:
        buf = io.BytesIO()
        st.session_state.annotated_image.save(buf, format="PNG")
        st.download_button("Download", data=buf.getvalue(), file_name="annotated_image.png", key="download_annotated")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Replace the preprocessing pipeline section in your app.py with this:
if uploaded_image is not None and "annotated_image" in st.session_state and st.session_state.annotated_image is not None:
    try:
        # Handle TIFF files by converting them to JPG first
        if uploaded_image.name.lower().endswith(('.tif', '.tiff')):
            with NamedTemporaryFile(suffix=".tif", delete=False) as tmp_tif:
                tmp_tif.write(uploaded_image.getvalue())
                tmp_tif_path = tmp_tif.name
            
            # Convert to JPG using your existing function
            with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_jpg:
                tmp_jpg_path = tmp_jpg.name
                convert_radar_tif_to_jpg(tmp_tif_path, tmp_jpg_path)
            
            # Call preprocessing pipeline with the converted JPG
            image_paths = preprocessing_pipeline(tmp_jpg_path)
            
            # Clean up temporary files
            if tmp_tif_path and os.path.exists(tmp_tif_path):
                try:
                    os.unlink(tmp_tif_path)
                except Exception:
                    pass
            if tmp_jpg_path and os.path.exists(tmp_jpg_path):
                try:
                    os.unlink(tmp_jpg_path)
                except Exception:
                    pass
        else:
            # For non-TIFF files, call directly with the uploaded file
            image_paths = preprocessing_pipeline(uploaded_image)
        
        st.session_state.preprocessing_paths = image_paths
        
        # Debug info
        available_steps = len([p for p in image_paths.values() if p is not None])
        print(f"🔍 Debug: {available_steps}/{len(image_paths)} preprocessing steps available")
        
    except Exception as e:
        st.error(f"❌ Error during preprocessing: {str(e)}")
        st.session_state.preprocessing_paths = None

    # Safe image display in the expander
    with st.expander("Preprocessing pipeline (all steps)"):
        col1, col2, col3, col4 = st.columns(4)
        col5, col6, col7, col8 = st.columns(4)

        def safe_image_display(col, image_path, caption, fallback_text="Image not available"):
            """Safely display an image with error handling"""
            with col:
                if image_path and os.path.exists(image_path):
                    try:
                        st.image(image_path, caption=caption, use_container_width=True)
                    except Exception as e:
                        st.error(f"{fallback_text}: {caption}")
                        print(f"❌ Error displaying {caption}: {e}")
                else:
                    st.error(f"{fallback_text}: {caption}")
                    if image_path:
                        print(f"❌ Path exists but file missing: {image_path}")
                    else:
                        print(f"❌ No path available for: {caption}")

        # Check if preprocessing was successful
        if hasattr(st.session_state, 'preprocessing_paths') and st.session_state.preprocessing_paths:
            paths = st.session_state.preprocessing_paths
            
            # Display all steps
            safe_image_display(col1, paths.get("initial"), "Initial Image")
            safe_image_display(col2, paths.get("step1"), "Step 1: Lee Filter")
            safe_image_display(col3, paths.get("step2"), "Step 2: Enhance") 
            safe_image_display(col4, paths.get("step3"), "Step 3: Thresholding")
            safe_image_display(col5, paths.get("step4"), "Step 4: Morphing")
            safe_image_display(col6, paths.get("step5"), "Step 5: Apply Mask")
            safe_image_display(col7, paths.get("masked_image"), "Step 6: Masked Image")
            with col8:        
            # Final image (for inference) - display separately below
                if paths.get("final") and os.path.exists(paths.get("final")):
                    st.image(paths.get("final"), caption="Step 8: Final Image for Inference", use_container_width=True)
                else:
                    st.error("❌ Final processed image not available")
                

        else:
            # Show placeholder messages for all steps
            safe_image_display(col1, None, "Initial Image")
            for col, caption in zip([col2, col3, col4, col5, col6, col7, col8], 
                                ["Step 1: Lee Filter", "Step 2: Enhance", "Step 3: Thresholding", 
                                    "Step 4: Morphing", "Step 5: Apply Mask", "Step 6: Masked Image", "Step 7: Final Mask"]):
                safe_image_display(col, None, caption)
            
            st.error("❌ Preprocessing has not been completed yet. Please process an image first.")
if st.session_state.ship_counter > 0:
    st.markdown("---")
    st.markdown("### 🔍 Ship Details")
    
    ship_names = [name for name, _ in st.session_state.ship_crops]
    selected_ship = st.selectbox("Choose a ship to view details", ship_names, key="ship_select")
    
    if selected_ship:
        
        col1, col2 = st.columns([2, 1])
        with col1:
            crop_img = dict(st.session_state.ship_crops)[selected_ship]

            # Convert PIL image to Base64
            buffer = io.BytesIO()
            crop_img.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode()

            # Display image with custom width using HTML & CSS
            st.markdown(f"""
                <div style="text-align:center;">
                    <img src="data:image/png;base64,{img_base64}" 
                            style="width:350px; border-radius:10px; display:block; margin:auto;">
                    <p style="text-align:center; color:#ffffff; font-size:16px;">📸 {selected_ship}</p>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            # Ship metadata
            for entry in st.session_state.metadata:
                if entry['ship_id'] == selected_ship:
                    pixel_area = entry['pixel_area']
                    surface_m2 = entry['surface_m2']
                    geoloc = entry.get("geolocation", None)
                    break
            
            st.markdown("### 📊 Ship Information")
            st.markdown(f"""
            - **Ship ID:** {selected_ship}
            - **Pixel Area:** {pixel_area} px²
            - **Surface:** {surface_m2} m²
            """)
            
            if geoloc:
                st.markdown(f"- **Geolocation:** {geoloc.get('lat')}, {geoloc.get('lon')}")
            else:
                st.markdown(f"- **Geolocation:** None")
            
            # Download button for individual ship
            crop_buf = io.BytesIO()
            crop_img.save(crop_buf, format="JPEG")
            st.download_button("📥 Download Ship", data=crop_buf.getvalue(), file_name=f"{selected_ship}.jpg", key="download_crop")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Metadata table
    st.markdown("### 📋 Ship Characteristics Table")
    df = pd.DataFrame(st.session_state.metadata)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        show_all = st.checkbox("Show full table", value=False, key="show_table")
    with col2:
        st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
    
    if show_all:
        st.dataframe(df, use_container_width=True)
    else:
        st.dataframe(df.head(5), use_container_width=True)

    # NEW: Display AIS search results table under the metadata table (only if present)
    if st.session_state.get("ais_results") is not None:
        st.markdown("### 🛰️ AIS Search Results (matched to metadata ships)")
        # ais_results is a dict ship_id -> dict or None
        ais_results = st.session_state.ais_results
        # build a table aligning with metadata order
        rows = []
        for entry in st.session_state.metadata:
            sid = entry.get("ship_id")
            res = ais_results.get(sid) if isinstance(ais_results, dict) else None
            if res is None:
                rows.append({"ship_id": sid, "AIS_found": False})
            else:
                # flatten some common AIS fields if present
                row = {"ship_id": sid, "AIS_found": True}
                row["MMSI"] = res.get("MMSI")
                row["VesselName"] = res.get("VesselName")
                row["BaseDateTime"] = res.get("BaseDateTime")
                row["LAT"] = res.get("LAT")
                row["LON"] = res.get("LON")
                row["SOG"] = res.get("SOG")
                row["COG"] = res.get("COG")
                row["IMO"] = res.get("IMO")
                rows.append(row)
        ais_df = pd.DataFrame(rows)
        st.dataframe(ais_df, use_container_width=True)

        # provide download of AIS_search.json if exists
        if os.path.exists("AIS_search.json"):
            with open("AIS_search.json", "rb") as f:
                st.download_button("Download AIS_search.json", data=f.read(), file_name="AIS_search.json", key="download_ais_json")
        else:
            # fallback: offer to download the in-memory ais_results as JSON
            ais_json_bytes = json.dumps(ais_results, indent=2, ensure_ascii=False).encode("utf-8")
            st.download_button("Download AIS results (JSON)", data=ais_json_bytes, file_name="AIS_search.json", key="download_ais_json_mem")

st.markdown('</div>', unsafe_allow_html=True)

# === Wrapper for preprocessing_pipeline ===
def preprocessing_pipeline(image_np, uploaded_image=None):
    """
    Wrapper for the preprocessing_pipeline from functions.py so it can be called from app.py.
    Args:
        image_np: numpy array of the image (grayscale)
        uploaded_image: the uploaded file or its path (optional, for initial image reference)
    Returns:
        dict: paths to all intermediate and final images
    """
    # Call the imported function from functions.py
    return globals()["preprocessing_pipeline"](image_np, uploaded_image)