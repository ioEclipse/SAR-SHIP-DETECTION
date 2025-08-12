import streamlit as st
import io
import base64
import pandas as pd
import sys
import os
from tempfile import NamedTemporaryFile
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from functions import *
from streamlit_option_menu import option_menu


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
    background-color: #1e90ff !important;
    color: white !important;
    font-weight: bold !important;
    border: none !important;
    padding: 8px 16px !important;
    border-radius: 6px !important;
    font-size: 14px !important;
}}

.stDownloadButton > button:hover {{
    background-color: #0066cc !important;
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
    
    # Upload section
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process button
    if st.button("🚀 Process & Predict", key="predict_button"):
        if uploaded_image:
            st.markdown('<div class="status-message">⏳ Running inference... Please wait</div>', unsafe_allow_html=True)
            try:
                # Gestion spécifique pour les fichiers TIFF
                if uploaded_image.name.lower().endswith(('.tif', '.tiff')):
                    with NamedTemporaryFile(suffix=".tif", delete=False) as tmp_tif:
                        tmp_tif.write(uploaded_image.getvalue())
                        tmp_path = tmp_tif.name
                    
                    annotated, crops, ship_counter, metadata = run_inference_with_crops(tmp_path)
                    os.unlink(tmp_path)  # Nettoyage du fichier temporaire
                else:
                    # First pre-process the uploaded image
                # Display pre-processed images (2.5s)   
                # Pass preprocessed image to the inference function
                annotated, crops, ship_counter, metadata = run_inference_with_crops(uploaded_image)
                
                st.session_state.annotated_image = annotated
                st.session_state.ship_crops = crops
                st.session_state.ship_counter = ship_counter
                st.session_state.metadata = metadata
                st.success("✅ Processing complete!")
            except Exception as e:
                st.error(f"❌ Error during inference: {str(e)}")
                if 'tmp_path' in locals() and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            st.warning("⚠️ Please upload an image first")

# === Main content ===
st.markdown('<div class="main-content" style=height:0;width:0;>', unsafe_allow_html=True)

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
                        break
                
                st.markdown("### 📊 Ship Information")
                st.markdown(f"""
                - **Ship ID:** {selected_ship}
                - **Pixel Area:** {pixel_area} px²
                - **Surface:** {surface_m2} m²
                """)
                
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

st.markdown('</div>', unsafe_allow_html=True)