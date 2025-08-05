import streamlit as st
import io
import base64
import pandas as pd
from infer2 import run_inference_with_crops
from streamlit_option_menu import option_menu

# === Fonction pour charger le logo ===
def load_logo_base64(path="logo.png"):
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
html, body, [class*="css"] {{
    background-color: #f5f5f5;
    color: #ffffff;
}}
[data-testid="stSidebar"] {{
    background-color: #1D1D1D !important;
    padding: 5px;
}}
.sidebar-header {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 20px;
    margin-top: 0px;
}}
.sidebar-header img {{
    margin-top: 10px;
    height: 40px;
}}
.sidebar-header h1 {{
    color: #1e90ff;
    margin: 0;
    font-size: 24px;
}}
.section-title {{
    font-size: 16px;
    color: white;
    margin-top: 10px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
    font-weight: bold;
}}
.block-upload {{
    background-color: #505050;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 10px;
}}
.block-upload .stFileUploader, .block-upload label {{
    color: black !important;
}}
.stButton > button, .stDownloadButton > button {{
    background-color: #1e90ff;
    color: white;
    font-weight: bold;
    border: none;
    padding: 10px 16px;
    border-radius: 5px;
    width: 100%;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background-color: #1c86ee;
}}
.ship-number {{
    font-size: 24px;
    font-weight: bold;
    color: #ffffff;
    background-color: #1e90ff;
    padding: 10px;
    border-radius: 8px;
    margin-top: 10px;
}}
.result-row {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 20px;
    gap: 20px;
}}
.image-container {{
    flex: 0 0 60%;
}}
.stSelectbox label, .stSelectbox div {{
    color: #ffffff;
}}
.stSelectbox > div[data-baseweb="select"] {{
    background-color: #ffffff;
    border-radius: 5px;
}}
.main-content {{
    padding: 20px;
    background-color: #ffffff;
}}
</style>
""", unsafe_allow_html=True)

# === Sidebar ===
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-header">
            <img src="data:image/png;base64,{logo_data}">
            <h1>BLUE GUARD</h1>
        </div>
        """, unsafe_allow_html=True
    )
    st.markdown('<div class="section-title">🔍 Raw SAR Data Processing</div>', unsafe_allow_html=True)
    st.markdown('<div class="block-upload" style="background-color: #0000;">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #ffffff;">Add your Image</h3>', unsafe_allow_html=True)
    uploaded_image = st.file_uploader('📤', type=["jpg", "png"], key="file_uploader")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Process & Predict", key="predict_button"):
        if uploaded_image:
            st.info("Running inference... Please wait ⏳")
            try:
                annotated, crops, ship_counter, metadata = run_inference_with_crops(uploaded_image)
                st.session_state.annotated_image = annotated
                st.session_state.ship_crops = crops
                st.session_state.ship_counter = ship_counter
                st.session_state.metadata = metadata
            except Exception as e:
                st.warning(f"⚠️ Error during inference: {e}")

# === Main content ===
st.markdown('<div class="main-content">', unsafe_allow_html=True)

if "annotated_image" not in st.session_state or st.session_state.annotated_image is None:
    st.markdown("<h2 style='color: #004d99;'>Hello</h2>", unsafe_allow_html=True)
else:
    st.markdown('<div class="result-row">', unsafe_allow_html=True)
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(st.session_state.annotated_image, caption="📦 Image globale prédit", use_column_width=False, width=500)

    buf = io.BytesIO()
    st.session_state.annotated_image.save(buf, format="PNG")
    st.download_button("Download", data=buf.getvalue(), file_name="annotated_image.png", key="download_annotated")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(f'<div><span class="ship-number">Total Ships Detected: {st.session_state.ship_counter}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.ship_counter > 0:
        ship_names = [name for name, _ in st.session_state.ship_crops]
        selected_ship = st.selectbox("🔎 Choose a ship to view", ship_names, key="ship_select")
        if selected_ship:
            crop_img = dict(st.session_state.ship_crops)[selected_ship]
            st.image(crop_img, caption=selected_ship, use_column_width=False, width=500)
            for entry in st.session_state.metadata:
                if entry['ship_id'] == selected_ship:
                    pixel_area = entry['pixel_area']
                    surface_m2 = entry['surface_m2']
                    break
            st.markdown(f"**Pixel Area:** {pixel_area} px²  \
**Surface:** {surface_m2} m²")
            crop_buf = io.BytesIO()
            crop_img.save(crop_buf, format="JPEG")
            st.download_button("Download", data=crop_buf.getvalue(), file_name=f"{selected_ship}.jpg", key="download_crop")

        # Display metadata table after subimages
        df = pd.DataFrame(st.session_state.metadata)
        st.markdown("### 🧮 Ship Characteristics Table")
        show_all = st.checkbox("See full table", value=False, key="show_table")
        if show_all:
            st.dataframe(df)
        else:
            st.dataframe(df.head(5))

st.markdown('</div>', unsafe_allow_html=True)
