import streamlit as st
from PIL import Image
import base64

# Helper function to resize image by height
fixed_height = 400


def resize_height(img, height):
    w, h = img.size
    new_w = int(w * (height / h))
    return img.resize((new_w, height))


# Load images with consistent resizing
raw_img = resize_height(Image.open("assets/raw_img_2.png"), fixed_height)
gee_img = Image.open("assets/ee_earth_satellite.png").resize((500, 364))
stats_img = resize_height(Image.open("assets/stats4.png"), fixed_height)

# Hide sidebar
st.set_page_config(page_title="Main", layout="wide", initial_sidebar_state="collapsed")

# CSS to set everything to black background
custom_css = """
    <style>
        /* Hide sidebar button */
        [data-testid="collapsedControl"] {
            display: none;
        }

        /* Set entire page to black background */
        html, body, [class*="stApp"] {
            background-color: #000000;
        }

        /* Options container */
        .option-container {
            background-color: #000000;
            padding: 40px;
            border-radius: 15px;
            text-align: center;
            color: white;
        }

        .option-container img {
            height: 10px;
            object-fit: contain;
            margin-bottom: 10px;
            }

        .description {
            margin-top: 10px;
            margin-bottom: 20px;
            font-size: 16px;
            color: #cccccc;
        }

        .stButton > button {
            background-color: #00BFFF;
            color: white;
            border: none;
            padding: 0.75em 2em;
            font-size: 16px;
            border-radius: 10px;
            cursor: pointer;
        }

        .stButton > button:hover {
            background-color: #009ACD;
            color: white;
        }
        .stButton > button:active {
            background-color: #009ACD;
            color: white;
        }
        .icon{
            color:#ffffff;!important;
        }


        .stImage {
            display: flex;
            justify-content: center;
        }
        .stImage img {
            max-height: 300px;
            max-width: 100%;
            width: auto;
            height: auto;
            object-fit: contain;
            border-radius: 10px;
        }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Load logo for header
logo_data = None
try:
    with open("assets/logo.png", "rb") as img_file:
        logo_data = base64.b64encode(img_file.read()).decode()
except Exception:
    pass

# --- HEADER SECTION ---
st.markdown(f'''
    <div style="display: flex; align-items: center; margin-bottom: 30px;">
        {f'<img src="data:image/png;base64,{logo_data}" style="height: 15vh; margin-right: 20px;">' if logo_data else ''}
        <div>
            <h1 style="color:#1e90ff; font-family: 'Kanit', Arial, sans-serif; margin-bottom: 0;">BlueGuard</h1>
        </div>
    </div>
    <hr style="border: 1px solid #E9E9E9; margin-bottom: 40px;">
''', unsafe_allow_html=True)

# === Layout with three columns ===
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <h2 style="color: #ffffff; font-size: 1.5em; font-family: 'Kanit', Arial, sans-serif; margin-bottom: 10px;">
    <img src='https://img.icons8.com/ios-filled/50/ffffff/satellite.png' height='48' style='margin-bottom: 8px;'>
    Process Raw SAR Data
    </h2>
    <div class="description" style="font-size: 1.1em; color: #cccccc; margin-bottom: 18px;">
    Detailed detection on multi-polarizations and resolutions.
    </div>
    """, unsafe_allow_html=True)
    st.image(raw_img, caption="Raw Data")
    if st.button("Get started", key="raw"):
        st.switch_page("pages/app.py")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown("""
    <h2 style="color: #ffffff; font-size: 1.5em; font-family: 'Kanit', Arial, sans-serif; margin-bottom: 10px;">
    <img src='https://img.icons8.com/ios-filled/50/ffffff/globe-earth.png' height='48' style='margin-bottom: 8px;'>
    Explore GEE SAR Collection
    </h2>
    <div class="description" style="font-size: 1.1em; color: #cccccc; margin-bottom: 18px;">Just select your AOI and you will get a detailed report.</div>
    """, unsafe_allow_html=True)
    st.image(gee_img, caption="Google Earth Engine")
    if st.button("Get started", key="gee1"):
        st.switch_page("pages/earthEngine.py")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown("""
    <h2 style="color: #ffffff; font-size: 1.5em; font-family: 'Kanit', Arial, sans-serif; margin-bottom: 10px;">
    <img src='https://img.icons8.com/ios-filled/100/ffffff/combo-chart.png' height='48' style='margin-bottom: 8px;'>
    Model metrics
    </h2>
    <div class="description" style="font-size: 1.1em; color: #cccccc; margin-bottom: 18px;">Visualize and analyze the model's metrics.</div>
    """, unsafe_allow_html=True)
    st.image(stats_img, caption="Insights")
    if st.button("Get started", key="gee"):
        st.switch_page("pages/insights.py")
    st.markdown('</div>', unsafe_allow_html=True)


