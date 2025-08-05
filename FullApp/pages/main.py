import streamlit as st
from PIL import Image

# Charger et redimensionner manuellement les images
raw_img = Image.open("assets/raw_img.png").resize((500, 250))
gee_img = Image.open("assets/engine1.png").resize((500, 250))
# Masquer la sidebar
st.set_page_config(page_title="Main", layout="wide", initial_sidebar_state="collapsed")

# CSS pour tout passer en fond noir
custom_css = """
    <style>
        /* Cacher le bouton de sidebar */
        [data-testid="collapsedControl"] {
            display: none;
        }

        /* Mettre toute la page en fond noir */
        html, body, [class*="stApp"] {
            background-color: #000000;
        }

        /* Conteneur des options */
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
        }
    </style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# === Mise en page avec deux colonnes ===
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="option-container" style="padding-left:200px;">', unsafe_allow_html=True)
    st.markdown("""
<h2 style="
    color: white;
    font-size: 2em;
    text-align: ;
    padding-right:50px;
                padding-left:10px;
    font-family: Arial, sans-serif;
    text-shadow: 2px 2px 4px #aaa, 0 0 8px #ddd;">
    Process with raw SAR data
</h2>
""", unsafe_allow_html=True)

    st.image(raw_img, caption="Raw Data",  width=450)
    st.markdown('<div class="description">We provide a detailed detection on multi polarizations and resolutions</div>', unsafe_allow_html=True)
    if st.button("Get started", key="raw"):
        st.switch_page("pages/app.py")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="option-container">', unsafe_allow_html=True)
    st.markdown("""
<h2 style="
    color: white;
    font-size: 2em;
    text-align: ;
    padding-right:50px;
    font-family: Arial, sans-serif;
    text-shadow: 2px 2px 4px #aaa, 0 0 8px #ddd;">
    Explore GEE SAR Collection
</h2>
""", unsafe_allow_html=True)
    st.image(gee_img, caption="Google Earth Engine",  width=450)
    st.markdown('<div class="description">Just select your AOI and you will get a detailed report</div>', unsafe_allow_html=True)
    if st.button("Get started", key="gee"):
        st.switch_page("pages/earthEngine.py")
    st.markdown('</div>', unsafe_allow_html=True)
