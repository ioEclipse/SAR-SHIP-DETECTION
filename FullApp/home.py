import streamlit as st
import base64

# Config de la page
st.set_page_config(
    page_title="Ship Tracker",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Fonction pour convertir une image en base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Charger les images
logo_data = get_base64_image("assets/logo.png")
background_data = get_base64_image("assets/home_background.png")

# CSS + fond + overlay + style boutons
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap');
    .stApp {{
        background-image: url("data:image/png;base64,{background_data}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    .overlay {{
        position: absolute;
        top: 60%;
        left: 5%;
        transform: translateY(50%);
        transform: translateX(-20%);
        color: white;
        z-index: 999;
    }}
    .overlay h1 {{
        font-family: 'Arial', sans-serif;
        font-size: 56px;
        font-weight: 800;
        margin-bottom: 0.5em;
    }}
    .overlay p {{
        color: white;
        padding-top: 50px;
        font-size: 30px;
        line-height: 1.6;
        max-width: 400px;
        margin-bottom: 1.5em;
    }}
    #info_text{{
    font-family: "Kanit", sans-serif;
    }}
    div.stButton > button {{
    background-color: #1e90ff;
    border: 1px solid #1e90ff;
    padding: 0px 24px;
    border-radius: 25px;
    color: white !important;
    font-size: 16px;
    cursor: pointer;
    margin-right:0px;
    }}

    div.stButton > button:hover {{
        background-color: #3D90D7 !important;
        color: white !important;
        border: 1px solid #3D90D7;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# === Bloc principal : logo, titre, texte ===
st.markdown('<div class="overlay">', unsafe_allow_html=True)

# Titre + logo
st.markdown(
    f"""
    <div class="header" style="display:flex;">
        <img src="data:image/png;base64,{logo_data}" style="height: 6%; width:6%;">
        <h1 style="color:#1e90ff;font-size:7vh;">BlueGuard</h1>
    </div>
    <p style="color:white;margin-top:50px;margin-bottom:50px;font-size: 30px;" id = "info_text">
        AI-powered ship detection<br>using Sentinel-1 SAR data
    </p>
    """,
    unsafe_allow_html=True
)

# Deux boutons côte à côte
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Try Here", key="try_here"):
        st.switch_page("pages/main.py")



st.markdown("</div>", unsafe_allow_html=True)

# Rien d’autre à afficher
st.write("")
