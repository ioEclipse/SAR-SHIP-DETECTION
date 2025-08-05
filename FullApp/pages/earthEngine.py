import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import Draw

# Configuration de la page
st.set_page_config(
    page_title="SAR Map Viewer",
    layout="wide"
)

# --- Sidebar ---
st.sidebar.title("Filters")

# 1. Choix de l'année
year = st.sidebar.selectbox("Select Year", ["2022", "2023", "2024", "2025"])

# 2. Choix du mois
month = st.sidebar.selectbox("Select Month", [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
])

# 3. Bande VV / VH
band = st.sidebar.selectbox("Select Band", ["VV", "VH"])



# Créer la carte centrée sur un point (ex: Tunisie)
m = folium.Map(location=[36.8, 10.2], zoom_start=6)

# Ajouter un bouton de dessin de polygone
draw = Draw(
    export=True,
    filename='polygon.geojson',
    draw_options={
        'polyline': False,
        'rectangle': False,
        'circle': False,
        'circlemarker': False,
        'marker': False,
        'polygon': {
            'allowIntersection': False,
            'showArea': True,
            'drawError': {'color': '#e1e100', 'message': "Invalid polygon"},
            'shapeOptions': {'color': '#97009c'}
        }
    }
)
draw.add_to(m)

# Afficher la carte dans Streamlit
map_data = st_folium(m, width=3000, height=600)

# Afficher les coordonnées si un polygone est dessiné
if map_data and map_data.get("last_drawn_geojson"):
    st.subheader("Polygon Coordinates (GeoJSON)")
    st.json(map_data["last_drawn_geojson"])
