import streamlit as st
import base64

# Page config
st.set_page_config(page_title="Model Insights", layout="wide", initial_sidebar_state="collapsed")

def get_base64_image(image_path: str) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Assets
try:
    hero_bg_b64 = get_base64_image("assets/insights_background.png")
except Exception:
    hero_bg_b64 = None

# Styles
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: #0f0f0f !important;
            color: #ffffff !important;
        }}
        /* Hero */
        .hero {{
            {f'background-image: url("data:image/png;base64,{hero_bg_b64}");' if hero_bg_b64 else ''}
            background-size: cover;
            background-position: center;
            height: 180px;
            border-bottom: 1px solid #222;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
        }}
        .hero-inner {{
            width: 100%;
            margin: 0 auto;
            padding: 0 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }}
        .hero-title {{
            font-size: 150px;
            font-weight: 800;
            margin: 0;
            align-items: center;
            line-height: 1.05;
        }}
        /* Back to home button */
        .back-home-btn a {{
            display: inline-block;
            background: #1a1a1a;
            color: #eaeaea !important;
            padding: 8px 14px;
            border-radius: 10px;
            border: 1px solid #2a2a2a;
            text-decoration: none !important;
            transition: background 0.2s ease, color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease, transform 0.12s ease;
        }}
        .back-home-btn a:hover {{
            background: #2563eb;
            border-color: #2563eb;
            color: #ffffff !important;
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.25);
            transform: translateY(-1px);
        }}
        /* Content container */
        .page-container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px;
        }}
        /* Individual metric card styling */
        .metric-card {{
            position: relative;
            background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
            border: 1px solid #2a2a2a;
            border-radius: 14px;
            padding: 18px;
            transition: transform 0.15s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            box-shadow: 0 4px 14px rgba(0,0,0,0.25);
            height: 180px;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            width: 100%;
            margin-bottom: 20px;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 24px rgba(0,0,0,0.35);
            border-color: #3a3a3a;
        }}
        .metric-card::before {{
            content: "";
            position: absolute;
            top: -1px;
            left: -1px;
            right: -1px;
            height: 3px;
            border-radius: 14px 14px 0 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        }}
        .metric-header {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }}
        .metric-icon {{
            width: 32px;
            height: 32px;
            border-radius: 999px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
            color: #ffffff;
            box-shadow: 0 6px 16px rgba(0,0,0,0.25);
            flex-shrink: 0;
        }}
        .metric-title {{
            color: #cfcfcf;
            font-size: 13px;
            font-weight: 600;
            margin: 0;
            line-height: 1.2;
        }}
        .metric-value {{
            font-size: 26px;
            font-weight: 800;
            margin: 8px 0;
        }}
        .metric-desc {{
            color: #9a9a9a;
            font-size: 11px;
            line-height: 1.3;
            text-align: center;
        }}
        /* Color accents per card */
        .metric-card.metric-precision .metric-icon {{ background: linear-gradient(135deg, #6366f1, #22d3ee); }}
        .metric-card.metric-precision::before {{ background: linear-gradient(90deg, #6366f1, #22d3ee); }}
        .metric-card.metric-recall .metric-icon {{ background: linear-gradient(135deg, #10b981, #34d399); }}
        .metric-card.metric-recall::before {{ background: linear-gradient(90deg, #10b981, #34d399); }}
        .metric-card.metric-f1 .metric-icon {{ background: linear-gradient(135deg, #f59e0b, #f97316); }}
        .metric-card.metric-f1::before {{ background: linear-gradient(90deg, #f59e0b, #f97316); }}
        .metric-card.metric-map .metric-icon {{ background: linear-gradient(135deg, #d946ef, #fb7185); }}
        .metric-card.metric-map::before {{ background: linear-gradient(90deg, #d946ef, #fb7185); }}
        .metric-card.metric-iou .metric-icon {{ background: linear-gradient(135deg, #0ea5e9, #22d3ee); }}
        .metric-card.metric-iou::before {{ background: linear-gradient(90deg, #0ea5e9, #22d3ee); }}
        .footer-note {{
            margin-top: 24px;
            color: #8e8e8e;
            font-size: 12px;
            text-align: center;
        }}
        /* Hide Streamlit default elements */
        .stColumn > div {{
            padding: 0 10px;
        }}
        @media (max-width: 900px) {{
            .hero-title {{
                font-size: 8vh !important;
            }}
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Hero header
st.markdown(
    """
    <div class="hero">
      <div class="hero-inner">
        <div class="hero-title" style="font-size: 10vh; font-weight: 800;">Learn More</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Back link
_, col_link, _ = st.columns([5, 2, 4])
with col_link:
    st.markdown('<div class="back-home-btn">', unsafe_allow_html=True)
    st.page_link("home.py", label="Back to home")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="page-container">', unsafe_allow_html=True)

# Metrics data
metrics = [
    ("Precision", "Proportion of predicted ships that are correct", 0.92),
    ("Recall", "Proportion of actual ships detected", 0.99),
    ("F1-score", "Harmonic mean of precision and recall", 0.91),
    ("mAP", "Mean average precision across all classes", 0.96),
    ("IoU", "Overlap between truth and prediction", 0.79),
]

icon_lookup = {
    "Precision": "🎯",
    "Recall": "📡",
    "F1-score": "⚖️",
    "mAP": "📈",
    "IoU": "🧩",
}

css_class_lookup = {
    "Precision": "metric-precision",
    "Recall": "metric-recall",
    "F1-score": "metric-f1",
    "mAP": "metric-map",
    "IoU": "metric-iou",
}

# Create columns for metric cards using Streamlit's native column system
cols = st.columns(5, gap="medium")

# Render metric cards in columns
for i, (title, desc, value) in enumerate(metrics):
    with cols[i]:
        icon = icon_lookup.get(title, "📊")
        css_class = css_class_lookup.get(title, "")
        
        card_html = f"""
            <div class="metric-card {css_class}">
                <div class="metric-header">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-title">{title}</div>
                </div>
                <div class="metric-value">{value:.2f}</div>
                <div class="metric-desc">{desc}</div>
            </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)