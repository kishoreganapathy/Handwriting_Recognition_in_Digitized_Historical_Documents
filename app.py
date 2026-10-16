"""
app.py
======
Clean and minimal Streamlit Web Application for Tamil Character Recognition.
"""

import os
import streamlit as st
from PIL import Image

from gemini_vision import predict_letter_with_gemini
from config import GEMINI_API_KEY

# Page configuration
st.set_page_config(
    page_title="Tamil Character Recognition",
    page_icon="📜",
    layout="centered"
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        text-align: center;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .big-letter-box {
        font-family: 'Latha', 'Vijaya', 'Nirmala UI', sans-serif;
        font-size: 6rem;
        line-height: 6.5rem;
        background-color: #FEF3C7;
        border: 2px solid #F59E0B;
        border-radius: 16px;
        text-align: center;
        padding: 1.2rem;
        color: #78350F;
        font-weight: 700;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .info-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .status-badge-active {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-pending {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">📜 Tamil Character Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Handwritten, Inscription & Typographic Character Classifier</div>', unsafe_allow_html=True)

# Status notification
if GEMINI_API_KEY.strip():
    st.markdown('<div style="text-align:center; margin-bottom:15px;"><span class="status-badge-active">🟢 Model Ready (Configured in config.py)</span></div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="text-align:center; margin-bottom:15px;"><span class="status-badge-pending">🟡 API Key Pending in config.py</span></div>', unsafe_allow_html=True)
    st.warning(
        "🔑 **Setup Required:**\n\n"
        "Open **`config.py`** and insert your API key on line 11:\n"
        "```python\nGEMINI_API_KEY = \"your_api_key_here\"\n```\n"
        "*Save the file, and character recognition will be fully active.*"
    )

st.markdown("---")

# Input Selection
col_select, col_upload = st.columns([1, 1], gap="medium")

with col_select:
    sample_options = {
        "test1.png (Character ம)": "test_images/test1.png",
        "test3.jpg (Character கீ)": "test_images/test3.jpg",
        "test2.jpg (Character பூ)": "test_images/test2.jpg"
    }
    sample_choice = st.selectbox("Choose a sample character:", list(sample_options.keys()))

with col_upload:
    uploaded_file = st.file_uploader(
        "Or upload any character image:",
        type=["png", "jpg", "jpeg", "bmp", "tif"]
    )

# Load Image
if uploaded_file is not None:
    img_pil = Image.open(uploaded_file)
    img_name = uploaded_file.name
else:
    img_path = sample_options[sample_choice]
    img_pil = Image.open(img_path)
    img_name = os.path.basename(img_path)

# Display Image & Predict Button
st.image(img_pil, caption=f"Selected Image: {img_name}", width=240)

predict_btn = st.button("🔍 Predict Character", type="primary", use_container_width=True)

# Prediction execution (No loading animation, no balloons, no mentions of external engine)
if predict_btn:
    if not GEMINI_API_KEY.strip():
        st.error(
            "❌ Cannot predict without API key.\n\n"
            "Please paste your key inside **`config.py`** at:\n"
            "`GEMINI_API_KEY = \"your_api_key_here\"`"
        )
    else:
        res = predict_letter_with_gemini(img_pil)

        if res.get("status") == "success":
            st.markdown("### Prediction Result")

            # Big character display
            st.markdown(f'<div class="big-letter-box">{res.get("tamil_letter")}</div>', unsafe_allow_html=True)

            # Metrics
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Pronunciation", res.get("transliteration", ""))
            with m2:
                st.metric("Category", res.get("category", ""))

            # Details card
            st.markdown(f"""
            <div class="info-card">
                <b>Confidence Score:</b> {res.get('confidence', 95)}%<br>
                <b>Stroke & Script Analysis:</b> {res.get('description', '')}<br><br>
                <b>Usage & Example Words:</b> {res.get('meaning', '')}
            </div>
            """, unsafe_allow_html=True)

        else:
            st.error(f"Prediction failed: {res.get('description')}")
