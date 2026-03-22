
import streamlit as st
from PIL import Image
import numpy as np

from utils.preprocess import validate_image
from utils.forgery_checks import check_blur
from utils.ocr import extract_text
from utils.yolo_detect import detect_regions
from utils.cnn_model import cnn_predict
from utils.metadata import get_metadata
from utils.report import generate_report

st.set_page_config(page_title="ID Forgery Detection", page_icon="🕵️")

st.markdown("""
    <style>
    .report-box {
        background-color: #000000;
        border: 1px solid #00ff00;
        border-radius: 8px;
        padding: 20px;
        font-family: monospace;
    }
    .section-title {
        color: #00ff00;
        font-size: 13px;
        border-bottom: 1px solid #1a4d1a;
        padding-bottom: 6px;
        margin-bottom: 8px;
        margin-top: 14px;
    }
    .bullet {
        color: #00ff00;
        font-size: 13px;
        padding: 3px 0;
    }
    .bullet::before {
        content: "► ";
        color: #00ff00;
    }
    .label { color: #88cc88; }
    .value-safe { color: #00ff00; }
    .value-danger { color: #ff4444; }
    </style>
""", unsafe_allow_html=True)

st.title("🕵️ ID Forgery Detection System")

uploaded_file = st.file_uploader("Upload ID Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if not validate_image(image):
        st.error("Invalid image format")
    else:
        with st.spinner("Analyzing ID..."):
            img_np = np.array(image)

            regions   = detect_regions(img_np)
            cnn_result = cnn_predict(img_np)
            blur_score = check_blur(img_np)
            text       = extract_text(img_np)
            metadata   = get_metadata(uploaded_file)
            report     = generate_report(blur_score, text, regions, cnn_result, metadata)

        st.subheader("📊 Fraud Detection Report")

        def is_danger(value):
            danger_words = ["fake", "forged", "tampered", "invalid", "fail",
                            "suspicious", "anomaly", "yes"]
            return any(word in str(value).lower() for word in danger_words)

        def render_bullet(label, value):
            css = "value-danger" if is_danger(value) else "value-safe"
            return f'<div class="bullet"><span class="label">{label}: </span><span class="{css}">{value}</span></div>'

        def render_section(title, items):
            html = f'<div class="section-title">🔍 {title}</div>'
            if isinstance(items, dict):
                for k, v in items.items():
                    html += render_bullet(k, v)
            else:
                css = "value-danger" if is_danger(items) else "value-safe"
                html += f'<div class="bullet"><span class="{css}">{items}</span></div>'
            return html

        # Build full report HTML
        html = '<div class="report-box">'

        # Blur
        html += render_section("Blur Detection", {
            "Blur Score": f"{blur_score:.2f}" if isinstance(blur_score, float) else blur_score
        })

        # OCR
        html += render_section("OCR Extracted Text", {
            "Text": text if text else "No text detected"
        })

        # YOLO
        html += render_section("YOLO Detected Regions", {
            "Regions Found": regions if regions else "None"
        })

        # CNN
        if isinstance(cnn_result, dict):
            html += render_section("CNN Prediction", cnn_result)
        else:
            html += render_section("CNN Prediction", {"Result": cnn_result})

        # Metadata
        if isinstance(metadata, dict):
            html += render_section("Metadata Analysis", metadata)
        else:
            html += render_section("Metadata Analysis", {"Info": metadata})

        # Final Report
        if isinstance(report, dict):
            html += render_section("Final Report", report)
        else:
            html += render_section("Final Report", {"Verdict": report})

        html += '</div>'

        st.markdown(html, unsafe_allow_html=True)