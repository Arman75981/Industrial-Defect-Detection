"""
Streamlit Web Application for Multi-Category Industrial Defect Detection.
Supports all 15 categories from the MVTec Anomaly Detection benchmark.
Features category selection, category-specific models, and exactly two clean output cards:
Output 1: Anomaly Heatmap
Output 2: Defect Label & Reliable Explanation
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
import streamlit as st

# Ensure project root is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import importlib
import config
import src.anomaly_detection
import src.visualization
import src.label_manager

importlib.reload(config)
importlib.reload(src.anomaly_detection)
importlib.reload(src.visualization)
importlib.reload(src.label_manager)

from src.anomaly_detection import AnomalyDetector
from src.visualization import generate_annotated_detection_image
from src.label_manager import resolve_labels_and_details

# Page configuration - Minimalist clean layout
st.set_page_config(
    page_title="Industrial Defect Detection",
    page_icon="⚙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean, professional presentation
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0f172a;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 0.3rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    .app-subtitle {
        font-size: 0.98rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .output-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        height: 100%;
    }
    .card-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 14px;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 8px;
    }
    .badge-defective {
        display: inline-block;
        background-color: #fee2e2;
        color: #b91c1c;
        font-weight: 700;
        font-size: 1.25rem;
        padding: 6px 16px;
        border-radius: 8px;
        border: 1px solid #fecaca;
        margin-bottom: 16px;
    }
    .badge-normal {
        display: inline-block;
        background-color: #dcfce7;
        color: #15803d;
        font-weight: 700;
        font-size: 1.25rem;
        padding: 6px 16px;
        border-radius: 8px;
        border: 1px solid #bbf7d0;
        margin-bottom: 16px;
    }
    .field-title {
        font-weight: 600;
        color: #334155;
        margin-top: 10px;
        margin-bottom: 2px;
        font-size: 0.93rem;
    }
    .field-text {
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.45;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

def get_available_categories():
    """Returns only active categories that have trained model weights on disk."""
    active_cats = [
        c for c in config.ALL_CATEGORIES 
        if config.get_model_weights_path(c).exists()
    ]
    return sorted(active_cats) if active_cats else [config.DEFAULT_CATEGORY]

@st.cache_resource
def load_category_detector(category: str):
    """Loads and caches the category-specific AnomalyDetector."""
    weights_path = config.get_model_weights_path(category)
    if not weights_path.exists():
        return None
    try:
        return AnomalyDetector(category=category, scoring_mode="localized")
    except Exception as e:
        st.error(f"Error loading model for {category}: {e}")
        return None

def main():
    # Page Header
    st.markdown('<div class="app-title">Industrial Defect Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-subtitle">MVTec Anomaly Detection — Multi-Category Quality Control</div>', unsafe_allow_html=True)

    available_categories = get_available_categories()
    
    # Category Selector
    col_cat, col_status = st.columns([2, 1])
    with col_cat:
        category_display_names = [c.replace("_", " ").title() for c in available_categories]
        default_index = available_categories.index("screw") if "screw" in available_categories else 0
        selected_display = st.selectbox(
            "Select Component / Material Category:",
            options=category_display_names,
            index=default_index
        )
        selected_category = available_categories[category_display_names.index(selected_display)]

    # Check if category model is trained
    detector = load_category_detector(selected_category)
    
    with col_status:
        st.write("")
        st.write("")
        if detector is not None:
            st.success(f"Model Ready: {selected_display}")
        else:
            st.warning("Model Pending")

    if detector is None:
        st.info(
            f"⚠️ Model weights for **{selected_display}** (`models/autoencoder_{selected_category}.pth`) have not been trained yet.\n\n"
            f"To train an Autoencoder for this category, run:\n"
            f"```bash\npython src/train_autoencoder.py --category {selected_category}\n```"
        )
        return

    # Image Uploader
    uploaded_file = st.file_uploader(
        f"Upload an image of {selected_display} (PNG, JPG, JPEG):", 
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is None:
        st.info(f"Please upload an image of {selected_display} above to inspect.")
        return

    # Load and process image
    try:
        file_bytes = uploaded_file.getvalue()
        input_image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Unable to read the uploaded image: {e}")
        return

    # Run Inference
    with st.spinner(f"Analyzing {selected_display} with {selected_display} Autoencoder..."):
        result = detector.predict(input_image)

    is_anomaly = result["is_anomaly"]
    status_label = "Defective" if is_anomaly else "Normal"
    heatmap_rgb = result["heatmap_rgb"]

    # Defect labels, ground-truth match, and details
    details = resolve_labels_and_details(
        category=selected_category,
        image_input=input_image,
        file_bytes=file_bytes,
        is_anomaly=is_anomaly,
        location_text=result["location_text"]
    )

    # Generate Original Image with Red Border Box around detected defect
    orig_np = np.array(input_image)
    annotated_img, detected_boxes = generate_annotated_detection_image(
        original_image=orig_np,
        error_map=result["error_map"],
        threshold=result["threshold"],
        is_anomaly=is_anomaly,
        box_label=details["box_banner_label"]
    )

    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    # Exactly Two Output Cards Side-by-Side
    col_heatmap, col_label = st.columns(2)

    # Output 1: Anomaly Heatmap
    with col_heatmap:
        st.markdown('<div class="output-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Output 1 — Anomaly Heatmap</div>', unsafe_allow_html=True)
        st.image(heatmap_rgb, use_container_width=True)
        st.caption(f"Localized reconstruction anomaly heatmap for {selected_display}.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Output 2: Defect in Original Image
    with col_label:
        st.markdown('<div class="output-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Output 2 — Defect in Original Image</div>', unsafe_allow_html=True)
        
        # Original Image with Red Border Box around Defect
        st.image(annotated_img, use_container_width=True)
        if is_anomaly:
            st.caption(f"Defect highlighted in red border box ({details['box_banner_label']}).")
        else:
            st.caption("Pristine component — no anomalous regions detected.")

        # Status Badge
        badge_class = "badge-defective" if is_anomaly else "badge-normal"
        st.markdown(f'<div class="{badge_class}">{status_label}</div>', unsafe_allow_html=True)

        # What was detected
        st.markdown('<div class="field-title">Detection:</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-text">{details["detected_label"]}</div>', unsafe_allow_html=True)

        # Where it was detected
        st.markdown('<div class="field-title">Location:</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-text">{result["location_text"]}</div>', unsafe_allow_html=True)

        # Interpretation & Details
        st.markdown('<div class="field-title">Details:</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-text">{details["detection_details"]}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
