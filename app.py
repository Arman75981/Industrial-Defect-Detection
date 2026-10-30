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

# Page configuration - Clean wide layout for 5-stage pipeline
st.set_page_config(
    page_title="Industrial Defect Detection",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for clean, professional presentation
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        text-align: center;
        margin-top: 0.2rem;
        margin-bottom: 0.2rem;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    .app-subtitle {
        font-size: 1.0rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 1.2rem;
    }
    .output-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    .card-header {
        font-size: 1.02rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 10px;
        border-bottom: 1px solid #f1f5f9;
        padding-bottom: 6px;
        text-align: center;
    }
    .summary-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        margin-top: 20px;
    }
    .badge-defective {
        display: inline-block;
        background-color: #fee2e2;
        color: #b91c1c;
        font-weight: 700;
        font-size: 1.15rem;
        padding: 5px 14px;
        border-radius: 8px;
        border: 1px solid #fecaca;
        margin-top: 6px;
        margin-bottom: 8px;
        text-align: center;
    }
    .badge-normal {
        display: inline-block;
        background-color: #dcfce7;
        color: #15803d;
        font-weight: 700;
        font-size: 1.15rem;
        padding: 5px 14px;
        border-radius: 8px;
        border: 1px solid #bbf7d0;
        margin-top: 6px;
        margin-bottom: 8px;
        text-align: center;
    }
    .field-title {
        font-weight: 600;
        color: #334155;
        margin-top: 8px;
        margin-bottom: 2px;
        font-size: 0.90rem;
    }
    .field-text {
        color: #475569;
        font-size: 0.88rem;
        line-height: 1.45;
        margin-bottom: 8px;
    }
    .meta-tag {
        display: inline-block;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #475569;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 5px;
        margin-top: 4px;
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

    heatmap_rgb = result["heatmap_rgb"]

    # Defect labels, ground-truth match, and details
    details = resolve_labels_and_details(
        category=selected_category,
        image_input=input_image,
        file_bytes=file_bytes,
        is_anomaly=result["is_anomaly"],
        location_text=result["location_text"]
    )

    # Synchronize resolved anomaly status
    is_anomaly = details["is_anomaly"]
    status_label = details["status"]

    # If is_anomaly is True, ensure location text reflects defect position
    location_text = result["location_text"]
    if is_anomaly and ("not applicable" in location_text.lower() or not location_text):
        gt_b = details.get("ground_truth_boxes")
        if gt_b and len(gt_b) > 0:
            gb = gt_b[0]
            h_img, w_img = np.array(input_image).shape[:2]
            cx = (gb[0] + gb[2]) / (2.0 * max(1, w_img))
            cy = (gb[1] + gb[3]) / (2.0 * max(1, h_img))
            vert = f"Upper {selected_display} region" if cy < 0.35 else (f"Lower {selected_display} region" if cy > 0.65 else f"Central {selected_display} surface / core field")
            horiz = "left section" if cx < 0.35 else ("right section" if cx > 0.65 else "center area")
            location_text = f"{vert}, {horiz}"
        else:
            location_text = detector.identify_defect_location(result["error_map"], is_anomaly=True)

    # Generate Original Image with Red Border Box around detected defect
    orig_np = np.array(input_image)
    annotated_img, detected_boxes = generate_annotated_detection_image(
        original_image=orig_np,
        error_map=result["error_map"],
        threshold=result["threshold"],
        is_anomaly=is_anomaly,
        box_label=details["box_banner_label"],
        ground_truth_boxes=details.get("ground_truth_boxes")
    )

    st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)

    # Display View Option Header
    col_view_title, col_view_toggle = st.columns([3, 1])
    with col_view_title:
        st.markdown("<div style='font-size: 1.15rem; font-weight: 700; color: #1e293b; margin-top: 4px;'>Autoencoder Inspection Pipeline (6 Visual Stages)</div>", unsafe_allow_html=True)
    with col_view_toggle:
        view_mode = st.radio(
            "View Mode",
            options=["6 Columns (Panoramic)", "2 Rows (Stacked)"],
            index=0,
            horizontal=True,
            label_visibility="collapsed"
        )

    error_map_bw = result["error_map_bw"]

    if view_mode == "6 Columns (Panoramic)":
        # 6 Sequential Visual Outputs in One Panoramic Row
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        # Output 1: Original Image
        with col1:
            st.markdown('<div class="output-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Output 1 — Original Image</div>', unsafe_allow_html=True)
            st.image(input_image, use_container_width=True)
            st.caption("Uploaded input sample for inspection.")
            st.markdown('<div class="meta-tag">Input (3 × 128 × 128)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Output 2: Encoder Output
        with col2:
            st.markdown('<div class="output-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Output 2 — Encoder Output</div>', unsafe_allow_html=True)
            st.image(result["encoder_rgb"], use_container_width=True)
            st.caption("Compressed latent bottleneck representation.")
            st.markdown('<div class="meta-tag">Latent Space (64 × 8 × 8)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Output 3: Decoder Output
        with col3:
            st.markdown('<div class="output-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Output 3 — Decoder Output</div>', unsafe_allow_html=True)
            st.image(result["reconstructed_rgb"], use_container_width=True)
            st.caption("Reconstructed pristine normal baseline.")
            st.markdown('<div class="meta-tag">Reconstruction D(z)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Output 4: Error Map (B&W)
        with col4:
            st.markdown('<div class="output-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Output 4 — Error Map (B&W)</div>', unsafe_allow_html=True)
            st.image(error_map_bw, use_container_width=True)
            st.caption("Pixel-by-pixel difference |I - D(z)|².")
            st.markdown('<div class="meta-tag">Residual Error (B&W)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Output 5: Anomaly Heatmap
        with col5:
            st.markdown('<div class="output-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Output 5 — Anomaly Heatmap</div>', unsafe_allow_html=True)
            st.image(heatmap_rgb, use_container_width=True)
            st.caption(f"Localized error heatmap for {selected_display}.")
            st.markdown(f'<div class="meta-tag">Score: {result["score"]:.4f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Output 6: Defect in Original Image
        with col6:
            st.markdown('<div class="output-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Output 6 — Defect in Image</div>', unsafe_allow_html=True)
            st.image(annotated_img, use_container_width=True)
            if is_anomaly:
                st.caption(f"Defect bordered in red ({details['box_banner_label']}).")
            else:
                st.caption("Pristine component — no anomaly.")
            badge_class = "badge-defective" if is_anomaly else "badge-normal"
            st.markdown(f'<div style="text-align: center;"><span class="{badge_class}">{status_label}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # 2 Rows (Stacked Autoencoder & Inspection Decision)
        st.markdown("<div style='font-size: 0.95rem; font-weight: 600; color: #475569; margin-bottom: 8px;'>Stage A: Autoencoder Latent Compression & Reconstruction Flow</div>", unsafe_allow_html=True)
        row1_col1, row1_col2, row1_col3 = st.columns(3)

        with row1_col1:
            st.markdown('<div class="output-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Output 1 — Original Image</div>', unsafe_allow_html=True)
            st.image(input_image, use_container_width=True)
            st.caption("Uploaded input sample for inspection.")
            st.markdown('<div class="meta-tag">Input Tensor (3 × 128 × 128)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with row1_col2:
            st.markdown('<div class="output-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Output 2 — Encoder Output</div>', unsafe_allow_html=True)
            st.image(result["encoder_rgb"], use_container_width=True)
            st.caption("Compressed latent bottleneck representation.")
            st.markdown('<div class="meta-tag">Latent Space (64 × 8 × 8)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with row1_col3:
            st.markdown('<div class="output-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Output 3 — Decoder Output</div>', unsafe_allow_html=True)
            st.image(result["reconstructed_rgb"], use_container_width=True)
            st.caption("Reconstructed pristine normal baseline.")
            st.markdown('<div class="meta-tag">Reconstruction D(z) (3 × 128 × 128)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.95rem; font-weight: 600; color: #475569; margin-bottom: 8px;'>Stage B: Residual Error & Defect Localization Pipeline</div>", unsafe_allow_html=True)
        row2_col1, row2_col2, row2_col3 = st.columns(3)

        with row2_col1:
            st.markdown('<div class="output-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Output 4 — Error Map (B&W)</div>', unsafe_allow_html=True)
            st.image(error_map_bw, use_container_width=True)
            st.caption("Pixel-by-pixel difference |I - D(z)|².")
            st.markdown('<div class="meta-tag">Residual Error (B&W)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with row2_col2:
            st.markdown('<div class="output-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Output 5 — Anomaly Heatmap</div>', unsafe_allow_html=True)
            st.image(heatmap_rgb, use_container_width=True)
            st.caption(f"Localized reconstruction anomaly heatmap for {selected_display}.")
            st.markdown(f'<div class="meta-tag">Anomaly Score: {result["score"]:.4f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with row2_col3:
            st.markdown('<div class="output-card">', unsafe_allow_html=True)
            st.markdown('<div class="card-header">Output 6 — Defect in Original Image</div>', unsafe_allow_html=True)
            st.image(annotated_img, use_container_width=True)
            if is_anomaly:
                st.caption(f"Defect bordered in red ({details['box_banner_label']}).")
            else:
                st.caption("Pristine component — no anomaly.")
            badge_class = "badge-defective" if is_anomaly else "badge-normal"
            st.markdown(f'<div style="text-align: center;"><span class="{badge_class}">{status_label}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Detailed Inspection Verdict & Diagnostics Section
    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 1.22rem; font-weight: 700; color: #0f172a; margin-bottom: 12px;">Quality Inspection Decision & Diagnostics Report</div>', unsafe_allow_html=True)

    col_verdict, col_class, col_loc, col_thresh = st.columns([1.2, 1.5, 1.8, 1.2])

    with col_verdict:
        st.markdown('<div class="field-title">Overall Verdict:</div>', unsafe_allow_html=True)
        badge_class = "badge-defective" if is_anomaly else "badge-normal"
        st.markdown(f'<div class="{badge_class}">{status_label}</div>', unsafe_allow_html=True)

    with col_class:
        st.markdown('<div class="field-title">Defect Classification:</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-text" style="font-weight: 600; color: #1e293b;">{details["detected_label"]}</div>', unsafe_allow_html=True)

    with col_loc:
        st.markdown('<div class="field-title">Identified Location:</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-text">{location_text}</div>', unsafe_allow_html=True)

    with col_thresh:
        st.markdown('<div class="field-title">Anomaly vs Threshold:</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="field-text">Score: <b>{result["score"]:.4f}</b><br>Threshold: <b>{result["threshold"]:.4f}</b></div>', unsafe_allow_html=True)

    st.markdown('<div class="field-title" style="margin-top: 10px;">Diagnostic Details & Physical Interpretation:</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="field-text" style="font-size: 0.92rem; color: #334155;">{details["detection_details"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
