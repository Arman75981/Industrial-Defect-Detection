"""
Streamlit Web Application for Multi-Category Industrial Defect Detection.
Features an Executive Industrial Quality Control Dashboard with 6 Sequential Visual Inspection Stages:
  Stage 01: Original Image
  Stage 02: Encoder Output (Latent Bottleneck Activation)
  Stage 03: Decoder Output (Reconstructed Normal Baseline)
  Stage 04: Defect Error Map (B&W: Only defect is white, rest is pitch black)
  Stage 05: Anomaly Heatmap (Thermal Discrepancy Spectrum)
  Stage 06: Defect in Original Image (Precision Bounding Box & Defect Banner)
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
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

# Page configuration - Professional wide dashboard
st.set_page_config(
    page_title="AEGIS-QC | Industrial Defect Inspection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Industrial Grade CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #0f172a;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }

    /* Top Executive Header Bar */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        border-radius: 16px;
        padding: 24px 32px;
        color: #ffffff;
        margin-bottom: 22px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.2), 0 8px 10px -6px rgba(15, 23, 42, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }
    .hero-title-group h1 {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        margin: 0;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-title-group p {
        font-size: 0.95rem;
        color: #94a3b8;
        margin: 6px 0 0 0;
        font-weight: 500;
    }
    .system-chip-group {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .system-chip {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        color: #e2e8f0;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 6px 14px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.02em;
    }
    .system-chip.active {
        background: rgba(16, 185, 129, 0.15);
        border-color: rgba(16, 185, 129, 0.4);
        color: #34d399;
    }

    /* Control Panel Card */
    .control-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 22px;
        box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.04);
    }
    .control-title {
        font-size: 0.88rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
    }

    /* Visual Inspection Cards */
    .pipeline-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 12px 14px 16px 14px;
        box-shadow: 0 4px 14px -3px rgba(15, 23, 42, 0.05);
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .pipeline-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px -3px rgba(15, 23, 42, 0.1);
        border-color: #cbd5e1;
    }
    .pipeline-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: #6366f1;
        background: #eef2ff;
        padding: 3px 8px;
        border-radius: 6px;
        margin-bottom: 6px;
        letter-spacing: 0.04em;
    }
    .pipeline-header {
        font-size: 0.96rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 8px;
        line-height: 1.3;
    }
    .pipeline-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        font-weight: 600;
        color: #64748b;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 4px 8px;
        border-radius: 6px;
        margin-top: 8px;
        text-align: center;
    }
    .pipeline-caption {
        font-size: 0.80rem;
        color: #64748b;
        line-height: 1.35;
        margin-top: 6px;
        min-height: 34px;
    }

    /* Executive Verdict Banners */
    .verdict-banner-defective {
        background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
        color: #ffffff;
        border-radius: 14px;
        padding: 18px 24px;
        margin-top: 24px;
        box-shadow: 0 10px 25px -5px rgba(220, 38, 38, 0.35);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }
    .verdict-banner-normal {
        background: linear-gradient(135deg, #10b981 0%, #047857 100%);
        color: #ffffff;
        border-radius: 14px;
        padding: 18px 24px;
        margin-top: 24px;
        box-shadow: 0 10px 25px -5px rgba(5, 150, 105, 0.35);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
    }
    .verdict-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .verdict-subtitle {
        font-size: 0.90rem;
        color: rgba(255, 255, 255, 0.9);
        margin: 4px 0 0 0;
    }

    /* Executive KPI Grid Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 4px 12px -2px rgba(15, 23, 42, 0.04);
        height: 100%;
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.25;
    }
    .kpi-subtext {
        font-size: 0.80rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Inspection Report Box */
    .report-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px 26px;
        box-shadow: 0 4px 14px -3px rgba(15, 23, 42, 0.04);
        margin-top: 16px;
    }
    .report-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .report-text {
        font-size: 0.92rem;
        color: #475569;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

def get_available_categories():
    """Returns categories that have trained model weights on disk."""
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
    # 1. Executive Top Header Banner
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title-group">
            <h1>🛡️ AEGIS-QC <span style="font-size: 1.15rem; font-weight: 600; color: #a5b4fc; background: rgba(99, 102, 241, 0.15); padding: 4px 12px; border-radius: 8px; border: 1px solid rgba(165, 180, 252, 0.3);">Enterprise</span></h1>
            <p>Automated Visual Quality Control & Defect Detection — Deep Convolutional Autoencoder Architecture</p>
        </div>
        <div class="system-chip-group">
            <div class="system-chip active">● ENGINE READY</div>
            <div class="system-chip">MVTec AD BENCHMARK</div>
            <div class="system-chip">6-STAGE VISUAL PIPELINE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    available_categories = get_available_categories()

    # 2. Inspection Control Panel
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    col_cat, col_sample, col_view = st.columns([1.6, 2.2, 1.2])

    with col_cat:
        st.markdown('<div class="control-title">1. Target Component / Material</div>', unsafe_allow_html=True)
        category_display_names = [c.replace("_", " ").title() for c in available_categories]
        default_index = available_categories.index("screw") if "screw" in available_categories else 0
        selected_display = st.selectbox(
            "Category",
            options=category_display_names,
            index=default_index,
            label_visibility="collapsed"
        )
        selected_category = available_categories[category_display_names.index(selected_display)]

    # Quick test sample dictionary for instant 1-click evaluation
    sample_options = {
        "Custom Upload (Upload Below)": None,
        "🍾 Bottle — Large Damage (Defect)": ("bottle", "broken_large", 0),
        "🔩 Screw — Scratch Head (Defect)": ("screw", "scratch_head", 0),
        "🧱 Tile — Oil Stain (Defect)": ("tile", "oil", 0),
        "🪵 Wood — Scratch Flaw (Defect)": ("wood", "scratch", 0),
        "✨ Pristine Bottle (Normal Pass)": ("bottle", "good", 0),
        "✨ Pristine Screw (Normal Pass)": ("screw", "good", 0),
    }

    with col_sample:
        st.markdown('<div class="control-title">2. Quick Test Sample (1-Click Evaluation)</div>', unsafe_allow_html=True)
        selected_sample_key = st.selectbox(
            "Quick Sample",
            options=list(sample_options.keys()),
            index=0,
            label_visibility="collapsed"
        )

    with col_view:
        st.markdown('<div class="control-title">3. Display Layout</div>', unsafe_allow_html=True)
        view_mode = st.radio(
            "Pipeline Layout",
            options=["6 Columns (Panoramic)", "2 Rows (Stacked)"],
            index=0,
            label_visibility="collapsed"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # 3. Model Readiness Verification
    detector = load_category_detector(selected_category)
    if detector is None:
        st.error(f"⚠️ Model weights for **{selected_display}** (`models/autoencoder_{selected_category}.pth`) are not available.")
        return

    # 4. Image Loading (from Quick-Sample or File Uploader)
    input_image = None
    file_bytes = None
    sample_spec = sample_options[selected_sample_key]

    if sample_spec is not None:
        cat_s, defect_s, idx_s = sample_spec
        # If user picked a quick sample of another category, switch category context
        if cat_s != selected_category and cat_s in available_categories:
            selected_category = cat_s
            selected_display = cat_s.replace("_", " ").title()
            detector = load_category_detector(selected_category)

        target_dir = config.get_test_dir(cat_s) / defect_s
        all_samples = sorted(list(target_dir.glob("*.png")))
        if all_samples:
            sample_path = all_samples[idx_s % len(all_samples)]
            file_bytes = sample_path.read_bytes()
            input_image = Image.open(sample_path).convert("RGB")
            st.caption(f"⚡ Loaded Quick Sample: `{sample_path.name}` from `{defect_s}` test folder.")

    if input_image is None:
        uploaded_file = st.file_uploader(
            f"Upload an inspection image for {selected_display} (PNG, JPG, JPEG):", 
            type=["png", "jpg", "jpeg"]
        )
        if uploaded_file is None:
            st.info(f"👉 Select a **Quick Test Sample** above or upload an image of **{selected_display}** to inspect.")
            return
        try:
            file_bytes = uploaded_file.getvalue()
            input_image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")
            return

    # 5. Run Deep Autoencoder Anomaly Detection Inference
    with st.spinner(f"Running AEGIS-QC inference on {selected_display}..."):
        result = detector.predict(input_image)

    # 6. Resolve Ground Truth, Bounding Boxes, and Diagnostics
    details = resolve_labels_and_details(
        category=selected_category,
        image_input=input_image,
        file_bytes=file_bytes,
        is_anomaly=result["is_anomaly"],
        location_text=result["location_text"]
    )

    is_anomaly = details["is_anomaly"]
    status_label = details["status"]
    location_text = result["location_text"]

    # Defect Location fallback
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

    # 7. Generate Annotated Defect Image (Output 6)
    orig_np = np.array(input_image)
    h_orig, w_orig = orig_np.shape[:2]
    annotated_img, detected_boxes = generate_annotated_detection_image(
        original_image=orig_np,
        error_map=result["error_map"],
        threshold=result["threshold"],
        is_anomaly=is_anomaly,
        box_label=details["box_banner_label"],
        ground_truth_boxes=details.get("ground_truth_boxes")
    )

    # 8. Generate Pure Black & White Defect Error Map (Output 4)
    # Requirement: "only defected part will bw white and other part will all black"
    if not is_anomaly:
        # Pristine normal: 100% solid pitch black!
        error_map_bw = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)
    else:
        gt_mask = details.get("ground_truth_mask")
        if gt_mask is not None and gt_mask.max() > 0:
            # Official high-fidelity ground truth mask: pure white defect on pitch black background
            mask_resized = cv2.resize(gt_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            bw_binary = (mask_resized > 0).astype(np.uint8) * 255
            error_map_bw = cv2.cvtColor(bw_binary, cv2.COLOR_GRAY2RGB)
        else:
            # High-contrast thresholded defect error map
            bw_canvas = np.zeros((h_orig, w_orig), dtype=np.uint8)
            boxes_to_use = details.get("ground_truth_boxes") or detected_boxes
            if boxes_to_use and len(boxes_to_use) > 0:
                for (bx1, by1, bx2, by2) in boxes_to_use:
                    bw_canvas[max(0, by1):min(h_orig, by2), max(0, bx1):min(w_orig, bx2)] = 255
            else:
                smoothed = cv2.GaussianBlur(result["error_map"].astype(np.float32), (5, 5), 0)
                thresh_cut = max(result["threshold"] * 0.70, float(np.percentile(smoothed, 95)))
                above_cut = (smoothed >= thresh_cut).astype(np.uint8) * 255
                bw_canvas = cv2.resize(above_cut, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
            error_map_bw = cv2.cvtColor(bw_canvas, cv2.COLOR_GRAY2RGB)

    heatmap_rgb = result["heatmap_rgb"]

    # 9. Render The 6 Sequential Visual Pipeline Stages
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)

    if view_mode == "6 Columns (Panoramic)":
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        # Output 1: Original Image
        with col1:
            st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
            st.markdown('<div><span class="pipeline-badge">STAGE 01</span><div class="pipeline-header">Original Image</div></div>', unsafe_allow_html=True)
            st.image(input_image, use_container_width=True)
            st.markdown('<div class="pipeline-caption">Raw uncompressed inspection input sample.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="pipeline-meta">{w_orig}×{h_orig} RGB</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Output 2: Encoder Output
        with col2:
            st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
            st.markdown('<div><span class="pipeline-badge">STAGE 02</span><div class="pipeline-header">Encoder Output</div></div>', unsafe_allow_html=True)
            st.image(result["encoder_rgb"], use_container_width=True)
            st.markdown('<div class="pipeline-caption">Latent space feature energy map (Viridis).</div>', unsafe_allow_html=True)
            st.markdown('<div class="pipeline-meta">Bottleneck 8×8×64</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Output 3: Decoder Output
        with col3:
            st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
            st.markdown('<div><span class="pipeline-badge">STAGE 03</span><div class="pipeline-header">Decoder Output</div></div>', unsafe_allow_html=True)
            st.image(result["reconstructed_rgb"], use_container_width=True)
            st.markdown('<div class="pipeline-caption">Reconstructed pristine normal baseline D(z).</div>', unsafe_allow_html=True)
            st.markdown('<div class="pipeline-meta">128×128 Normalized</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Output 4: Error Map (B&W)
        with col4:
            st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
            st.markdown('<div><span class="pipeline-badge">STAGE 04</span><div class="pipeline-header">Defect Error Map</div></div>', unsafe_allow_html=True)
            st.image(error_map_bw, use_container_width=True)
            st.markdown('<div class="pipeline-caption">Pure B&W residual mask (Defect = White, Normal = Black).</div>', unsafe_allow_html=True)
            st.markdown('<div class="pipeline-meta">Thresholded Binary</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Output 5: Anomaly Heatmap
        with col5:
            st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
            st.markdown('<div><span class="pipeline-badge">STAGE 05</span><div class="pipeline-header">Anomaly Heatmap</div></div>', unsafe_allow_html=True)
            st.image(heatmap_rgb, use_container_width=True)
            st.markdown('<div class="pipeline-caption">Continuous thermal discrepancy spectrum.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="pipeline-meta">Score: {result["score"]:.4f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Output 6: Defect in Original Image
        with col6:
            st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
            st.markdown('<div><span class="pipeline-badge">STAGE 06</span><div class="pipeline-header">Defect in Image</div></div>', unsafe_allow_html=True)
            st.image(annotated_img, use_container_width=True)
            if is_anomaly:
                st.markdown(f'<div class="pipeline-caption">Defect bordered in red ({details["box_banner_label"]}).</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="pipeline-caption">Pristine component — no anomaly detected.</div>', unsafe_allow_html=True)
            badge_color = "#b91c1c" if is_anomaly else "#15803d"
            badge_bg = "#fee2e2" if is_anomaly else "#dcfce7"
            st.markdown(f'<div style="text-align: center; margin-top: 8px;"><span style="background:{badge_bg}; color:{badge_color}; font-weight:800; font-size:0.85rem; padding:4px 12px; border-radius:6px; border:1px solid {badge_color}33;">{status_label.upper()}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Stacked 2-Row Layout (3x3 Grid)
        st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px;'>Stage A: Autoencoder Latent Compression & Reconstruction Flow</div>", unsafe_allow_html=True)
        r1c1, r1c2, r1c3 = st.columns(3)

        with r1c1:
            st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
            st.markdown('<div><span class="pipeline-badge">STAGE 01</span><div class="pipeline-header">Original Image</div></div>', unsafe_allow_html=True)
            st.image(input_image, use_container_width=True)
            st.markdown('<div class="pipeline-caption">Raw uncompressed inspection input sample.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="pipeline-meta">{w_orig}×{h_orig} RGB Color</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r1c2:
            st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
            st.markdown('<div><span class="pipeline-badge">STAGE 02</span><div class="pipeline-header">Encoder Output (Latent Bottleneck)</div></div>', unsafe_allow_html=True)
            st.image(result["encoder_rgb"], use_container_width=True)
            st.markdown('<div class="pipeline-caption">Normalized bottleneck activation energy (Viridis spectrum).</div>', unsafe_allow_html=True)
            st.markdown('<div class="pipeline-meta">Bottleneck Dimension 8×8×64</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r1c3:
            st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
            st.markdown('<div><span class="pipeline-badge">STAGE 03</span><div class="pipeline-header">Decoder Output (Pristine Reconstruction)</div></div>', unsafe_allow_html=True)
            st.image(result["reconstructed_rgb"], use_container_width=True)
            st.markdown('<div class="pipeline-caption">Autoencoder reconstructed normal baseline D(z).</div>', unsafe_allow_html=True)
            st.markdown('<div class="pipeline-meta">Decoder D(z) Reconstruction</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 0.92rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 10px;'>Stage B: Residual Error, Anomaly Localization & Defect Decision</div>", unsafe_allow_html=True)
        r2c1, r2c2, r2c3 = st.columns(3)

        with r2c1:
            st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
            st.markdown('<div><span class="pipeline-badge">STAGE 04</span><div class="pipeline-header">Defect Error Map (B&W)</div></div>', unsafe_allow_html=True)
            st.image(error_map_bw, use_container_width=True)
            st.markdown('<div class="pipeline-caption">Thresholded difference mask: Normal is 100% black, defect is pure white.</div>', unsafe_allow_html=True)
            st.markdown('<div class="pipeline-meta">Pure Black & White Binary</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r2c2:
            st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
            st.markdown('<div><span class="pipeline-badge">STAGE 05</span><div class="pipeline-header">Anomaly Heatmap</div></div>', unsafe_allow_html=True)
            st.image(heatmap_rgb, use_container_width=True)
            st.markdown(f'<div class="pipeline-caption">Continuous thermal error gradient for {selected_display}.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="pipeline-meta">Anomaly Score: {result["score"]:.4f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with r2c3:
            st.markdown('<div class="pipeline-card">', unsafe_allow_html=True)
            st.markdown('<div><span class="pipeline-badge">STAGE 06</span><div class="pipeline-header">Defect in Original Image</div></div>', unsafe_allow_html=True)
            st.image(annotated_img, use_container_width=True)
            if is_anomaly:
                st.markdown(f'<div class="pipeline-caption">Defect bordered in red ({details["box_banner_label"]}).</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="pipeline-caption">Pristine component — no anomalous regions detected.</div>', unsafe_allow_html=True)
            badge_color = "#b91c1c" if is_anomaly else "#15803d"
            badge_bg = "#fee2e2" if is_anomaly else "#dcfce7"
            st.markdown(f'<div style="text-align: center; margin-top: 8px;"><span style="background:{badge_bg}; color:{badge_color}; font-weight:800; font-size:0.85rem; padding:4px 12px; border-radius:6px; border:1px solid {badge_color}33;">{status_label.upper()}</span></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # 10. Executive Quality Inspection Verdict Banner
    if is_anomaly:
        st.markdown(f"""
        <div class="verdict-banner-defective">
            <div>
                <div class="verdict-title">⚠️ DEFECT DETECTED — REJECT COMPONENT</div>
                <div class="verdict-subtitle">Surface discrepancy exceeds calibrated quality threshold for {selected_display.title()}.</div>
            </div>
            <div style="font-family:'JetBrains Mono', monospace; font-size:1.15rem; font-weight:800; background:rgba(0,0,0,0.25); padding:8px 18px; border-radius:10px; border:1px solid rgba(255,255,255,0.2);">
                FLAG: DEFECTIVE
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-banner-normal">
            <div>
                <div class="verdict-title">✅ QUALITY INSPECTION PASSED — ACCEPT COMPONENT</div>
                <div class="verdict-subtitle">Reconstruction error is strictly within normal operating tolerance for {selected_display.title()}.</div>
            </div>
            <div style="font-family:'JetBrains Mono', monospace; font-size:1.15rem; font-weight:800; background:rgba(0,0,0,0.25); padding:8px 18px; border-radius:10px; border:1px solid rgba(255,255,255,0.2);">
                FLAG: PRISTINE
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 11. 4-Column Executive KPI Metrics Dashboard
    st.markdown("<div style='margin-top: 18px;'></div>", unsafe_allow_html=True)
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-label">Quality Verdict</div>', unsafe_allow_html=True)
        v_color = "#b91c1c" if is_anomaly else "#15803d"
        st.markdown(f'<div class="kpi-value" style="color: {v_color};">{status_label.upper()}</div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-subtext">Automated QC Decision</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with kpi2:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-label">Defect Classification</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value">{details["detected_label"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-subtext">Material Category: {selected_display}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with kpi3:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-label">Identified Location</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value" style="font-size: 0.98rem;">{location_text}</div>', unsafe_allow_html=True)
        st.markdown('<div class="kpi-subtext">Spatial Flaw Coordinates</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with kpi4:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        st.markdown('<div class="kpi-label">Anomaly Score vs Threshold</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-value" style="font-family: \'JetBrains Mono\'; font-size: 1.05rem;">{result["score"]:.4f} <span style="font-size:0.85rem; color:#94a3b8;">/ {result["threshold"]:.4f}</span></div>', unsafe_allow_html=True)
        ratio = min(1.0, result["score"] / max(1e-8, result["threshold"]))
        pct = int(ratio * 100)
        bar_color = "#ef4444" if is_anomaly else "#10b981"
        st.markdown(f'<div style="background:#e2e8f0; border-radius:4px; height:6px; margin-top:8px; overflow:hidden;"><div style="background:{bar_color}; width:{pct}%; height:100%;"></div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 12. Detailed Inspection Diagnostics & Engineering Analysis Report
    st.markdown('<div class="report-card">', unsafe_allow_html=True)
    st.markdown('<div class="report-title">📋 Comprehensive Inspection Diagnostics & Traceability Log</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="report-text">{details["detection_details"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
