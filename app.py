"""DefectVerse: Intelligent Industrial Visual Inspection & Anomaly Detection Dashboard.

Final Year B.Tech Project - Computer Science & Engineering.
A real-time anomaly detection and defect localization platform built with
PatchCore, featuring multi-category support and ML-based Out-of-Distribution (OOD) Guardrails.
"""

import datetime
import io
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from src.models.patchcore import PatchCore
from src.evaluation.metrics import evaluate_dataset

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DefectVerse | Industrial Inspection AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Industrial AI Custom Theme & CSS
# -----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-main: #0B0F19;
    --bg-card: #111827;
    --bg-card-hover: #1F2937;
    --border-color: #1F293D;
    --accent-cyan: #00F0FF;
    --accent-blue: #38BDF8;
    --accent-emerald: #10B981;
    --accent-rose: #F43F5E;
    --text-primary: #F3F4F6;
    --text-secondary: #94A3B8;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: radial-gradient(circle at 15% 15%, #0d1527 0%, #080c14 100%);
}

/* Metric card styling */
.metric-card {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.8) 0%, rgba(30, 41, 59, 0.6) 100%);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(8px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.metric-card:hover {
    border-color: var(--accent-cyan);
    transform: translateY(-2px);
}
.metric-title {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: #FFFFFF;
    font-family: 'JetBrains Mono', monospace;
}
.metric-sub {
    font-size: 0.78rem;
    color: var(--accent-blue);
    margin-top: 4px;
}

/* Status Badges */
.badge-pass {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 9999px;
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid #10B981;
    color: #34D399;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-fail {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 9999px;
    background: rgba(244, 63, 94, 0.15);
    border: 1px solid #F43F5E;
    color: #FB7185;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* Guardrail Alerts */
.guardrail-alert {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.1) 100%);
    border: 2px solid #EF4444;
    border-radius: 12px;
    padding: 18px 22px;
    margin-bottom: 22px;
    box-shadow: 0 0 25px rgba(239, 68, 68, 0.3);
}
.guardrail-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #FCA5A5;
    letter-spacing: 0.02em;
    margin-bottom: 6px;
}
.guardrail-desc {
    color: #FEE2E2;
    font-size: 0.96rem;
    line-height: 1.45;
}
.guardrail-meta {
    margin-top: 10px;
    font-size: 0.85rem;
    color: #FECACA;
    font-family: 'JetBrains Mono', monospace;
}
.guardrail-pass {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 9999px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #6EE7B7;
    font-size: 0.82rem;
    font-weight: 600;
}

/* Hero Section */
.hero-box {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(99, 102, 241, 0.05) 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 16px;
    padding: 24px 30px;
    margin-bottom: 24px;
}
.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #FFFFFF 0%, #38BDF8 60%, #00F0FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
}
.hero-subtitle {
    color: #94A3B8;
    font-size: 1rem;
    margin: 0;
}

/* Sidebar Branding */
.brand-container {
    padding: 10px 0 20px 0;
    border-bottom: 1px solid #1F293D;
    margin-bottom: 15px;
}
.brand-title {
    font-size: 1.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #FFFFFF, #00F0FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.brand-subtitle {
    font-size: 0.75rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 2px;
}

code, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "inspection_history" not in st.session_state:
    st.session_state.inspection_history = []
if "eval_cache" not in st.session_state:
    st.session_state.eval_cache = {}
if "selected_threshold" not in st.session_state:
    st.session_state.selected_threshold = {}

# -----------------------------------------------------------------------------
# Multi-Category Setup & Model Caching
# -----------------------------------------------------------------------------
SUPPORTED_CATEGORIES = ["Bottle", "Capsule"]

with st.sidebar:
    st.markdown(
        """
        <div class="brand-container">
            <div class="brand-title">🔬 DefectVerse</div>
            <div class="brand-subtitle">Industrial Visual Inspection AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🏷️ Product Line Domain")
    selected_category = st.selectbox(
        "Industrial Category",
        SUPPORTED_CATEGORIES,
        index=0,
        help="Select the product line model and dataset domain.",
    )
    cat_slug = selected_category.lower()

    page = st.radio(
        "Navigation",
        [
            "📊 Overview Dashboard",
            "🔍 Live Inspection Studio",
            "📈 Evaluation Lab",
            "🖼️ Dataset Explorer",
            "📋 Inspection History",
            "📖 About & Methodology",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🖥️ Model & Hardware Status")

MODEL_PATH = PROJECT_ROOT / "outputs" / "models" / f"patchcore_{cat_slug}.pt"
DATA_ROOT = PROJECT_ROOT / "data" / cat_slug


@st.cache_resource(show_spinner="Loading PatchCore model checkpoint...")
def load_cached_model(checkpoint_path_str: str):
    """Load model once per category checkpoint and keep in memory."""
    path = Path(checkpoint_path_str)
    if not path.exists():
        return None
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = PatchCore.load(path, device=device)
        # Ensure backward-compatible safe attributes on instance
        if not hasattr(model, "ood_threshold"):
            setattr(model, "ood_threshold", 0.65)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


model = load_cached_model(str(MODEL_PATH))

# Initialize threshold per category
if model is not None:
    if cat_slug not in st.session_state.selected_threshold:
        st.session_state.selected_threshold[cat_slug] = float(getattr(model, "threshold", 0.5))

# Sidebar health indicators
with st.sidebar:
    if model is not None:
        device_name = str(model.device).upper()
        patches_count = model.memory_bank.shape[0] if getattr(model, "memory_bank", None) is not None else 0
        st.success(f"**Domain**: {selected_category} ({getattr(model, 'backbone_name', 'ResNet18')})")
        st.info(f"**Compute**: `{device_name}`")
        base_thresh_display = getattr(model, "threshold", 0.5)
        ood_cutoff_display = getattr(model, "ood_threshold", 0.65)
        st.caption(
            f"Memory Bank: **{patches_count:,} patches**\n\n"
            f"Base Thresh: **{base_thresh_display:.4f}**\n\n"
            f"OOD Cutoff: **{ood_cutoff_display:.4f}**"
        )

    else:
        st.warning(f"⚠️ Model for **{selected_category}** not found.")
        st.caption(f"Expected at: `{MODEL_PATH.name}`")
        if st.button(f"🚀 Train {selected_category} Model Now", use_container_width=True):
            with st.spinner(f"Training PatchCore on {selected_category} dataset..."):
                from src.models.train import train_patchcore
                try:
                    # Ensure mock data exists if real data missing
                    if not (DATA_ROOT / "train" / "good").exists():
                        from src.data.download_dataset import setup_category
                        setup_category(cat_slug, PROJECT_ROOT / "data", mock=True)

                    train_patchcore(
                        data_dir=f"data/{cat_slug}",
                        output_path=f"outputs/models/patchcore_{cat_slug}.pt",
                        backbone="resnet18",
                    )
                    st.success(f"Successfully trained {selected_category} model!")
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as err:
                    st.error(f"Training failed: {err}")

    st.markdown("---")
    st.caption("College Final Year B.Tech Project\nComputer Science & Engineering")


# =============================================================================
# 1. OVERVIEW DASHBOARD
# =============================================================================
if page == "📊 Overview Dashboard":
    st.markdown(
        f"""
        <div class="hero-box">
            <h1 class="hero-title">DefectVerse - {selected_category} Inspection System</h1>
            <p class="hero-subtitle">
                Autonomous Visual Quality Assurance and Anomaly Localization platform using 
                sub-sampled memory-bank feature representations (PatchCore) with active Out-of-Distribution Guardrails.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI Metrics Row
    history_items = [h for h in st.session_state.inspection_history if h.get("category") == selected_category]
    history_count = len(history_items)
    defect_count = sum(1 for item in history_items if item["is_anomaly"])
    defect_rate = (defect_count / history_count * 100) if history_count > 0 else 0.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Active Domain</div>
                <div class="metric-value">{selected_category}</div>
                <div class="metric-sub">Multi-Category Supported</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Inspected Units</div>
                <div class="metric-value">{history_count}</div>
                <div class="metric-sub">{defect_count} defects flagged ({defect_rate:.1f}%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        thresh_val = getattr(model, "threshold", 0.5) if model else 0.5
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Calibrated Cutoff</div>
                <div class="metric-value">{thresh_val:.4f}</div>
                <div class="metric-sub">Normal distribution 99.5% tile</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi4:
        ood_cutoff = getattr(model, "ood_threshold", 0.65) if model else 0.65

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">OOD Guardrail Bound</div>
                <div class="metric-value">{ood_cutoff:.4f}</div>
                <div class="metric-sub">Cosine structural profile bound</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Architectural Overview
    st.markdown("### ⚙️ Enterprise Inspection Architecture")
    flow_col1, flow_col2, flow_col3 = st.columns(3)
    with flow_col1:
        st.markdown(
            """
            **1. ML Structural OOD Guardrail**
            - Global feature similarity checks against the domain centroid profile.
            - Intercepts foreign inputs (faces, screenshots, noise) before downstream scoring.
            """
        )
    with flow_col2:
        st.markdown(
            """
            **2. Coreset Memory Bank**
            - Multi-scale intermediate features from ResNet `layer2` & `layer3`.
            - Greedy minimax $k$-center coreset subsampling preserving edge-case boundary recall.
            """
        )
    with flow_col3:
        st.markdown(
            """
            **3. Pixel Heatmap Localization**
            - Euclidean nearest-neighbor search with cubic upsampling and Gaussian smoothing.
            - Visualizes precise defect segmentation contours for quality assurance operators.
            """
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"💡 **Active Domain:** Operating on **{selected_category}**. Switch to **🔍 Live Inspection Studio** to test unit camera frames.")


# =============================================================================
# 2. LIVE INSPECTION STUDIO
# =============================================================================
elif page == "🔍 Live Inspection Studio":
    st.markdown(f"## 🔍 Live Inspection Studio ({selected_category})")
    st.markdown(
        f"Inspect industrial **{selected_category.lower()}** units in real time. Upload an image from your computer "
        f"or select a benchmark validation sample from the MVTec AD dataset."
    )

    if model is None:
        st.error(f"⚠️ PatchCore model for '{selected_category}' is not loaded. Please train it via the sidebar button.")
        st.stop()

    # Source Selection Toggle
    st.markdown("### 1. Select Inspection Source")
    source_mode = st.radio(
        "Choose image input source:",
        [
            "📁 Upload Image from Computer Folder",
            f"🧪 Select Sample Image from MVTec {selected_category} Dataset",
        ],
        horizontal=True,
        key=f"inspection_source_radio_{cat_slug}",
    )

    selected_image_input = None
    sample_name = None
    sample_metadata = {}

    if "Upload Image" in source_mode:
        st.markdown(f"#### Upload {selected_category} Image from Local Computer")
        uploaded_file = st.file_uploader(
            f"Choose a {selected_category.lower()} inspection image (.png, .jpg, .jpeg):",
            type=["png", "jpg", "jpeg"],
            help="Upload any high-resolution image directly from your local drive.",
        )
        if uploaded_file is not None:
            try:
                selected_image_input = Image.open(uploaded_file).convert("RGB")
                sample_name = uploaded_file.name
                sample_metadata = {
                    "Domain": selected_category,
                    "Source": "Local Computer Upload",
                    "Filename": uploaded_file.name,
                    "Size": f"{uploaded_file.size / 1024:.1f} KB",
                    "Dimensions": f"{selected_image_input.width} x {selected_image_input.height} px",
                }
            except Exception as e:
                st.error(f"Error reading image: {e}")
        else:
            st.info(f"👆 Please browse and upload an image to begin inspection.")

    else:  # Select from MVTec dataset
        st.markdown(f"#### Select Benchmark Sample from MVTec {selected_category} Dataset")
        test_dir = DATA_ROOT / "test"
        if not test_dir.exists():
            st.warning(f"⚠️ Dataset directory not found at `data/{cat_slug}/test`.")
        else:
            cat_col, img_col = st.columns([1, 2])
            categories = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
            with cat_col:
                selected_condition = st.selectbox(
                    "Inspection Condition / Defect Class",
                    categories,
                    index=0,
                    help="Choose between normal (good) or specific defect classes.",
                )
            
            cond_dir = test_dir / selected_condition
            images = sorted(list(cond_dir.glob("*.png")) + list(cond_dir.glob("*.jpg")) + list(cond_dir.glob("*.jpeg")))
            with img_col:
                if images:
                    image_map = {img.name: img for img in images}
                    chosen_img_name = st.selectbox("Select Sample Image", list(image_map.keys()))
                    selected_img_path = image_map[chosen_img_name]
                    selected_image_input = selected_img_path
                    sample_name = f"{selected_condition}/{chosen_img_name}"
                    
                    pil_temp = Image.open(selected_img_path)
                    sample_metadata = {
                        "Domain": selected_category,
                        "Source": f"MVTec AD ({selected_condition})",
                        "Filename": chosen_img_name,
                        "File Path": str(selected_img_path),
                        "Dimensions": f"{pil_temp.width} x {pil_temp.height} px",
                    }
                else:
                    st.warning(f"No image files found in '{selected_condition}'.")

    # Threshold Tuning Card
    curr_cat_thresh = st.session_state.selected_threshold.get(cat_slug, float(getattr(model, "threshold", 0.5)))
    with st.expander("⚙️ Decision Threshold Configuration", expanded=False):
        t_col1, t_col2 = st.columns([3, 1])
        with t_col1:
            current_thresh = st.slider(
                "Anomaly Decision Cutoff",
                min_value=0.01,
                max_value=1.50,
                value=float(curr_cat_thresh),
                step=0.005,
                help=f"Samples with anomaly score >= threshold will be flagged as DEFECTIVE for {selected_category}.",
            )
            st.session_state.selected_threshold[cat_slug] = current_thresh
            model.threshold = current_thresh
        with t_col2:
            st.write("")
            st.write("")
            if st.button("Reset to Calibrated Base", use_container_width=True):
                st.session_state.selected_threshold[cat_slug] = float(getattr(model, "threshold", 0.5))
                st.rerun()


    # Prediction & Guardrail Handling
    if selected_image_input is not None:
        st.markdown("---")
        st.markdown("### 2. Live Inspection Results")

        with st.spinner("⚡ Running PatchCore feature extraction & guardrail verification..."):
            t_start = time.time()
            prediction = model.predict(selected_image_input)
            latency_ms = (time.time() - t_start) * 1000.0

            score = float(prediction["anomaly_score"])
            thresh = float(st.session_state.selected_threshold[cat_slug])
            is_anomaly = bool(score >= thresh)
            is_ood = bool(prediction.get("is_ood", False))
            ood_sim = float(prediction.get("ood_similarity", 1.0))
            ood_cutoff = float(prediction.get("ood_threshold", 0.65))

        # Check OOD Guardrail Interception
        if is_ood:
            st.markdown(
                f"""
                <div class="guardrail-alert">
                    <div class="guardrail-title">🛡️ [GUARDRAIL TRIGGERED]: Out-of-Distribution input. This image does not match the structural profile of the selected industrial category.</div>
                    <div class="guardrail-desc">
                        The visual activation pattern deviates significantly from the <b>{selected_category}</b> reference distribution. 
                        This safeguard intercepts unrelated imagery (e.g., human faces, landscapes, text screenshots, or foreign objects) 
                        to prevent false positives on the production line.
                    </div>
                    <div class="guardrail-meta">
                        Structural Profile Similarity: <b>{ood_sim:.4f}</b> &nbsp;|&nbsp; 
                        Minimum In-Distribution Bound: <b>{ood_cutoff:.4f}</b> &nbsp;|&nbsp;
                        Domain: <b>{selected_category}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            bypass_guardrail = st.checkbox("⚠️ Bypass Guardrail (Force Inspection Breakdown)", value=False)
            if not bypass_guardrail:
                st.info("💡 To inspect anyway, check 'Bypass Guardrail' above, or switch to an in-distribution sample.")
                st.stop()

        # Status & Metrics Row
        status_col, metric1, metric2, metric3 = st.columns([1.5, 1, 1, 1])
        with status_col:
            if is_anomaly:
                st.markdown('<div class="badge-fail">🚨 DEFECTIVE (FAIL)</div>', unsafe_allow_html=True)
                st.caption("Anomaly detected exceeding decision cutoff.")
            else:
                st.markdown('<div class="badge-pass">✅ NORMAL (PASS)</div>', unsafe_allow_html=True)
                st.caption(f"Sample conforms to {selected_category} memory bank.")

        with metric1:
            st.metric(
                label="Anomaly Score",
                value=f"{score:.4f}",
                delta=f"{score - thresh:+.4f} vs cutoff",
                delta_color="inverse" if is_anomaly else "normal",
            )
        with metric2:
            st.metric(label="Decision Cutoff", value=f"{thresh:.4f}")
        with metric3:
            st.metric(label="Inference Latency", value=f"{latency_ms:.1f} ms")

        # Guardrail confirmation badge
        st.markdown(
            f'<div class="guardrail-pass">🛡️ Guardrail Status: In-Distribution Verified (Structural Match: {ood_sim * 100:.1f}%)</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Side-by-side Inspection Visualization
        st.markdown("#### 🔬 Visual Inspection Breakdown")
        col_orig, col_heat, col_over, col_mask = st.columns(4)

        with col_orig:
            st.markdown("##### 1. Original Image")
            st.image(
                prediction["original_image"],
                caption=f"Input: {sample_name}",
                use_container_width=True,
            )

        with col_heat:
            st.markdown("##### 2. Model Heatmap")
            st.image(
                prediction["heatmap_colored"],
                caption="Patch distance (Jet scale)",
                use_container_width=True,
            )

        with col_over:
            st.markdown("##### 3. Heatmap Overlay")
            st.image(
                prediction["heatmap_overlay"],
                caption="Overlay blend (localization)",
                use_container_width=True,
            )

        with col_mask:
            st.markdown("##### 4. Segmented Defect Mask")
            norm_map = prediction["anomaly_map"]
            binary_mask = (norm_map > 0.45).astype(np.uint8) * 255
            defect_pixel_pct = (np.count_nonzero(binary_mask) / binary_mask.size) * 100.0
            st.image(
                binary_mask,
                caption=f"Defect area: {defect_pixel_pct:.2f}%",
                use_container_width=True,
            )

        # Metadata expander
        if sample_metadata:
            with st.expander("📋 Inspection Metadata & Sample Info", expanded=False):
                st.json(sample_metadata)

        # Logging action
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📥 Record Inspection to History Log", use_container_width=True, type="secondary"):
            st.session_state.inspection_history.append(
                {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "category": selected_category,
                    "sample": sample_name,
                    "score": round(score, 4),
                    "threshold": round(thresh, 4),
                    "is_anomaly": is_anomaly,
                    "is_ood": is_ood,
                    "status": "FAIL (Defective)" if is_anomaly else "PASS (Normal)",
                    "latency_ms": round(latency_ms, 1),
                }
            )
            st.success(f"Inspection record for '{sample_name}' ({selected_category}) logged to Session History!")
    else:
        st.info(f"👆 Please upload or select a {selected_category.lower()} image above to run real-time defect analysis.")


# =============================================================================
# 3. EVALUATION LAB
# =============================================================================
elif page == "📈 Evaluation Lab":
    st.markdown(f"## 📈 Evaluation & Benchmark Lab ({selected_category})")
    st.markdown(
        f"Empirical performance assessment of PatchCore on the MVTec AD **{selected_category}** test suite. "
        "Evaluates Area Under ROC Curve (AUROC), classification accuracy, precision, recall, and confusion matrix."
    )

    if model is None:
        st.error(f"Model checkpoint for '{selected_category}' not found. Please train model first.")
        st.stop()

    test_dir = DATA_ROOT / "test"
    if not test_dir.exists():
        st.error(f"Test directory not found at `data/{cat_slug}/test`.")
        st.stop()

    eval_col1, eval_col2 = st.columns([1, 3])
    with eval_col1:
        run_bench = st.button(f"🚀 Run Full {selected_category} Benchmark", type="primary", use_container_width=True)

    eval_thresh_default = getattr(model, "threshold", 0.5)
    cache_key = f"{cat_slug}_{st.session_state.selected_threshold.get(cat_slug, eval_thresh_default)}"
    cached_results = st.session_state.eval_cache.get(cache_key)

    if run_bench or cached_results is not None:
        if run_bench or cached_results is None:
            with st.spinner(f"Executing model inference across {selected_category} test split..."):
                results = evaluate_dataset(
                    model, test_dir, threshold=st.session_state.selected_threshold.get(cat_slug, eval_thresh_default)
                )
                st.session_state.eval_cache[cache_key] = results


        results = st.session_state.eval_cache[cache_key]

        # Summary KPIs
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("ROC-AUC Score", f"{results['roc_auc']:.3f}")
        m2.metric("Accuracy", f"{results['accuracy'] * 100:.1f}%")
        m3.metric("Precision", f"{results['precision'] * 100:.1f}%")
        m4.metric("Recall (Sensitivity)", f"{results['recall'] * 100:.1f}%")
        m5.metric("F1-Score", f"{results['f1_score']:.3f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Plotly ROC & Confusion Matrix
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown(f"### 📉 Receiver Operating Characteristic ({selected_category})")
            fig_roc = go.Figure()
            fig_roc.add_trace(
                go.Scatter(
                    x=results["fpr"],
                    y=results["tpr"],
                    mode="lines",
                    name=f"PatchCore (AUC = {results['roc_auc']:.3f})",
                    line=dict(color="#00F0FF", width=3),
                )
            )
            fig_roc.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Random Chance",
                    line=dict(color="#64748B", dash="dash"),
                )
            )
            fig_roc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17, 24, 39, 0.6)",
                font=dict(color="#F3F4F6"),
                xaxis_title="False Positive Rate (1 - Specificity)",
                yaxis_title="True Positive Rate (Recall)",
                margin=dict(l=40, r=40, t=30, b=40),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        with chart_col2:
            st.markdown(f"### 🔲 Confusion Matrix ({selected_category})")
            cm = results["confusion_matrix"]
            labels = ["Normal (Good)", "Defective"]
            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=labels,
                    y=labels,
                    colorscale="Blues",
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 18, "color": "white"},
                )
            )
            fig_cm.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(17, 24, 39, 0.6)",
                font=dict(color="#F3F4F6"),
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                margin=dict(l=40, r=40, t=30, b=40),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # Defect Sub-Category Breakdown Table
        st.markdown(f"### 📊 Breakdown by {selected_category} Defect Sub-Category")
        cat_data = []
        for cat_name, stats in results["category_stats"].items():
            cat_data.append(
                {
                    "Category": cat_name,
                    "Total Samples": stats["total"],
                    "Correctly Classified": stats["correct"],
                    "Category Accuracy": f"{stats['accuracy'] * 100:.1f}%",
                    "Mean Anomaly Score": round(stats["mean_score"], 4),
                }
            )
        st.dataframe(pd.DataFrame(cat_data), use_container_width=True)


# =============================================================================
# 4. DATASET EXPLORER
# =============================================================================
elif page == "🖼️ Dataset Explorer":
    st.markdown(f"## 🖼️ MVTec {selected_category} Dataset Explorer")
    st.markdown(f"Explore inspection samples across normal training images, defective test variants, and ground-truth masks for **{selected_category}**.")

    if not DATA_ROOT.exists():
        st.error(f"Dataset not found at `data/{cat_slug}`.")
        st.stop()

    tab1, tab2 = st.tabs(["Test Defect Samples & Ground Truth", "Normal Training Samples"])

    with tab1:
        test_dir = DATA_ROOT / "test"
        gt_dir = DATA_ROOT / "ground_truth"

        categories = sorted([d.name for d in test_dir.iterdir() if d.is_dir()])
        filter_cat = st.selectbox("Filter by Category", ["All"] + categories)

        images_to_show = []
        for cat in categories:
            if filter_cat == "All" or filter_cat == cat:
                c_dir = test_dir / cat
                imgs = sorted(list(c_dir.glob("*.png")) + list(c_dir.glob("*.jpg")))
                for img in imgs:
                    gt_img = gt_dir / cat / f"{img.stem}_mask.png"
                    if not gt_img.exists():
                        gt_img = gt_dir / cat / img.name
                    images_to_show.append((img, cat, gt_img if gt_img.exists() else None))

        st.caption(f"Showing {len(images_to_show)} images for {selected_category}")
        grid_cols = st.columns(4)
        for idx, (img_path, cat, gt_path) in enumerate(images_to_show[:20]):
            col = grid_cols[idx % 4]
            with col:
                st.image(str(img_path), caption=f"{cat} - {img_path.name}", use_container_width=True)
                if gt_path is not None:
                    with st.expander("Show Ground Truth Mask"):
                        st.image(str(gt_path), use_container_width=True)

    with tab2:
        train_dir = DATA_ROOT / "train" / "good"
        if train_dir.exists():
            train_imgs = sorted(list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpg")))
            st.caption(f"Total Normal Training Samples: {len(train_imgs)}")
            t_cols = st.columns(4)
            for idx, p in enumerate(train_imgs[:16]):
                t_cols[idx % 4].image(str(p), caption=f"train/good: {p.name}", use_container_width=True)
        else:
            st.info("Train directory not found.")


# =============================================================================
# 5. INSPECTION HISTORY
# =============================================================================
elif page == "📋 Inspection History":
    st.markdown("## 📋 Session Inspection History")
    st.markdown("Operational audit trail of parts inspected in this session with domain category, scores, and decision verdicts.")

    history = st.session_state.inspection_history
    if not history:
        st.info("No units inspected in this session yet. Run inspections from the **Live Inspection Studio** to log records.")
    else:
        df = pd.DataFrame(history)
        
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Inspected", len(df))
        s2.metric("Flagged Defects", int((df["status"].str.contains("FAIL")).sum()))
        s3.metric("OOD Blocked", int(df.get("is_ood", pd.Series([False] * len(df))).sum()))
        defect_pct = (df["status"].str.contains("FAIL")).mean() * 100
        s4.metric("Defect Rate", f"{defect_pct:.1f}%")

        st.markdown("### 📑 Inspection Log Records")
        st.dataframe(df, use_container_width=True)

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="💾 Export Inspection Log as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"defectverse_inspection_log_{datetime.date.today()}.csv",
            mime="text/csv",
        )

        if st.button("🗑️ Clear Session History"):
            st.session_state.inspection_history.clear()
            st.rerun()


# =============================================================================
# 6. ABOUT & METHODOLOGY
# =============================================================================
elif page == "📖 About & Methodology":
    st.markdown("## 📖 About DefectVerse & Research Methodology")

    st.markdown(
        """
        ### 🎯 Project Overview
        **DefectVerse** is an enterprise-grade final-year B.Tech engineering capstone project designed 
        to solve visual defect detection challenges on modern high-speed production lines.

        Unlike standard supervised classifiers which fail in production due to the extreme scarcity of annotated defect examples, 
        DefectVerse implements **unsupervised anomaly detection** with **ML-based Guardrails**.
        """
    )

    st.markdown("---")
    st.markdown(
        """
        ### 🛡️ Out-of-Distribution (OOD) Guardrail
        Industrial inspection models deployed in factory environments frequently encounter unintended camera captures 
        (e.g., human operator hands, calibration cards, foreign objects, or wrong part models on a mixed conveyor).

        Standard anomaly detectors erroneously classify these foreign inputs as "massive defects". 
        DefectVerse introduces a **structural validation gate**:
        - Computes the cosine similarity of the global feature representation against the calibrated category centroid $\\mu_{class}$.
        - If an input diverges from the reference structural distribution, the guardrail triggers:
          `[GUARDRAIL TRIGGERED]: Out-of-Distribution input. This image does not match the structural profile of the selected industrial category.`
        - Prevents downstream alarm corruption and guarantees line reliability.
        """
    )

    st.markdown("---")
    st.markdown(
        """
        ### 🧠 PatchCore Architecture (Roth et al., CVPR 2022)
        1. **Feature Extraction**:
           Pretrained ResNet backbones extract spatial feature maps from intermediate layers (`layer2` and `layer3`).
        2. **Neighborhood Pooling**:
           Local $3 \\times 3$ average pooling provides contextual locality without losing resolution.
        3. **Greedy Minimax Coreset Subsampling**:
           Random projection-accelerated $k$-center reduction retains a compact, highly representative memory bank of nominal product states.
        4. **Multi-Category Scalability**:
           Seamlessly adapts to rigid containers (**Bottle**) and pharmaceutical consumables (**Capsule**) with domain-isolated memory banks.
        """
    )

    st.markdown("---")
    st.markdown(
        """
        ### 📚 Dataset Attribution & License
        - **Dataset**: MVTec Anomaly Detection (MVTec AD) - *Bottle & Capsule Categories*
        - **Authors**: Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger (MVTec Software GmbH)
        - **Publication**: *The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, International Journal of Computer Vision (IJCV), 2021.
        - **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
        """
    )
