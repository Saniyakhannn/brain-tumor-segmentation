import streamlit as st
import cv2
import numpy as np
from PIL import Image
from predict import predict_image, unet_model, attn_model, device
from gradcam import run_gradcam, apply_heatmap
import torch

# ── Page Config ─────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Segmentation",
    layout="wide"
)

# ── Style ────────────────────────────────────────────
st.markdown("""
<style>
.main {padding-top: 2rem;}
.block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

# ── Title ────────────────────────────────────────────
st.title("🧠 Brain Tumor Segmentation")
st.caption("UNet vs Attention UNet — MRI Brain Tumor Detection")
st.markdown("---")

# ── Sidebar ──────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

unet_threshold = st.sidebar.slider(
    "UNet Threshold", 0.01, 0.5, 0.15, 0.01)
attn_threshold = st.sidebar.slider(
    "Attention UNet Threshold", 0.01, 0.5, 0.15, 0.01)

clean_mask   = st.sidebar.checkbox(
    "🧹 Remove Noise", True)
show_gradcam = st.sidebar.checkbox(
    "🔥 Show Grad-CAM", False)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Results")
st.sidebar.markdown("""
| Metric | UNet | Attention |
|--------|------|-----------|
| IoU | 0.7469 | 0.7354 |
| Dice | 0.8306 | 0.8235 |
| Precision | 0.8580 | 0.8087 |
| Recall | 0.8601 | 0.8915 |
""")

# ── Helper ───────────────────────────────────────────
def post_process(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Remove small noise regions
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    clean_mask = np.zeros_like(mask)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # keep only large regions
            cv2.drawContours(
                clean_mask, [contour], -1, 1, -1)
    return clean_mask


def create_overlay(img_bgr, mask, color=(0, 255, 0)):
    colored          = np.zeros_like(img_bgr)
    colored[:, :, 1] = mask * 255
    overlay          = cv2.addWeighted(
                       img_bgr, 0.7, colored, 0.3, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


# ── Upload ───────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📤 Upload Brain MRI Image",
    type=["jpg", "jpeg", "png", "tif", "tiff"]
)

# ── Main ─────────────────────────────────────────────
if uploaded_file is not None:
    try:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image     = np.array(image_pil)
        image_cv  = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Predictions
        with st.spinner("🔍 Running UNet..."):
            unet_prob = predict_image(
                image_cv, model_type="unet")

        with st.spinner("🔍 Running Attention UNet..."):
            attn_prob = predict_image(
                image_cv, model_type="attention")

        # Threshold
        unet_mask = (unet_prob >= unet_threshold).astype(
                    np.uint8)
        attn_mask = (attn_prob >= attn_threshold).astype(
                    np.uint8)

        if clean_mask:
            unet_mask = post_process(unet_mask)
            attn_mask = post_process(attn_mask)

        st.success("✅ Both models completed successfully!")

        # ── Metrics ──────────────────────────────────
        unet_tumor = np.mean(unet_mask) * 100
        attn_tumor = np.mean(attn_mask) * 100

        st.markdown("### 📊 Tumor Coverage")
        col1, col2 = st.columns(2)
        col1.metric("UNet",          f"{unet_tumor:.2f}%")
        col2.metric("Attention UNet", f"{attn_tumor:.2f}%")

        # ── Model Outputs ─────────────────────────────
        st.markdown("### 🧠 Segmentation Masks")
        col1, col2, col3 = st.columns(3)
        col1.image(image,
                   caption="MRI Input",
                   use_container_width=True)
        col2.image(unet_mask * 255,
                   caption="UNet Mask",
                   use_container_width=True)
        col3.image(attn_mask * 255,
                   caption="Attention Mask",
                   use_container_width=True)

        # ── Overlays ──────────────────────────────────
        unet_overlay = create_overlay(image_cv, unet_mask)
        attn_overlay = create_overlay(image_cv, attn_mask)

        st.markdown("### 🛰️ Tumor Overlay")
        col1, col2 = st.columns(2)
        col1.image(unet_overlay,
                   caption="UNet Overlay",
                   use_container_width=True)
        col2.image(attn_overlay,
                   caption="Attention Overlay",
                   use_container_width=True)

        # ── Grad-CAM ──────────────────────────────────
        if show_gradcam:
            st.markdown("---")
            st.markdown("### 🔥 Grad-CAM Explainability")
            st.caption(
                "Heatmap shows where model focused to detect tumor")

            with st.spinner("🔥 Computing Grad-CAM..."):
                gradcam_results = run_gradcam(image_cv)

            col1, col2, col3 = st.columns(3)
            col1.image(
                gradcam_results["unet_overlay"],
                caption="🔵 UNet Grad-CAM",
                use_container_width=True)
            col2.image(
                gradcam_results["attn_overlay"],
                caption="🔴 Attention Grad-CAM",
                use_container_width=True)
            col3.image(
                gradcam_results["diff"],
                caption="⚖️ Difference Map",
                use_container_width=True)

            st.info(
                "🔴 Red/Yellow = High activation | "
                "🔵 Blue = Low activation | "
                "Difference shows where models focus differently"
            )

        # ── Download ──────────────────────────────────
        st.markdown("---")
        _, buffer = cv2.imencode(
            ".png",
            cv2.cvtColor(attn_overlay, cv2.COLOR_RGB2BGR))
        st.download_button(
            "📥 Download Result",
            buffer.tobytes(),
            "tumor_result.png",
            "image/png"
        )

        st.markdown("---")
        st.caption(
            "Built with Streamlit • PyTorch • Grad-CAM")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.exception(e)