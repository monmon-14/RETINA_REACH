"""
RetinaCam - Eye Fundus Diagnostic AI & Grad-CAM Visual Explainer
Interactive Streamlit application for deep learning explainability in ophthalmology.
Features a modular Ophthalmic Technician Assistant powered by UiPath RPA.
"""

import os
import io
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

from gradcam_engine import (
    GradCAM,
    GradCAMPlusPlus,
    preprocess_fundus_image,
    generate_gradcam_overlay,
    COLORMAP_OPTIONS
)
from models import (
    RETINAL_DISEASE_CLASSES,
    CLINICAL_INTERPRETATION,
    get_available_backbones,
    build_fundus_model
)

# Import the modular technician chatbot
try:
    from chatbot.chatbot_ui import render_technician_chatbot
    CHATBOT_FEATURE_AVAILABLE = True
except ImportError:
    CHATBOT_FEATURE_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="RetinaCam | Retinal AI Explainer & Clinic Platform",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #2563EB;
        margin-bottom: 15px;
    }
    .info-box {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 14px;
        color: #1E40AF;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cached_model(backbone_name: str, custom_weights_path: str = None):
    return build_fundus_model(
        backbone_name=backbone_name,
        num_classes=len(RETINAL_DISEASE_CLASSES),
        pretrained_backbone=True,
        custom_weights_path=custom_weights_path,
        device="cpu"
    )


def main():
    st.markdown('<div class="main-header">👁️ RetinaCam: Retinal Disease AI Explainer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Deep Learning Explainability for Ophthalmology '
        '(Microaneurysms, Hemorrhages, Exudates, and Vasculature) with an integrated UiPath Technician Co-Pilot.</div>',
        unsafe_allow_html=True
    )

    # Primary tabs: Main Web App vs Technician Assistant Feature
    if CHATBOT_FEATURE_AVAILABLE:
        tab_main_app, tab_chatbot_feature = st.tabs([
            "🔍 Retinal AI & Grad-CAM Analysis",
            "💬 Technician Assistant (UiPath RPA)"
        ])
    else:
        tab_main_app = st.container()
        tab_chatbot_feature = None

    with tab_main_app:
        # Sidebar: Model & Grad-CAM controls
        st.sidebar.header("⚙️ Model & CAM Settings")

        backbones = get_available_backbones()
        selected_backbone = st.sidebar.selectbox("Model Architecture", backbones, index=0)

        # Custom weights option
        custom_model_file = st.sidebar.file_uploader("Upload Custom Model (.pth/.pt)", type=["pth", "pt"])
        custom_weights_path = None
        if custom_model_file is not None:
            temp_weights_dir = Path("temp_weights")
            temp_weights_dir.mkdir(exist_ok=True)
            custom_weights_path = str(temp_weights_dir / custom_model_file.name)
            with open(custom_weights_path, "wb") as f:
                f.write(custom_model_file.getbuffer())
            st.sidebar.success(f"Loaded custom weights: {custom_model_file.name}")

        # Load model & inspect layers
        model, target_layers_dict = load_cached_model(selected_backbone, custom_weights_path)

        layer_names = list(target_layers_dict.keys())
        selected_layer_name = st.sidebar.selectbox("Target Conv Layer", layer_names, index=0)
        target_layer = target_layers_dict[selected_layer_name]

        # Grad-CAM Algorithm
        cam_algorithm = st.sidebar.radio(
            "CAM Method",
            ["Grad-CAM++ (Recommended for fine lesions)", "Grad-CAM (Classic)"],
            index=0
        )

        st.sidebar.subheader("🎨 Heatmap Visualization")
        colormap_name = st.sidebar.selectbox("Colormap", list(COLORMAP_OPTIONS.keys()), index=0)
        alpha = st.sidebar.slider("Overlay Blending (Alpha)", min_value=0.1, max_value=1.0, value=0.55, step=0.05)
        threshold = st.sidebar.slider("Activation Threshold", min_value=0.0, max_value=0.8, value=0.0, step=0.05,
                                      help="Suppresses low-intensity activations to focus exclusively on salient peaks.")
        apply_clahe = st.sidebar.checkbox("Apply CLAHE Preprocessing", value=False,
                                          help="Contrast Limited Adaptive Histogram Equalization enhances microvasculature.")

        # Image source tabs
        tab_upload, tab_samples = st.tabs(["📤 Upload Fundus Image", "🖼️ Sample Fundus Images"])
        
        input_image = None
        image_title = "Fundus Image"

        with tab_upload:
            uploaded_file = st.file_uploader(
                "Select retinal fundus photograph (JPEG, PNG, TIFF)",
                type=["jpg", "jpeg", "png", "tif", "tiff"]
            )
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file)
                image_title = uploaded_file.name

        with tab_samples:
            samples_dir = Path("samples")
            sample_files = list(samples_dir.glob("*.jpg")) + list(samples_dir.glob("*.png"))
            
            if sample_files:
                sample_cols = st.columns(min(len(sample_files), 4))
                for i, sfile in enumerate(sample_files):
                    with sample_cols[i % 4]:
                        simg = Image.open(sfile)
                        st.image(simg, caption=sfile.stem.replace("_", " ").title(), use_container_width=True)
                        if st.button(f"Load {sfile.stem}", key=f"btn_{sfile.stem}"):
                            input_image = simg
                            image_title = sfile.name
            else:
                st.info("No sample images found in `samples/` directory.")

        if input_image is None:
            st.markdown("""
            <div style="text-align: center; padding: 40px; border: 2px dashed #D1D5DB; border-radius: 12px; margin-top: 20px;">
                <p style="font-size: 1.2rem; color: #6B7280;">📸 <b>Upload a fundus image above or click a sample to begin diagnostic explainability</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Process and run inference
            with st.spinner("Analyzing retinal morphology and computing gradient activations..."):
                input_tensor, processed_pil = preprocess_fundus_image(
                    input_image,
                    target_size=(224, 224),
                    apply_clahe=apply_clahe
                )

                # Instantiate CAM
                cam_generator = (
                    GradCAMPlusPlus(model, target_layer)
                    if "PlusPlus" in cam_algorithm
                    else GradCAM(model, target_layer)
                )

                try:
                    # First pass with None to identify predicted class & logits
                    cam_mask_raw, top_class_idx, top_score, raw_logits = cam_generator.generate(
                        input_tensor,
                        target_class=None
                    )
                    probs = torch.softmax(raw_logits, dim=1).numpy()[0]
                finally:
                    cam_generator.remove_hooks()

            st.markdown("---")

            # Diagnostic & Class Selection controls
            pred_class_name = RETINAL_DISEASE_CLASSES[top_class_idx]
            
            col_info, col_bars = st.columns([1.2, 1.8])
            with col_info:
                st.markdown(f"""
                <div class="metric-card">
                    <h4 style="margin: 0; color: #1F2937;">Primary AI Prediction</h4>
                    <h2 style="margin: 6px 0; color: #1E40AF;">{pred_class_name}</h2>
                    <p style="margin: 0; font-size: 1.1rem; color: #4B5563;">Confidence: <b>{top_score * 100:.1f}%</b></p>
                </div>
                """, unsafe_allow_html=True)

                target_class_choice = st.selectbox(
                    "Explain Specific Diagnostic Class (What-If Analysis):",
                    RETINAL_DISEASE_CLASSES,
                    index=top_class_idx,
                    help="Switch to any class to see what retinal regions the network considers evidence for that specific condition."
                )
                chosen_class_idx = RETINAL_DISEASE_CLASSES.index(target_class_choice)

            with col_bars:
                st.write("📊 **Class Probability Distribution**")
                fig, ax = plt.subplots(figsize=(6, 2.3))
                bar_colors = ["#2563EB" if i == top_class_idx else "#93C5FD" for i in range(len(RETINAL_DISEASE_CLASSES))]
                short_names = [
                    "Normal",
                    "Diabetic Retinopathy",
                    "Glaucoma",
                    "AMD / Drusen",
                    "Vasculopathy"
                ]
                ax.barh(short_names[::-1], probs[::-1] * 100, color=bar_colors[::-1])
                ax.set_xlim(0, 100)
                ax.set_xlabel("Probability (%)")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            # Re-compute CAM if user chose a different class than top
            if chosen_class_idx != top_class_idx:
                cam_generator = (
                    GradCAMPlusPlus(model, target_layer)
                    if "PlusPlus" in cam_algorithm
                    else GradCAM(model, target_layer)
                )
                try:
                    cam_mask_raw, _, _, _ = cam_generator.generate(
                        input_tensor,
                        target_class=chosen_class_idx
                    )
                finally:
                    cam_generator.remove_hooks()

            # Generate visual overlay
            overlay_img, heatmap_img = generate_gradcam_overlay(
                input_image,
                cam_mask_raw,
                colormap_name=colormap_name,
                alpha=alpha,
                threshold=threshold
            )

            # 3-Column Visualization Display
            st.subheader("🔍 Retinal Visual Explanation")
            v_col1, v_col2, v_col3 = st.columns(3)

            with v_col1:
                st.markdown("**1. Original Fundus Photograph**")
                st.image(input_image, caption=f"Source: {image_title}", use_container_width=True)

            with v_col2:
                st.markdown(f"**2. {cam_algorithm.split(' ')[0]} Heatmap**")
                st.image(heatmap_img, caption=f"Activation Intensity ({colormap_name})", use_container_width=True)

            with v_col3:
                st.markdown("**3. Diagnostic Overlay Blend**")
                st.image(overlay_img, caption=f"Overlay (\u03b1 = {alpha})", use_container_width=True)

            # Clinical interpretation box
            st.markdown(f"""
            <div class="info-box">
                <b>💡 Clinical Attention Insight for <i>{target_class_choice}</i>:</b><br/>
                {CLINICAL_INTERPRETATION.get(target_class_choice, "Grad-CAM visualizes prominent visual features corresponding to this retinal condition.")}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # Download section
            st.subheader("💾 Export Visual Explanations")
            dl_col1, dl_col2, _ = st.columns([1, 1, 2])

            buf_overlay = io.BytesIO()
            overlay_img.save(buf_overlay, format="JPEG", quality=95)
            with dl_col1:
                st.download_button(
                    label="⬇️ Download Grad-CAM Overlay (.jpg)",
                    data=buf_overlay.getvalue(),
                    file_name=f"gradcam_overlay_{Path(image_title).stem}.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )

            buf_heatmap = io.BytesIO()
            heatmap_img.save(buf_heatmap, format="JPEG", quality=95)
            with dl_col2:
                st.download_button(
                    label="⬇️ Download Pure Heatmap (.jpg)",
                    data=buf_heatmap.getvalue(),
                    file_name=f"heatmap_{Path(image_title).stem}.jpg",
                    mime="image/jpeg",
                    use_container_width=True
                )

    # Render Technician Chatbot as a clean secondary tab
    if CHATBOT_FEATURE_AVAILABLE and tab_chatbot_feature is not None:
        with tab_chatbot_feature:
            render_technician_chatbot()


if __name__ == "__main__":
    main()
