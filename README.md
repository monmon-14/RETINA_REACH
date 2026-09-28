# 👁️ RetinaCam - Fundus Eye Image Grad-CAM Visualization

RetinaCam is a deep learning explainability tool designed for retinal fundus imaging. It generates Grad-CAM and Grad-CAM++ activation maps to visualize what anatomical and pathological features (e.g., optic cup, macula/fovea, hard exudates, hemorrhages, and retinal vasculature) a convolutional neural network focuses on when analyzing fundus images.

---

## 🚀 Quick Start

### 1. Launch with One Click
Double-click the **`run_app.bat`** file in this folder. It will launch the Streamlit web dashboard in your browser.

### 2. Or Launch via Terminal
```bash
# Activate the environment
cd fundus_gradcam_app
.\venv\Scripts\activate

# Run the Streamlit web app
streamlit run app.py
```

### 3. Command-Line Interface (CLI)
You can also generate Grad-CAM overlays directly from the terminal without opening the browser:
```bash
.\venv\Scripts\python.exe cli.py --image samples/sample_diabetic_retinopathy.jpg --output output_cam.jpg
```
Additional options:
```bash
# Specify target class (e.g., class 1: Diabetic Retinopathy, class 2: Glaucoma)
.\venv\Scripts\python.exe cli.py --image my_eye.png --target-class 1 --colormap Turbo --alpha 0.65

# Use a specific backbone (ResNet50, EfficientNet-B0, MobileNetV3-Large)
.\venv\Scripts\python.exe cli.py --image my_eye.png --backbone EfficientNet-B0 --method gradcam++
```

---

## ✨ Features

- **Multi-Backbone Support**: Choose between `ResNet50`, `ResNet18`, `EfficientNet-B0`, and `MobileNetV3-Large`.
- **Target Layer Selection**: Dynamically inspect any convolutional layer (e.g., `layer4`, `layer3`, `features[-1]`).
- **Grad-CAM & Grad-CAM++**: Grad-CAM++ utilizes higher-order gradient weighting to isolate small, distributed lesions like microaneurysms and hard exudates.
- **Custom Model Weights**: Easily upload your own fine-tuned PyTorch checkpoint (`.pth` or `.pt`).
- **Interactive Heatmap Controls**:
  - Colormaps: `Jet`, `Turbo`, `Viridis`, `Inferno`, `Hot`, `Plasma`.
  - Alpha blending transparency slider.
  - Activation thresholding slider to eliminate background noise.
  - Optional CLAHE contrast enhancement for fundus vasculature.
- **Side-by-Side Diagnostic View**:
  1. Original Fundus Photograph
  2. Pure Activation Heatmap
  3. Blended Diagnostic Overlay
- **Retinal Feature Insights**: Displays clinical context explaining what landmarks (optic disc, macula, blood vessels, exudates) the model is evaluating.
- **One-Click Export**: Download high-resolution Grad-CAM overlays and standalone heatmaps.
- **Sample Gallery**: Includes pre-loaded synthetic fundus cases (Normal, Diabetic Retinopathy, Glaucoma, AMD) for immediate evaluation.

---

## 🔬 Retinal Disease Diagnostic Classes

| Index | Condition | Typical Grad-CAM Attention Region |
|---|---|---|
| **0** | Normal Retinal Fundus | Symmetrical physiological landmarks: Optic disc margins, healthy foveal avascular zone (FAZ). |
| **1** | Diabetic Retinopathy | Microvascular lesions: hard exudates, microaneurysms, blot hemorrhages along retinal arcades. |
| **2** | Glaucoma | Optic nerve head (ONH), neuroretinal rim thinning, elevated cup-to-disc ratio (CDR > 0.6). |
| **3** | Age-Related Macular Degeneration (AMD) | Central macula & fovea, confluent drusen deposits, geographic atrophy. |
| **4** | Retinal Vasculopathy | Main branch arterioles, arteriovenous nicking, vascular caliber changes. |

---

## 📁 Project Structure

```
fundus_gradcam_app/
├── app.py                  # Interactive Streamlit Web Dashboard
├── cli.py                  # Command-line interface for batch/terminal execution
├── gradcam_engine.py       # Core Grad-CAM and Grad-CAM++ implementation & hooks
├── models.py               # Retinal classification backbones & layer introspection
├── create_sample_fundus.py # Generates demonstration fundus images
├── test_pipeline.py        # Automated test verification suite
├── requirements.txt        # Python package dependencies
├── run_app.bat             # 1-click Windows launcher
├── samples/                # Sample fundus images (Normal, DR, Glaucoma, AMD)
└── venv/                   # Python 3.12 isolated virtual environment
```
