# 👁️ RetinaReach — Diabetic Retinopathy Diagnostic AI & Explainability Platform

RetinaReach is a deep learning web application for ophthalmology, specializing in **Diabetic Retinopathy (DR)** detection, lesion localization, and clinical explainability using retinal fundus and OCT imagery.

---

## 🌟 Key Capabilities

1. **Multimodal Retinal Disease Detection**:
   - Deep neural network backbones (ResNet50, ResNet18, EfficientNet-B0, MobileNetV3-Large).
   - Multi-class diagnostic classification: *Normal*, *Diabetic Retinopathy (Mild, Moderate, Severe, PDR)*, *Glaucoma*, *AMD / Drusen*, and *Retinal Vasculopathy*.

2. **Visual Explainability (Grad-CAM & Grad-CAM++)**:
   - Highlights fine retinal lesions (microaneurysms, hard exudates, hemorrhages, neovascularization, and optic disc cupping).
   - Multi-colormap overlay rendering with adjustable transparency ($\alpha$) and activation thresholding.
   - What-If Diagnostic Exploration across all clinical classes.

3. **💬 Ophthalmic Technician Co-Pilot (UiPath RPA)** *(Located in `/chatbot`)*:
   - Dedicated clinical assistant for technicians and mobile screening staff.
   - Answers questions on ETDRS staging, the international 4:2:1 rule, OCT biomarkers (CST, IRF, SRF, DRIL), and camera acquisition troubleshooting (glare, blur, small pupils).
   - One-click hospital referral automation via **UiPath Orchestrator Queues**.

---

## 🏗️ Repository Structure

```
RETINA_REACH/
├── app.py                      # Main Streamlit Web Application
├── models.py                   # Deep learning architectures & model builder
├── gradcam_engine.py           # Grad-CAM and Grad-CAM++ implementation
├── cli.py                      # Command-line interface for batch inference
├── test_pipeline.py            # Vision pipeline automated test suite
├── samples/                    # Sample retinal fundus photographs
│
├── chatbot/                    # 💬 Ophthalmic Technician Chatbot Feature
│   ├── __init__.py             # Python package initializer
│   ├── dr_clinical_knowledge.py# Clinical knowledge base & symptom guide
│   ├── uipath_bridge.py        # UiPath Orchestrator REST client & demo runner
│   ├── chatbot_ui.py           # Streamlit Chatbot component & RPA panel
│   ├── standalone_app.py       # Standalone launcher for just the chatbot
│   ├── test_chatbot.py         # Chatbot & UiPath automated tests
│   └── UIPATH_INTEGRATION_GUIDE.md # UiPath setup & hackathon pitch guide
│
├── requirements.txt            # Project dependencies
├── run_app.bat                 # 1-click Windows application launcher
└── .gitignore                  # Git ignore rules (excludes venv, weights, cache)
```

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/monmon-14/RETINA_REACH.git
cd RETINA_REACH
pip install -r requirements.txt
```

### 2. Run the Main Web Application
```bash
streamlit run app.py
```
Or double-click `run_app.bat` on Windows.

### 3. Run the Chatbot Standalone (Optional)
If you wish to test only the technician chatbot:
```bash
streamlit run chatbot/standalone_app.py
```

---

## 🧪 Verification & Testing

Run the automated test suites:
```bash
# Test the AI Grad-CAM vision pipeline
python test_pipeline.py

# Test the Technician Chatbot & UiPath bridge
python chatbot/test_chatbot.py
```
