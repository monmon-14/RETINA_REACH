"""
Model Architecture & Diagnostic Classes for Fundus Eye Image Analysis.
Supports ResNet, EfficientNet, MobileNetV3 backbones, custom checkpoints, and target layer inspection.
"""

from typing import Dict, List, Optional, Tuple
import os
import torch
import torch.nn as nn
from torchvision import models


RETINAL_DISEASE_CLASSES = [
    "Normal Retinal Fundus",
    "Diabetic Retinopathy (Exudates / Hemorrhages)",
    "Glaucoma (Optic Cup Enlargement / CDR)",
    "Age-Related Macular Degeneration (AMD / Drusen)",
    "Retinal Vasculopathy / Hypertensive Changes"
]

CLINICAL_INTERPRETATION: Dict[str, str] = {
    "Normal Retinal Fundus": (
        "Grad-CAM typically focuses symmetrically on physiological retinal landmarks: "
        "the optic nerve head margin, healthy foveal avascular zone (FAZ), and uniform vascular arcades."
    ),
    "Diabetic Retinopathy (Exudates / Hemorrhages)": (
        "Grad-CAM highlights micro-vascular pathology: dot/blot hemorrhages, "
        "yellow lipid hard exudates, cotton wool spots, or neovascularization around the vascular arcades."
    ),
    "Glaucoma (Optic Cup Enlargement / CDR)": (
        "Grad-CAM localizes sharply onto the Optic Disc region, inspecting neuroretinal rim thinning, "
        "increased vertical cup-to-disc ratio (CDR > 0.6), and peripapillary chorioretinal atrophy."
    ),
    "Age-Related Macular Degeneration (AMD / Drusen)": (
        "Grad-CAM concentrates on the Central Macula (fovea), detecting discrete or confluent drusen "
        "deposits, retinal pigment epithelium (RPE) alterations, or geographic atrophy."
    ),
    "Retinal Vasculopathy / Hypertensive Changes": (
        "Grad-CAM detects vascular caliber changes: arteriolar narrowing, arteriovenous (AV) nicking, "
        "copper/silver wiring, or focal vascular tortuosity."
    )
}


def get_available_backbones() -> List[str]:
    return ["ResNet50", "ResNet18", "EfficientNet-B0", "MobileNetV3-Large"]


def get_target_layers_for_model(model: nn.Module, backbone_name: str) -> Dict[str, nn.Module]:
    """
    Returns a dictionary of selectable convolutional layers suitable for Grad-CAM.
    """
    layers: Dict[str, nn.Module] = {}
    
    if backbone_name in ["ResNet50", "ResNet18"]:
        if hasattr(model, "layer4"):
            layers["layer4 (Deepest / Recommended)"] = model.layer4[-1]
            layers["layer4 (Block group)"] = model.layer4
        if hasattr(model, "layer3"):
            layers["layer3 (Mid-level Features)"] = model.layer3[-1]
        if hasattr(model, "layer2"):
            layers["layer2 (Early Structural)"] = model.layer2[-1]
            
    elif "EfficientNet" in backbone_name:
        if hasattr(model, "features"):
            layers["features[-1] (Final Conv / Recommended)"] = model.features[-1]
            layers["features[-2] (High-level Block)"] = model.features[-2]
            layers["features[-3] (Mid-level Block)"] = model.features[-3]

    elif "MobileNet" in backbone_name:
        if hasattr(model, "features"):
            layers["features[-1] (Final Conv / Recommended)"] = model.features[-1]
            layers["features[-2] (Penultimate Block)"] = model.features[-2]

    # Fallback to the last convolutional child module found
    if not layers:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                layers[f"{name} (Auto-detected)"] = module
                break

    return layers


def build_fundus_model(
    backbone_name: str = "ResNet50",
    num_classes: int = 5,
    pretrained_backbone: bool = True,
    custom_weights_path: Optional[str] = None,
    device: str = "cpu"
) -> Tuple[nn.Module, Dict[str, nn.Module]]:
    """
    Constructs and initializes the neural network classifier for retinal imaging.
    """
    if backbone_name == "ResNet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        
    elif backbone_name == "ResNet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )

    elif backbone_name == "EfficientNet-B0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained_backbone else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif backbone_name == "MobileNetV3-Large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    # Load custom weights if supplied
    if custom_weights_path and os.path.exists(custom_weights_path):
        checkpoint = torch.load(custom_weights_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    target_layers = get_target_layers_for_model(model, backbone_name)
    return model, target_layers
