"""
Verification test suite for Fundus Grad-CAM system.
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
import torch

from gradcam_engine import (
    GradCAM,
    GradCAMPlusPlus,
    preprocess_fundus_image,
    generate_gradcam_overlay
)
from models import (
    RETINAL_DISEASE_CLASSES,
    build_fundus_model,
    get_available_backbones
)


def test_fundus_pipeline():
    sys.stdout.reconfigure(encoding='utf-8')
    print("--- 1. Testing Model Building ---")
    backbones = get_available_backbones()
    for bb in backbones:
        model, layers = build_fundus_model(backbone_name=bb, num_classes=5, pretrained_backbone=False)
        assert len(layers) > 0, f"No target layers found for {bb}"
        print(f"  [OK] {bb} loaded successfully with {len(layers)} target layer candidates.")

    print("\n--- 2. Testing Preprocessing & Hooks with ResNet50 ---")
    model, layers = build_fundus_model(backbone_name="ResNet50", num_classes=5, pretrained_backbone=False)
    target_layer = list(layers.values())[0]

    # Create dummy retinal fundus image
    dummy_img = Image.fromarray(np.uint8(np.random.rand(400, 400, 3) * 255))
    tensor, proc_pil = preprocess_fundus_image(dummy_img, target_size=(224, 224), apply_clahe=True)
    assert tensor.shape == (1, 3, 224, 224), f"Unexpected tensor shape {tensor.shape}"
    print(f"  [OK] Preprocessed image tensor shape: {tensor.shape}")

    print("\n--- 3. Testing Grad-CAM ---")
    cam = GradCAM(model, target_layer)
    cam_mask, chosen_class, score, logits = cam.generate(tensor, target_class=1)
    cam.remove_hooks()
    assert cam_mask.shape == (224, 224) or len(cam_mask.shape) == 2
    assert 0 <= chosen_class < len(RETINAL_DISEASE_CLASSES)
    print(f"  [OK] Grad-CAM computed mask shape: {cam_mask.shape}, class: {RETINAL_DISEASE_CLASSES[chosen_class]}")

    print("\n--- 4. Testing Grad-CAM++ ---")
    cam_pp = GradCAMPlusPlus(model, target_layer)
    cam_mask_pp, chosen_class_pp, score_pp, _ = cam_pp.generate(tensor, target_class=2)
    cam_pp.remove_hooks()
    assert cam_mask_pp.shape == (224, 224) or len(cam_mask_pp.shape) == 2
    print(f"  [OK] Grad-CAM++ computed mask shape: {cam_mask_pp.shape}, class: {RETINAL_DISEASE_CLASSES[chosen_class_pp]}")

    print("\n--- 5. Testing Overlay & Colormapping ---")
    overlay, heatmap = generate_gradcam_overlay(dummy_img, cam_mask, colormap_name="Jet", alpha=0.5)
    assert overlay.size == dummy_img.size
    assert heatmap.size == dummy_img.size
    print(f"  [OK] Generated overlay and heatmap matching original dimensions {dummy_img.size}")

    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")


if __name__ == "__main__":
    test_fundus_pipeline()
