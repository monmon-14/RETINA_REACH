"""
CLI tool to generate Grad-CAM visualizations for retinal fundus images from the command line.

Usage:
    python cli.py --image path/to/fundus.jpg --output result_cam.jpg
    python cli.py --image path/to/fundus.jpg --backbone EfficientNet-B0 --colormap Turbo --alpha 0.6
"""

import argparse
import sys
from pathlib import Path
from PIL import Image
import torch

from gradcam_engine import (
    GradCAM,
    GradCAMPlusPlus,
    preprocess_fundus_image,
    generate_gradcam_overlay,
    COLORMAP_OPTIONS
)
from models import (
    RETINAL_DISEASE_CLASSES,
    build_fundus_model,
    get_available_backbones
)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for Eye Fundus Images")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input fundus image")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path for overlay image")
    parser.add_argument("--output-heatmap", type=str, default=None, help="Output path for standalone heatmap")
    parser.add_argument("--backbone", "-b", type=str, default="ResNet50", choices=get_available_backbones())
    parser.add_argument("--weights", "-w", type=str, default=None, help="Path to custom weights checkpoint (.pth/.pt)")
    parser.add_argument("--method", "-m", type=str, default="gradcam++", choices=["gradcam", "gradcam++"])
    parser.add_argument("--colormap", "-c", type=str, default="Jet", choices=list(COLORMAP_OPTIONS.keys()))
    parser.add_argument("--alpha", "-a", type=float, default=0.55, help="Heatmap blend transparency (0.0 - 1.0)")
    parser.add_argument("--target-class", "-t", type=int, default=None, help="Target class index (0-4), or None for auto")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE contrast enhancement to retinal image")

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    print(f"[+] Loading input fundus photograph: {args.image}")
    input_pil = Image.open(image_path)

    print(f"[+] Initializing model backbone: {args.backbone}")
    model, target_layers_dict = build_fundus_model(
        backbone_name=args.backbone,
        num_classes=len(RETINAL_DISEASE_CLASSES),
        pretrained_backbone=True,
        custom_weights_path=args.weights,
        device="cpu"
    )

    # Use primary recommended layer
    target_layer_name = list(target_layers_dict.keys())[0]
    target_layer = target_layers_dict[target_layer_name]
    print(f"[+] Target convolutional layer: {target_layer_name}")

    input_tensor, _ = preprocess_fundus_image(input_pil, target_size=(224, 224), apply_clahe=args.clahe)

    # Instantiate CAM
    if args.method == "gradcam++":
        cam_generator = GradCAMPlusPlus(model, target_layer)
    else:
        cam_generator = GradCAM(model, target_layer)

    try:
        cam_mask, chosen_class, score, raw_logits = cam_generator.generate(
            input_tensor,
            target_class=args.target_class
        )
    finally:
        cam_generator.remove_hooks()

    class_name = RETINAL_DISEASE_CLASSES[chosen_class]
    print(f"[+] Primary Prediction: '{class_name}' (Confidence: {score * 100:.2f}%)")

    overlay_pil, heatmap_pil = generate_gradcam_overlay(
        input_pil,
        cam_mask,
        colormap_name=args.colormap,
        alpha=args.alpha
    )

    out_overlay_path = args.output or f"cam_overlay_{image_path.stem}.jpg"
    overlay_pil.save(out_overlay_path, quality=95)
    print(f"[OK] Saved Grad-CAM overlay image to: {out_overlay_path}")

    if args.output_heatmap:
        heatmap_pil.save(args.output_heatmap, quality=95)
        print(f"[OK] Saved Heatmap image to: {args.output_heatmap}")


if __name__ == "__main__":
    main()
