"""
Grad-CAM and Grad-CAM++ Engine for Retinal Fundus Image Analysis
Computes visual explanation heatmaps highlighting critical pathological regions
(such as optic cup/disc, macula, hard exudates, hemorrhages, and vascular patterns).
"""

from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


COLORMAP_OPTIONS = {
    "Jet": cv2.COLORMAP_JET,
    "Turbo": cv2.COLORMAP_TURBO,
    "Viridis": cv2.COLORMAP_VIRIDIS,
    "Inferno": cv2.COLORMAP_INFERNO,
    "Hot": cv2.COLORMAP_HOT,
    "Plasma": cv2.COLORMAP_PLASMA,
}


class GradCAM:
    """
    Standard Grad-CAM implementation.
    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float, torch.Tensor]:
        """
        Generate Grad-CAM activation map.
        
        Args:
            input_tensor: Torch tensor of shape [1, C, H, W]
            target_class: Index of target class, or None to use model's top prediction
            
        Returns:
            Tuple of:
              - cam: 2D numpy array [H, W] normalized in [0, 1]
              - pred_class: Index of predicted or targeted class
              - pred_score: Softmax probability of target class
              - raw_logits: Model output logits
        """
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(probs, dim=1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Global average pooling of gradients to get channel weights
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)  # [C, 1, 1]

        # Weighted combination of activation maps
        cam = torch.sum(weights * activations, dim=0)  # [H, W]
        cam = F.relu(cam)  # ReLU to focus on positive contributions

        cam_np = cam.cpu().numpy()
        if np.max(cam_np) > np.min(cam_np):
            cam_np = (cam_np - np.min(cam_np)) / (np.max(cam_np) - np.min(cam_np) + 1e-8)
        else:
            cam_np = np.zeros_like(cam_np)

        pred_score = float(probs[0, target_class].item())
        return cam_np, target_class, pred_score, logits.detach()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []


class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation with higher-order gradient weighting for
    fine-grained localization of multiple lesions or subtle lesions.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self._handles.append(self.target_layer.register_forward_hook(forward_hook))
        self._handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, int, float, torch.Tensor]:
        self.model.eval()
        self.model.zero_grad()

        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(probs, dim=1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Grad-CAM++ weights: alpha_k_ij
        g2 = gradients.pow(2)
        g3 = gradients.pow(3)
        # Sum over spatial dimensions
        sum_activations = activations.sum(dim=(1, 2), keepdim=True)
        eps = 1e-7

        alpha = g2 / (2 * g2 + sum_activations * g3 + eps)
        alpha = torch.where(gradients != 0, alpha, torch.zeros_like(alpha))

        # Weights = sum(alpha * relu(grad))
        weights = torch.sum(alpha * F.relu(gradients), dim=(1, 2), keepdim=True)

        cam = torch.sum(weights * activations, dim=0)
        cam = F.relu(cam)

        cam_np = cam.detach().cpu().numpy()
        if np.max(cam_np) > np.min(cam_np):
            cam_np = (cam_np - np.min(cam_np)) / (np.max(cam_np) - np.min(cam_np) + 1e-8)
        else:
            cam_np = np.zeros_like(cam_np)

        pred_score = float(probs[0, target_class].item())
        return cam_np, target_class, pred_score, logits.detach()

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def preprocess_fundus_image(
    image: Union[Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
    apply_clahe: bool = False
) -> Tuple[torch.Tensor, Image.Image]:
    """
    Standardizes a fundus eye photograph for deep learning.
    Optional CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances retinal microvasculature.
    
    Returns:
        input_tensor: [1, 3, H, W] normalized tensor
        processed_pil: PIL image resized to target size in RGB
    """
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        pil_img = Image.fromarray(image)
    else:
        pil_img = image.convert("RGB")

    resized_pil = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    img_np = np.array(resized_pil)

    if apply_clahe:
        # Apply CLAHE on the green channel which carries the strongest retinal contrast
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        resized_pil = Image.fromarray(img_np)

    # Normalize with ImageNet mean and std
    norm_tensor = torch.from_numpy(img_np).float() / 255.0
    norm_tensor = norm_tensor.permute(2, 0, 1)  # [3, H, W]

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    norm_tensor = (norm_tensor - mean) / std

    return norm_tensor.unsqueeze(0), resized_pil


def generate_gradcam_overlay(
    original_pil: Image.Image,
    cam_mask: np.ndarray,
    colormap_name: str = "Jet",
    alpha: float = 0.5,
    threshold: float = 0.0
) -> Tuple[Image.Image, Image.Image]:
    """
    Generates colorized heatmap and blended overlay.
    
    Args:
        original_pil: Original PIL image
        cam_mask: 2D numpy array [H_cam, W_cam] with values in [0, 1]
        colormap_name: Key from COLORMAP_OPTIONS
        alpha: Heatmap blend strength (0.0 to 1.0)
        threshold: Threshold below which activations are zeroed
        
    Returns:
        overlay_pil: Original image with colored heatmap blended
        heatmap_pil: Standalone colored heatmap
    """
    orig_w, orig_h = original_pil.size
    
    # Resize CAM mask to match image
    cam_resized = cv2.resize(cam_mask, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0.0, 1.0)

    if threshold > 0.0:
        cam_resized = np.where(cam_resized >= threshold, cam_resized, 0.0)
        if np.max(cam_resized) > 0:
            cam_resized = cam_resized / np.max(cam_resized)

    # Convert to uint8 0-255
    heatmap_uint8 = np.uint8(255 * cam_resized)
    cmap = COLORMAP_OPTIONS.get(colormap_name, cv2.COLORMAP_JET)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cmap)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    orig_np = np.array(original_pil.convert("RGB"))
    
    # Alpha blend
    blended = cv2.addWeighted(orig_np, 1.0 - alpha, heatmap_colored_rgb, alpha, 0)
    
    return Image.fromarray(blended), Image.fromarray(heatmap_colored_rgb)
