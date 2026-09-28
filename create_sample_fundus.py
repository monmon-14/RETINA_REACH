"""
Generates synthetic anatomical fundus eye images for immediate local demonstration.
Models the fundus aperture, retinal background, optic disc/cup, macula, and vessel arcades.
"""

from pathlib import Path
import numpy as np
import cv2
from PIL import Image


def create_base_fundus(size: int = 512) -> np.ndarray:
    """Generates base retinal fundus background with vascular tree, disc, and macula."""
    h, w = size, size
    center = (w // 2, h // 2)
    radius = int(size * 0.44)

    # 1. Warm orange-red retinal background with radial gradient
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    
    # Red-orange base: [B, G, R]
    retina = np.zeros((h, w, 3), dtype=np.float32)
    retina[:, :, 2] = 200 - 45 * (dist_from_center / radius)  # Red
    retina[:, :, 1] = 95 - 35 * (dist_from_center / radius)   # Green
    retina[:, :, 0] = 30 - 20 * (dist_from_center / radius)   # Blue
    
    # Add subtle choroidal texture noise
    noise = np.random.normal(0, 5, (h, w, 3)).astype(np.float32)
    retina = np.clip(retina + noise, 0, 255)

    # 2. Optic Disc (Nasal side, e.g. x = 0.32 * w)
    disc_center = (int(w * 0.33), int(h * 0.50))
    disc_radius_x, disc_radius_y = int(size * 0.075), int(size * 0.085)
    
    # Draw yellowish-pink optic disc
    cv2.ellipse(retina, disc_center, (disc_radius_x, disc_radius_y), 0, 0, 360, (50, 190, 245), -1)
    
    # 3. Macula / Fovea (Temporal side, e.g. x = 0.62 * w)
    macula_center = (int(w * 0.62), int(h * 0.51))
    macula_radius = int(size * 0.07)
    
    # Slightly darker brownish-red macula
    fovea_mask = np.exp(-((x - macula_center[0])**2 + (y - macula_center[1])**2) / (2 * (macula_radius * 0.8)**2))
    retina[:, :, 2] -= fovea_mask * 40
    retina[:, :, 1] -= fovea_mask * 30
    retina[:, :, 0] -= fovea_mask * 15
    retina = np.clip(retina, 0, 255)

    # 4. Retinal Blood Vessels (Arcades emerging from optic disc)
    vessel_color = (15, 25, 120)  # Dark brownish red
    # Superior arcade
    pts_sup = np.array([
        disc_center,
        (int(w * 0.38), int(h * 0.30)),
        (int(w * 0.55), int(h * 0.22)),
        (int(w * 0.72), int(h * 0.26)),
        (int(w * 0.82), int(h * 0.35))
    ], np.int32)
    cv2.polylines(retina, [pts_sup], False, vessel_color, thickness=4, lineType=cv2.LINE_AA)

    # Inferior arcade
    pts_inf = np.array([
        disc_center,
        (int(w * 0.38), int(h * 0.70)),
        (int(w * 0.55), int(h * 0.78)),
        (int(w * 0.72), int(h * 0.74)),
        (int(w * 0.82), int(h * 0.65))
    ], np.int32)
    cv2.polylines(retina, [pts_inf], False, vessel_color, thickness=4, lineType=cv2.LINE_AA)

    # Nasal and macular fine branches
    cv2.line(retina, disc_center, (int(w * 0.18), int(h * 0.42)), vessel_color, 2, cv2.LINE_AA)
    cv2.line(retina, disc_center, (int(w * 0.16), int(h * 0.58)), vessel_color, 2, cv2.LINE_AA)
    cv2.line(retina, (int(w * 0.45), int(h * 0.28)), (int(w * 0.56), int(h * 0.42)), vessel_color, 2, cv2.LINE_AA)
    cv2.line(retina, (int(w * 0.45), int(h * 0.72)), (int(w * 0.56), int(h * 0.60)), vessel_color, 2, cv2.LINE_AA)

    # Blur vessels slightly for organic look
    retina = cv2.GaussianBlur(retina, (3, 3), 0)

    return retina, center, radius, disc_center, macula_center


def finalize_fundus(retina: np.ndarray, center: tuple, radius: int) -> Image.Image:
    """Applies circular aperture boundary of fundus camera."""
    h, w, _ = retina.shape
    y, x = np.ogrid[:h, :w]
    aperture_mask = ((x - center[0])**2 + (y - center[1])**2) <= radius**2
    
    # Smooth edge
    mask_float = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask_float, center, radius, 1.0, -1, lineType=cv2.LINE_AA)
    mask_float = cv2.GaussianBlur(mask_float, (9, 9), 0)[:, :, None]

    retina_final = retina * mask_float
    retina_final = np.clip(retina_final, 0, 255).astype(np.uint8)
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(retina_final, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def generate_all_samples():
    out_dir = Path("samples")
    out_dir.mkdir(exist_ok=True)

    # 1. Normal Fundus
    retina, center, radius, disc_center, macula_center = create_base_fundus(512)
    # Small physiological cup (CDR ~ 0.3)
    cv2.circle(retina, disc_center, 12, (110, 230, 255), -1)
    img_normal = finalize_fundus(retina, center, radius)
    img_normal.save(out_dir / "sample_normal_fundus.jpg", quality=95)

    # 2. Diabetic Retinopathy Fundus (Exudates + Hemorrhages)
    retina, center, radius, disc_center, macula_center = create_base_fundus(512)
    cv2.circle(retina, disc_center, 12, (110, 230, 255), -1)
    
    # Hard exudates (bright yellowish lipid plaques around macula)
    np.random.seed(42)
    for _ in range(35):
        ox = int(np.random.normal(macula_center[0] - 25, 28))
        oy = int(np.random.normal(macula_center[1] + 15, 25))
        cv2.circle(retina, (ox, oy), np.random.randint(2, 5), (80, 240, 255), -1)

    # Blot hemorrhages (dark red spots)
    for _ in range(25):
        hx = int(np.random.uniform(200, 420))
        hy = int(np.random.uniform(180, 360))
        cv2.circle(retina, (hx, hy), np.random.randint(3, 7), (10, 15, 80), -1)
        
    img_dr = finalize_fundus(retina, center, radius)
    img_dr.save(out_dir / "sample_diabetic_retinopathy.jpg", quality=95)

    # 3. Glaucoma Fundus (Large pale cup, CDR > 0.7)
    retina, center, radius, disc_center, macula_center = create_base_fundus(512)
    # Enlarged pale cup
    cv2.ellipse(retina, disc_center, (28, 33), 0, 0, 360, (140, 245, 255), -1)
    img_glaucoma = finalize_fundus(retina, center, radius)
    img_glaucoma.save(out_dir / "sample_glaucoma_fundus.jpg", quality=95)

    # 4. AMD Fundus (Macular Drusen)
    retina, center, radius, disc_center, macula_center = create_base_fundus(512)
    cv2.circle(retina, disc_center, 12, (110, 230, 255), -1)
    
    # Soft drusen (confluent yellowish deposits right in central fovea/macula)
    for _ in range(50):
        dx = int(np.random.normal(macula_center[0], 20))
        dy = int(np.random.normal(macula_center[1], 18))
        cv2.circle(retina, (dx, dy), np.random.randint(2, 6), (90, 220, 245), -1)

    img_amd = finalize_fundus(retina, center, radius)
    img_amd.save(out_dir / "sample_amd_macular_drusen.jpg", quality=95)

    print(f"Generated 4 clinical demonstration fundus images in '{out_dir}'")


if __name__ == "__main__":
    generate_all_samples()
