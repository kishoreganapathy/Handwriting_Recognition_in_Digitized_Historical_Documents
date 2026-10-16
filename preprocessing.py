"""
preprocessing.py
================
Implements:
1. Box 2: PREPROCESSING & ENHANCEMENT
   - Binarization (Sauvola, Otsu, Adaptive Gaussian)
   - Denoising (Bilateral, Non-local Means, Morphological)
   - Deskew & Orientation correction
   - Contrast Normalization (CLAHE)
2. Box 4: DATA AUGMENTATION
   - Geometric warping
   - Ink & stain simulation
   - Background blending
   - Photometric jitter
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Optional


# =====================================================================
# BOX 2: PREPROCESSING & ENHANCEMENT
# =====================================================================

def contrast_normalization(image: np.ndarray, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to enhance local contrast while avoiding noise over-amplification.
    """
    if len(image.shape) == 3:
        # Convert BGR to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l)
        enhanced_lab = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)


def denoise_image(image: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """
    Denoise manuscript images while preserving sharp glyph edges.
    Methods: 'bilateral', 'nlm' (non-local means), 'median', 'gaussian'.
    """
    is_color = len(image.shape) == 3

    if method == "nlm":
        if is_color:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    elif method == "median":
        return cv2.medianBlur(image, 3)
    elif method == "gaussian":
        return cv2.GaussianBlur(image, (3, 3), 0)
    else:
        # Default: Bilateral filtering (excellent edge preservation)
        if is_color:
            return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        else:
            return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


def estimate_skew_angle(image: np.ndarray) -> float:
    """
    Estimate skew angle using the minimum area bounding rectangle of text contours.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Invert binary: text becomes white, background black
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate horizontally to merge letters into text lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours of lines
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    angles = []

    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            rect = cv2.minAreaRect(cnt)
            angle = rect[-1]
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
            # Filter extreme outlier angles
            if abs(angle) < 45:
                angles.append(angle)

    if len(angles) > 0:
        return float(np.median(angles))
    return 0.0


def deskew_image(image: np.ndarray, angle: Optional[float] = None) -> Tuple[np.ndarray, float]:
    """
    Deskew and orient manuscript image using estimated or specified angle.
    """
    if angle is None:
        angle = estimate_skew_angle(image)

    # Only rotate if the angle is significant (> 0.3 degrees)
    if abs(angle) < 0.3:
        return image, 0.0

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, m, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle


def sauvola_binarization(gray: np.ndarray, window_size: int = 25, k: float = 0.2, r: float = 128.0) -> np.ndarray:
    """
    Sauvola adaptive thresholding designed specifically for degraded historical
    documents with uneven illumination, staining, and ink fading.
    Threshold = Mean * (1 + k * (StdDev / r - 1))
    """
    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1

    gray_float = gray.astype(np.float32)
    mean = cv2.blur(gray_float, (window_size, window_size))
    mean_sq = cv2.blur(gray_float ** 2, (window_size, window_size))
    variance = mean_sq - (mean ** 2)
    variance = np.maximum(variance, 0)
    std = np.sqrt(variance)

    threshold = mean * (1.0 + k * ((std / r) - 1.0))
    binarized = (gray_float > threshold).astype(np.uint8) * 255
    return binarized


def binarize_image(image: np.ndarray, method: str = "sauvola") -> np.ndarray:
    """
    Convert image to high-contrast binary (black text on white background or vice versa).
    Methods: 'sauvola', 'otsu', 'adaptive'.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10
        )
    else:
        # Default: Sauvola
        binary = sauvola_binarization(gray, window_size=25, k=0.2)

    return binary


def full_preprocessing_pipeline(
    image: np.ndarray,
    binarize_method: str = "sauvola",
    denoise_method: str = "bilateral",
    do_deskew: bool = True
) -> Dict[str, Any]:
    """
    Executes the entire Step 2 Preprocessing & Enhancement pipeline:
    1. Deskew & Orientation
    2. Contrast Normalization (CLAHE)
    3. Denoising
    4. Binarization
    """
    # 1. Deskew
    if do_deskew:
        deskewed, angle = deskew_image(image)
    else:
        deskewed, angle = image.copy(), 0.0

    # 2. Contrast Normalization
    contrast_norm = contrast_normalization(deskewed)

    # 3. Denoise
    denoised = denoise_image(contrast_norm, method=denoise_method)

    # 4. Binarize
    binary = binarize_image(denoised, method=binarize_method)

    return {
        "original": image,
        "deskewed": deskewed,
        "skew_angle": angle,
        "contrast_normalized": contrast_norm,
        "denoised": denoised,
        "binary": binary
    }


# =====================================================================
# BOX 4: DATA AUGMENTATION / ROBUSTNESS SIMULATION
# =====================================================================

def geometric_warping(image: np.ndarray, intensity: float = 0.05) -> np.ndarray:
    """
    Applies geometric warping / perspective distortion to simulate
    palm leaf curling or non-flat document scanning.
    """
    h, w = image.shape[:2]
    # Source corners
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    # Jittered destination corners
    dx = int(w * intensity)
    dy = int(h * intensity)
    pts2 = np.float32([
        [np.random.randint(-dx, dx + 1), np.random.randint(-dy, dy + 1)],
        [w + np.random.randint(-dx, dx + 1), np.random.randint(-dy, dy + 1)],
        [np.random.randint(-dx, dx + 1), h + np.random.randint(-dy, dy + 1)],
        [w + np.random.randint(-dx, dx + 1), h + np.random.randint(-dy, dy + 1)]
    ])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return warped


def simulate_ink_and_stains(image: np.ndarray, num_stains: int = 4) -> np.ndarray:
    """
    Simulates water stains, oil drops, and ink bleed typical in ancient palm-leaf manuscripts.
    """
    output = image.copy()
    h, w = image.shape[:2]

    for _ in range(num_stains):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        radius = np.random.randint(15, max(20, min(w, h) // 4))

        # Create elliptical stain mask
        overlay = output.copy()
        axis_x = radius
        axis_y = int(radius * np.random.uniform(0.6, 1.4))
        angle = np.random.randint(0, 180)

        # Stain color: brownish / tea-colored or dark ink
        stain_type = np.random.choice(["brown", "ink"])
        if stain_type == "brown":
            color = (
                np.random.randint(40, 80),   # B
                np.random.randint(80, 130),  # G
                np.random.randint(120, 180)  # R
            )
            alpha = np.random.uniform(0.25, 0.45)
        else:
            color = (
                np.random.randint(10, 30),
                np.random.randint(10, 30),
                np.random.randint(10, 30)
            )
            alpha = np.random.uniform(0.15, 0.35)

        cv2.ellipse(overlay, (cx, cy), (axis_x, axis_y), angle, 0, 360, color, -1)
        output = cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0)

    return output


def background_blending(glyph_or_text: np.ndarray, parchment_color: Tuple[int, int, int] = (195, 220, 235)) -> np.ndarray:
    """
    Blends text glyphs onto realistic ancient palm leaf parchment background.
    """
    h, w = glyph_or_text.shape[:2]
    # Create textured parchment background
    noise = np.random.normal(0, 12, (h, w, 3)).astype(np.float32)
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[:, :] = parchment_color  # BGR order: golden palmleaf tint

    # Add subtle horizontal fiber lines typical of dried palm leaves
    for y in range(0, h, 2):
        fiber_intensity = np.random.uniform(-15, 10)
        base[y, :, :] += fiber_intensity

    parchment = np.clip(base + noise, 0, 255).astype(np.uint8)

    if len(glyph_or_text.shape) == 2:
        gray_glyph = glyph_or_text
    else:
        gray_glyph = cv2.cvtColor(glyph_or_text, cv2.COLOR_BGR2GRAY)

    # Invert so ink is dark mask
    _, ink_mask = cv2.threshold(gray_glyph, 128, 255, cv2.THRESH_BINARY_INV)

    # Composite ink onto parchment
    blended = parchment.copy()
    ink_color = np.array([25, 25, 30], dtype=np.uint8)
    for c in range(3):
        blended[:, :, c] = np.where(ink_mask > 0, ink_color[c], parchment[:, :, c])

    return blended


def photometric_jitter(image: np.ndarray, brightness: float = 0.2, contrast: float = 0.2, gamma: float = 0.2) -> np.ndarray:
    """
    Applies photometric jitter: random brightness, contrast, and gamma shifts.
    """
    img_float = image.astype(np.float32)

    # Brightness adjustment
    b_factor = 1.0 + np.random.uniform(-brightness, brightness)
    img_float = img_float * b_factor

    # Contrast adjustment
    c_factor = 1.0 + np.random.uniform(-contrast, contrast)
    mean = np.mean(img_float)
    img_float = (img_float - mean) * c_factor + mean

    # Gamma adjustment
    g_factor = 1.0 + np.random.uniform(-gamma, gamma)
    img_float = np.clip(img_float, 0, 255)
    img_float = 255.0 * ((img_float / 255.0) ** (1.0 / max(g_factor, 0.1)))

    return np.clip(img_float, 0, 255).astype(np.uint8)


def full_augmentation_pipeline(image: np.ndarray) -> np.ndarray:
    """Runs the complete Box 4 data augmentation / degradation simulation."""
    warped = geometric_warping(image, intensity=0.03)
    stained = simulate_ink_and_stains(warped, num_stains=3)
    jittered = photometric_jitter(stained, brightness=0.15, contrast=0.15, gamma=0.15)
    return jittered
