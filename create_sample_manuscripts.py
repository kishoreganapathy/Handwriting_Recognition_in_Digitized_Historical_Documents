"""
create_sample_manuscripts.py
============================
Generates realistic multi-line ancient Tamil palm-leaf manuscript test samples
using actual character glyphs from the Vatteluttu dataset on textured parchment backgrounds.
Saves generated samples into test_images/ for instant pipeline demonstration and testing.
"""

import os
import cv2
import numpy as np
from PIL import Image

DATASET_ROOT = os.path.join(
    os.path.dirname(__file__),
    "Tamil palmleaf Vatteluttu Datastet",
    "palmleaf-tamil",
    "FINAL DATASET",
    "DATASET"
)
TEST_DIR = os.path.join(os.path.dirname(__file__), "test_images")


def create_palm_leaf_background(width: int = 900, height: int = 240) -> np.ndarray:
    """Create realistic ancient dried palm leaf texture."""
    # Base warm golden-brown palm leaf color (BGR)
    base_color = np.array([140, 185, 215], dtype=np.float32)
    bg = np.zeros((height, width, 3), dtype=np.float32)
    bg[:, :] = base_color

    # Add subtle horizontal fiber grain
    for y in range(height):
        fiber = np.sin(y * 0.4) * 8 + np.random.normal(0, 4)
        bg[y, :, :] += fiber

    # Add subtle vertical age gradient
    for x in range(width):
        grad = np.sin(x * 0.01) * 6
        bg[:, x, :] += grad

    # Gaussian noise for organic parchment feel
    noise = np.random.normal(0, 8, (height, width, 3))
    bg = np.clip(bg + noise, 0, 255).astype(np.uint8)

    # Darker burnt / worn edges
    border = 12
    cv2.rectangle(bg, (0, 0), (width, height), (90, 130, 160), border)
    bg = cv2.GaussianBlur(bg, (5, 5), 0)

    return bg


def get_sample_glyph(class_num: int) -> np.ndarray:
    """Load a representative glyph image from the dataset."""
    folder = os.path.join(DATASET_ROOT, str(class_num))
    if not os.path.exists(folder):
        return None
    # Pick a sample image (ignoring non-images like desktop.ini)
    valid_files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    if not valid_files:
        return None
    img_path = os.path.join(folder, valid_files[10 % len(valid_files)])
    glyph = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    return glyph


def generate_manuscript(lines_classes: list, output_filename: str):
    """
    Assembles lines of glyphs onto a palm-leaf strip.
    lines_classes: List of lists containing class IDs for each line.
    """
    os.makedirs(TEST_DIR, exist_ok=True)
    num_lines = len(lines_classes)
    line_h = 55
    top_margin = 35
    total_h = top_margin * 2 + num_lines * line_h + 20
    total_w = 920

    canvas = create_palm_leaf_background(width=total_w, height=total_h)

    for l_idx, line in enumerate(lines_classes):
        current_y = top_margin + l_idx * line_h
        current_x = 40

        for class_id in line:
            glyph_img = get_sample_glyph(class_id)
            if glyph_img is None:
                current_x += 35
                continue

            # Resize glyph to line height
            gh, gw = glyph_img.shape[:2]
            target_h = 42
            target_w = int(gw * (target_h / gh))
            target_w = min(max(target_w, 28), 50)
            resized_glyph = cv2.resize(glyph_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

            # Extract alpha or threshold to get ink mask
            if resized_glyph.shape[-1] == 4:
                alpha = resized_glyph[:, :, 3] / 255.0
                gray_glyph = cv2.cvtColor(resized_glyph[:, :, :3], cv2.COLOR_BGR2GRAY)
            else:
                gray_glyph = cv2.cvtColor(resized_glyph, cv2.COLOR_BGR2GRAY)
                alpha = np.ones((target_h, target_w), dtype=np.float32)

            # Invert so ink is dark
            _, ink_mask = cv2.threshold(gray_glyph, 120, 255, cv2.THRESH_BINARY_INV)

            # Place onto canvas
            if current_x + target_w < total_w - 40:
                roi = canvas[current_y:current_y + target_h, current_x:current_x + target_w]
                # Inscribe ink with organic ancient ink color (dark charcoal / black soot)
                ink_color = np.array([28, 30, 36], dtype=np.uint8)
                for c in range(3):
                    roi[:, :, c] = np.where(ink_mask > 50, ink_color[c], roi[:, :, c])
                canvas[current_y:current_y + target_h, current_x:current_x + target_w] = roi

            current_x += target_w + np.random.randint(6, 12)

    # Save output
    out_path = os.path.join(TEST_DIR, output_filename)
    cv2.imwrite(out_path, canvas)
    print(f"Sample palm leaf manuscript created: {out_path} ({total_w}x{total_h})")


if __name__ == "__main__":
    if os.path.exists(DATASET_ROOT):
        # Sample 1: 3-line historical palm leaf strip
        sample_1_lines = [
            [1, 14, 32, 20, 44, 23, 56, 3, 22, 28, 31],
            [33, 21, 46, 25, 62, 57, 18, 48, 26, 70],
            [2, 36, 16, 45, 23, 60, 24, 50, 19, 30]
        ]
        generate_manuscript(sample_1_lines, "palmleaf_sample1.png")

        # Sample 2: 2-line royal grant / poetry inscription
        sample_2_lines = [
            [5, 20, 32, 25, 40, 23, 58, 28, 31, 44, 62],
            [10, 14, 34, 21, 47, 22, 63, 27, 30, 52]
        ]
        generate_manuscript(sample_2_lines, "palmleaf_sample2.png")
    else:
        print(f"Dataset root not found at {DATASET_ROOT}")
