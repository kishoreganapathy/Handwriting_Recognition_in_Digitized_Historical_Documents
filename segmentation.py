"""
segmentation.py
===============
Implements Box 3: TEXT LINE & WORD SEGMENTATION
- Detect text regions
- Line level cropping
- Glyph / character level cropping
- Visualization with color-coded bounding boxes
- Handles both full multi-line documents and single cropped glyphs
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from PIL import Image


class SegmentedGlyph:
    """Represents an individual segmented character/glyph."""
    def __init__(self, crop: np.ndarray, bbox: Tuple[int, int, int, int], line_idx: int, glyph_idx: int):
        self.crop = crop            # Cropped image (numpy array)
        self.bbox = bbox            # (x, y, w, h) in original/preprocessed coordinates
        self.line_idx = line_idx    # 0-indexed line number
        self.glyph_idx = glyph_idx  # 0-indexed character sequence number in line

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_idx": self.line_idx,
            "glyph_idx": self.glyph_idx,
            "bbox": {
                "x": int(self.bbox[0]),
                "y": int(self.bbox[1]),
                "w": int(self.bbox[2]),
                "h": int(self.bbox[3])
            }
        }


class SegmentedLine:
    """Represents a segmented line of text with its child glyphs."""
    def __init__(self, crop: np.ndarray, bbox: Tuple[int, int, int, int], line_idx: int):
        self.crop = crop            # Cropped line image
        self.bbox = bbox            # (x, y, w, h)
        self.line_idx = line_idx
        self.glyphs: List[SegmentedGlyph] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_idx": self.line_idx,
            "bbox": {
                "x": int(self.bbox[0]),
                "y": int(self.bbox[1]),
                "w": int(self.bbox[2]),
                "h": int(self.bbox[3])
            },
            "num_glyphs": len(self.glyphs)
        }


def is_single_character_image(image: np.ndarray) -> bool:
    """
    Check if the input is already a single cropped glyph
    (e.g., approximately square, aspect ratio close to 1:1, or smaller dimensions).
    """
    h, w = image.shape[:2]
    aspect_ratio = max(w / max(h, 1), h / max(w, 1))
    # If the aspect ratio is fairly compact and image is relatively small or square-ish
    if aspect_ratio < 2.0 and (max(h, w) <= 450 or min(h, w) > 200 and aspect_ratio < 1.3):
        return True
    return False


def segment_lines_horizontal_projection(
    binary_inv: np.ndarray,
    min_line_height: int = 15,
    min_line_gap: int = 5
) -> List[Tuple[int, int]]:
    """
    Detect text lines using horizontal projection profile.
    Returns list of (y_start, y_end) tuples.
    """
    h, w = binary_inv.shape[:2]
    # Sum across columns: high value = line containing text
    proj = np.sum(binary_inv, axis=1) / 255.0

    # Dynamic threshold: lines with ink above 1% of width
    thresh_val = max(w * 0.015, 2.0)
    in_line = False
    start_y = 0
    raw_lines = []

    for y in range(h):
        if proj[y] > thresh_val and not in_line:
            in_line = True
            start_y = y
        elif proj[y] <= thresh_val and in_line:
            in_line = False
            end_y = y
            if (end_y - start_y) >= min_line_height:
                raw_lines.append((start_y, end_y))

    if in_line and (h - start_y) >= min_line_height:
        raw_lines.append((start_y, h))

    # Merge lines that are too close together
    merged_lines = []
    for line in raw_lines:
        if not merged_lines:
            merged_lines.append(line)
        else:
            prev_start, prev_end = merged_lines[-1]
            curr_start, curr_end = line
            if curr_start - prev_end < min_line_gap:
                merged_lines[-1] = (prev_start, curr_end)
            else:
                merged_lines.append(line)

    return merged_lines


def segment_glyphs_in_line(
    line_crop: np.ndarray,
    line_binary_inv: np.ndarray,
    line_idx: int,
    offset_y: int,
    offset_x: int = 0,
    min_glyph_area: int = 40,
    min_height: int = 10,
    min_width: int = 8
) -> List[SegmentedGlyph]:
    """
    Segment individual glyphs within a single text line crop using
    connected components & contour bounding boxes, ordered left to right.
    """
    # Morphological closing to keep broken historical strokes together
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(line_binary_inv, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if area >= min_glyph_area and h >= min_height and w >= min_width:
            boxes.append((x, y, w, h))

    # Sort bounding boxes left-to-right (X coordinate)
    boxes.sort(key=lambda b: b[0])

    glyphs: List[SegmentedGlyph] = []
    padding = 3

    for idx, (gx, gy, gw, gh) in enumerate(boxes):
        # Add slight padding around glyph crop
        lh, lw = line_crop.shape[:2]
        pad_y1 = max(0, gy - padding)
        pad_y2 = min(lh, gy + gh + padding)
        pad_x1 = max(0, gx - padding)
        pad_x2 = min(lw, gx + gw + padding)

        crop = line_crop[pad_y1:pad_y2, pad_x1:pad_x2]
        global_bbox = (offset_x + gx, offset_y + gy, gw, gh)

        glyphs.append(SegmentedGlyph(
            crop=crop,
            bbox=global_bbox,
            line_idx=line_idx,
            glyph_idx=idx
        ))

    return glyphs


def segment_document(
    original_image: np.ndarray,
    binary_image: np.ndarray
) -> Tuple[List[SegmentedLine], np.ndarray]:
    """
    Full Box 3 segmentation pipeline:
    1. Detects whether document is multi-line or single-glyph.
    2. Identifies lines and glyphs.
    3. Returns list of SegmentedLines and a visualization image with
       color-coded bounding boxes (red, blue, green dotted/solid boxes).
    """
    h, w = original_image.shape[:2]
    vis_image = original_image.copy()
    if len(vis_image.shape) == 2:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)

    # Invert binary so text is white on black background
    if np.mean(binary_image) > 127:
        binary_inv = cv2.bitwise_not(binary_image)
    else:
        binary_inv = binary_image.copy()

    # Check if single character
    if is_single_character_image(original_image):
        line = SegmentedLine(crop=original_image, bbox=(0, 0, w, h), line_idx=0)
        glyph = SegmentedGlyph(crop=original_image, bbox=(0, 0, w, h), line_idx=0, glyph_idx=0)
        line.glyphs.append(glyph)
        # Draw bounding box
        cv2.rectangle(vis_image, (2, 2), (w - 2, h - 2), (0, 0, 255), 2)
        cv2.putText(vis_image, "Glyph 1", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return [line], vis_image

    # Multi-line document segmentation
    line_spans = segment_lines_horizontal_projection(binary_inv)

    # Fallback if horizontal projection finds no lines
    if len(line_spans) == 0:
        line_spans = [(0, h)]

    lines: List[SegmentedLine] = []
    # Distinct colors for lines (red, blue, green, orange, magenta) matching architecture diagram
    line_colors = [
        (0, 0, 220),    # Red
        (220, 100, 0),  # Blue
        (0, 180, 0),    # Green
        (0, 140, 255),  # Orange
        (180, 0, 180)   # Purple/Magenta
    ]

    for line_idx, (y1, y2) in enumerate(line_spans):
        line_crop = original_image[y1:y2, 0:w]
        line_bin = binary_inv[y1:y2, 0:w]
        line_obj = SegmentedLine(crop=line_crop, bbox=(0, y1, w, y2 - y1), line_idx=line_idx)

        # Segment glyphs within line
        glyphs = segment_glyphs_in_line(
            line_crop=line_crop,
            line_binary_inv=line_bin,
            line_idx=line_idx,
            offset_y=y1,
            offset_x=0
        )
        line_obj.glyphs = glyphs
        lines.append(line_obj)

        # Draw line level box (dashed/dotted simulation)
        color = line_colors[line_idx % len(line_colors)]
        cv2.rectangle(vis_image, (4, y1), (w - 4, y2), color, 2)
        cv2.putText(vis_image, f"Line {line_idx+1}", (10, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw character level boxes
        for g in glyphs:
            gx, gy, gw, gh = g.bbox
            cv2.rectangle(vis_image, (gx, gy), (gx + gw, gy + gh), (255, 255, 0), 1)

    return lines, vis_image
