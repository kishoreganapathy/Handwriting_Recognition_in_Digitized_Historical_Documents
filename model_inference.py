"""
model_inference.py
==================
Implements Box 5: RECOGNITION MODEL
- Loads the trained CNN handwriting recognition model (best_handwriting_model_fixed.keras)
- Preprocesses character crops for inference: (64, 64, 1) float32
- Performs batch prediction
- Computes top-1 and top-5 candidates, softmax confidence scores
- Maps class IDs to Tamil characters via class_mapping.py
"""

import os
import cv2
import numpy as np
from PIL import Image
import keras
from typing import List, Dict, Any, Optional

from class_mapping import (
    KERAS_CLASS_NAMES,
    class_index_to_label,
    get_character_info
)
from segmentation import SegmentedGlyph

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_handwriting_model_fixed.keras")
IMG_SIZE = (64, 64)


class VatteluttuClassifier:
    """Classifier wrapper for the trained Tamil palmleaf Vatteluttu CNN model."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the fixed Keras model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        print(f"Loading trained recognition model from: {self.model_path}...")
        self.model = keras.models.load_model(self.model_path)
        print("Model loaded successfully!")
        print(f"Input shape: {self.model.input_shape}, Output shape: {self.model.output_shape}")

    def preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        """
        Preprocesses a single glyph crop for model input:
        1. Convert to grayscale.
        2. Ensure dark background (0) with bright glyph (255), matching training data.
        3. Resize directly to (64, 64) without corrupting white border padding.
        4. Add channel dimension -> (64, 64, 1).
        Note: Model contains internal Rescaling(1/255), so input is [0, 255].
        """
        if len(crop.shape) == 3:
            if crop.shape[2] == 4:
                # If transparent PNG, check if background is transparent or black
                alpha = crop[:, :, 3]
                if np.all(alpha == 255):
                    gray = cv2.cvtColor(crop[:, :, :3], cv2.COLOR_BGR2GRAY)
                else:
                    # Composite onto black background
                    alpha_f = alpha.astype(np.float32) / 255.0
                    bgr = crop[:, :, :3].astype(np.float32)
                    composite = (bgr * alpha_f[:, :, None]).astype(np.uint8)
                    gray = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # Check background polarity: training data is dark background (0) with bright glyph (255)
        # Sample border pixels to detect whether background is light or dark
        h, w = gray.shape[:2]
        if h > 2 and w > 2:
            top_mean = float(np.mean(gray[0, :]))
            bottom_mean = float(np.mean(gray[-1, :]))
            left_mean = float(np.mean(gray[:, 0]))
            right_mean = float(np.mean(gray[:, -1]))
            border_mean = (top_mean + bottom_mean + left_mean + right_mean) / 4.0
        else:
            border_mean = float(np.mean(gray))

        if border_mean > 128:
            # Light background (e.g. palm leaf parchment with dark ink) -> invert so ink is bright
            gray = cv2.bitwise_not(gray)

        # Resize directly to (64, 64) matching training input size
        resized = cv2.resize(gray, IMG_SIZE, interpolation=cv2.INTER_AREA)

        # Convert to float32 with channel dim (64, 64, 1)
        arr = resized.astype(np.float32)
        arr = np.expand_dims(arr, axis=-1)
        return arr

    def predict_glyph(self, crop: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """Classify a single glyph crop."""
        tensor = self.preprocess_crop(crop)
        batch = np.expand_dims(tensor, axis=0)  # (1, 64, 64, 1)
        preds = self.model.predict(batch, verbose=0)[0]

        top_indices = np.argsort(preds)[::-1][:top_k]
        best_idx = int(top_indices[0])
        best_label = class_index_to_label(best_idx)
        best_conf = float(preds[best_idx] * 100)
        char_info = get_character_info(best_label)

        top_candidates = []
        for idx in top_indices:
            lbl = class_index_to_label(int(idx))
            conf = float(preds[idx] * 100)
            c_info = get_character_info(lbl)
            top_candidates.append({
                "class_label": lbl,
                "confidence": round(conf, 2),
                "tamil": c_info["tamil"],
                "translit": c_info["translit"],
                "category": c_info["category"]
            })

        return {
            "predicted_class": best_label,
            "confidence": round(best_conf, 2),
            "tamil": char_info["tamil"],
            "translit": char_info["translit"],
            "category": char_info["category"],
            "top_candidates": top_candidates
        }

    def predict_batch_glyphs(self, glyphs: List[SegmentedGlyph], top_k: int = 5) -> List[Dict[str, Any]]:
        """Classifies a list of segmented glyphs in a single fast vectorized batch."""
        if not glyphs:
            return []

        tensors = [self.preprocess_crop(g.crop) for g in glyphs]
        batch = np.stack(tensors, axis=0)  # (N, 64, 64, 1)

        predictions = self.model.predict(batch, batch_size=32, verbose=0)
        results = []

        for i, preds in enumerate(predictions):
            top_indices = np.argsort(preds)[::-1][:top_k]
            best_idx = int(top_indices[0])
            best_label = class_index_to_label(best_idx)
            best_conf = float(preds[best_idx] * 100)
            char_info = get_character_info(best_label)

            top_candidates = []
            for idx in top_indices:
                lbl = class_index_to_label(int(idx))
                conf = float(preds[idx] * 100)
                c_info = get_character_info(lbl)
                top_candidates.append({
                    "class_label": lbl,
                    "confidence": round(conf, 2),
                    "tamil": c_info["tamil"],
                    "translit": c_info["translit"],
                    "category": c_info["category"]
                })

            results.append({
                "line_idx": glyphs[i].line_idx,
                "glyph_idx": glyphs[i].glyph_idx,
                "bbox": glyphs[i].to_dict()["bbox"],
                "predicted_class": best_label,
                "confidence": round(best_conf, 2),
                "tamil": char_info["tamil"],
                "translit": char_info["translit"],
                "category": char_info["category"],
                "top_candidates": top_candidates
            })

        return results
