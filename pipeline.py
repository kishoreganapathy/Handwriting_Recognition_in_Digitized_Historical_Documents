"""
pipeline.py
===========
End-to-End Historical Document Transcription Pipeline
Integrating all components from the architecture:
  1. INPUT: Digitized Historical Document
  2. PREPROCESSING & ENHANCEMENT: Binarization, Denoising, Deskew, Contrast Normalization
  3. TEXT LINE & WORD SEGMENTATION: Region detection, Line-level cropping, Glyph segmentation
  4. DATA AUGMENTATION: Simulation of warping, stains, background blending, photometric jitter
  5. RECOGNITION MODEL: Feature extractor, Transformer / CNN classifier, Confidence scoring
  6. OUTPUT / FINAL PART: Searchable Digital Transcription + Confidence Score via Gemini GenAI
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Union

from preprocessing import (
    full_preprocessing_pipeline,
    full_augmentation_pipeline,
    contrast_normalization,
    denoise_image,
    deskew_image,
    binarize_image
)
from segmentation import segment_document, SegmentedLine, SegmentedGlyph
from model_inference import VatteluttuClassifier
from gemini_postprocessor import GeminiDocumentPostProcessor

try:
    from config import GEMINI_API_KEY, GEMINI_MODEL, MODEL_PATH
except ImportError:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL = "gemini-2.5-flash"
    MODEL_PATH = "best_handwriting_model_fixed.keras"


class HistoricalDocumentPipeline:
    """
    Unified end-to-end pipeline that executes all 6 architectural stages.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None
    ):
        print("Initializing Historical Document Pipeline...")
        m_path = model_path or MODEL_PATH
        self.classifier = VatteluttuClassifier(m_path)
        api_k = gemini_api_key if gemini_api_key is not None else GEMINI_API_KEY
        g_model = gemini_model or GEMINI_MODEL
        self.gemini = GeminiDocumentPostProcessor(api_key=api_k, model_name=g_model)
        print("Pipeline initialized successfully!")

    def set_gemini_api_key(self, api_key: str):
        """Configure or update the Gemini API key."""
        self.gemini.set_api_key(api_key)

    def load_image(self, input_source: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Load image from path, PIL Image, or numpy array into BGR format."""
        if isinstance(input_source, str):
            if not os.path.exists(input_source):
                raise FileNotFoundError(f"Input image file not found: {input_source}")
            img = cv2.imread(input_source)
            if img is None:
                raise ValueError(f"Could not decode image at: {input_source}")
            return img
        elif isinstance(input_source, Image.Image):
            rgb = np.array(input_source.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(input_source, np.ndarray):
            return input_source.copy()
        else:
            raise TypeError("Unsupported image input type")

    def run(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        binarize_method: str = "sauvola",
        denoise_method: str = "bilateral",
        do_deskew: bool = True,
        simulate_augmentation: bool = False,
        context_hints: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the entire 6-stage architecture:
        Stage 1: Input Document
        Stage 2: Preprocessing & Enhancement
        Stage 3: Text Line & Character Segmentation
        Stage 4: Data Augmentation / Simulation (optional)
        Stage 5: Recognition Model Inference
        Stage 6: Gemini GenAI Final Part (Transcription, Error Correction, Translation)
        """
        # 1. INPUT
        original_bgr = self.load_image(image_input)

        # 4. DATA AUGMENTATION (if enabled for robustness testing)
        if simulate_augmentation:
            working_image = full_augmentation_pipeline(original_bgr)
        else:
            working_image = original_bgr.copy()

        # 2. PREPROCESSING & ENHANCEMENT
        prep_results = full_preprocessing_pipeline(
            working_image,
            binarize_method=binarize_method,
            denoise_method=denoise_method,
            do_deskew=do_deskew
        )
        binary_image = prep_results["binary"]

        # 3. TEXT LINE & WORD SEGMENTATION
        lines, visual_bbox_image = segment_document(
            original_image=prep_results["deskewed"],
            binary_image=binary_image
        )

        # Collect all segmented glyphs
        all_glyphs: List[SegmentedGlyph] = []
        for line in lines:
            all_glyphs.extend(line.glyphs)

        # 5. RECOGNITION MODEL
        glyph_predictions = self.classifier.predict_batch_glyphs(all_glyphs)

        # 6. FINAL PART: GEMINI GENAI POST-PROCESSING & SEARCHABLE TRANSCRIPTION
        final_transcription = self.gemini.process_transcription(
            glyph_predictions=glyph_predictions,
            document_image=prep_results["deskewed"],
            context_hints=context_hints
        )

        return {
            # Stage 1: Input
            "original_image": original_bgr,
            # Stage 2: Preprocessing
            "deskewed_image": prep_results["deskewed"],
            "skew_angle": prep_results["skew_angle"],
            "contrast_normalized": prep_results["contrast_normalized"],
            "denoised_image": prep_results["denoised"],
            "binary_image": prep_results["binary"],
            # Stage 3: Segmentation
            "lines": lines,
            "segmented_glyphs": all_glyphs,
            "segmented_visualization": visual_bbox_image,
            # Stage 5: Recognition Model Predictions
            "glyph_predictions": glyph_predictions,
            # Stage 6: Final Gemini GenAI Output
            "final_transcription": final_transcription
        }


def format_searchable_output(results: Dict[str, Any]) -> str:
    """Format the results into a clean, searchable plain text output."""
    final = results["final_transcription"]
    lines = [
        "=" * 70,
        "HISTORICAL DOCUMENT SEARCHABLE TRANSCRIPTION REPORT",
        "=" * 70,
        f"Overall Confidence Score : {final.get('overall_confidence', 0)}%",
        f"Gemini GenAI Engine      : {'ACTIVE' if final.get('gemini_active') else 'PENDING API KEY'}",
        "-" * 70,
        "1. MODERN TAMIL TRANSCRIPTION (நவீன தமிழ்):",
        final.get("modern_tamil", ""),
        "",
        "2. PHONETIC TRANSLITERATION:",
        final.get("transliteration", ""),
        "",
        "3. ENGLISH TRANSLATION:",
        final.get("english_translation", ""),
        "",
        "4. HISTORICAL & EPIGRAPHICAL COMMENTARY:",
        final.get("historical_commentary", ""),
        "-" * 70,
        "5. RAW RECOGNIZED GLYPH SEQUENCE (FROM CNN MODEL):",
        final.get("raw_tamil", ""),
        "=" * 70
    ]
    return "\n".join(lines)
