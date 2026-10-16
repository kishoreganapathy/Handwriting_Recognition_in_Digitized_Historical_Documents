"""
gemini_postprocessor.py
=======================
FINAL PART: GEMINI GENAI POST-PROCESSING, CONTEXTUAL ERROR CORRECTION,
MODERN TAMIL TRANSCRIPTION, AND ENGLISH TRANSLATION.

Features:
- Powered by Google GenAI (google.genai SDK)
- Contextual Post-OCR Error Correction for archaic Vatteluttu glyphs
- Conversion to searchable Modern Tamil Unicode text (நவீன தமிழ்)
- Phonetic Romanized transliteration (ISO 15919)
- English translation & historical/epigraphical commentary
- Multimodal document inspection (Vision fallback)
- Seamless offline/pending mode when API key has not yet been provided by the user
"""

import os
import json
import base64
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional

try:
    from config import GEMINI_API_KEY, GEMINI_MODEL
except ImportError:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL = "gemini-2.5-flash"


class GeminiDocumentPostProcessor:
    """
    Intelligent historical manuscript transcription and translation engine
    powered by Google Gemini GenAI.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        # Resolve API Key from parameter, config.py, or environment
        if api_key is not None:
            self.api_key = api_key.strip()
        else:
            self.api_key = (GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")).strip()
        self.model_name = model_name or GEMINI_MODEL or "gemini-2.5-flash"
        self.client = None
        self._init_client()

    def set_api_key(self, api_key: str):
        """Update or set the Gemini API key at runtime."""
        self.api_key = api_key.strip()
        self._init_client()

    def _init_client(self):
        """Initialize Google GenAI client if API key is available."""
        if not self.api_key:
            self.client = None
            return

        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
            print("Google GenAI client initialized successfully.")
        except ImportError:
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=self.api_key)
                self.client = "legacy"
                print("Google GenerativeAI (legacy) configured successfully.")
            except Exception as e:
                print(f"Error initializing Gemini SDK: {e}")
                self.client = None
        except Exception as e:
            print(f"Error creating GenAI client: {e}")
            self.client = None

    @property
    def is_configured(self) -> bool:
        """Check if Gemini API key is configured and active."""
        return bool(self.api_key and self.client is not None)

    def process_transcription(
        self,
        glyph_predictions: List[Dict[str, Any]],
        document_image: Optional[np.ndarray] = None,
        context_hints: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Processes model predictions and produces the final searchable transcription.

        If API Key is provided:
            Executes Gemini GenAI contextual error correction, modern Tamil
            conversion, and English translation.
        If API Key is pending:
            Gracefully generates baseline model transcription with clear notice
            and instructions for entering the API key.
        """
        # 1. Assemble raw line-by-line prediction strings
        lines_dict: Dict[int, List[Dict[str, Any]]] = {}
        for pred in glyph_predictions:
            l_idx = pred.get("line_idx", 0)
            if l_idx not in lines_dict:
                lines_dict[l_idx] = []
            lines_dict[l_idx].append(pred)

        raw_lines_tamil = []
        raw_lines_translit = []
        raw_lines_classes = []
        confidences = []

        for l_idx in sorted(lines_dict.keys()):
            glyphs_in_line = lines_dict[l_idx]
            line_tamil = "".join(g.get("tamil", "") for g in glyphs_in_line)
            line_translit = " ".join(g.get("translit", "") for g in glyphs_in_line)
            line_classes = " ".join(f"[{g.get('predicted_class')}]" for g in glyphs_in_line)

            raw_lines_tamil.append(line_tamil)
            raw_lines_translit.append(line_translit)
            raw_lines_classes.append(line_classes)

            for g in glyphs_in_line:
                confidences.append(g.get("confidence", 0.0))

        raw_tamil_text = "\n".join(raw_lines_tamil)
        raw_translit_text = "\n".join(raw_lines_translit)
        raw_classes_text = "\n".join(raw_lines_classes)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        # Baseline fallback result (used if Gemini key is pending or call fails)
        baseline_result: Dict[str, Any] = {
            "status": "pending_api_key" if not self.is_configured else "completed",
            "gemini_active": self.is_configured,
            "raw_tamil": raw_tamil_text,
            "raw_transliteration": raw_translit_text,
            "raw_class_sequence": raw_classes_text,
            "modern_tamil": raw_tamil_text,
            "english_translation": "(Provide your Gemini API key to generate English translation and historical commentary.)",
            "historical_commentary": "Ancient Tamil Palm Leaf manuscript written in Vatteluttu script. Awaiting Gemini GenAI key for historical context and orthographic reconstruction.",
            "corrected_text": raw_tamil_text,
            "corrections_made": [],
            "overall_confidence": round(avg_confidence, 2),
            "searchable_text": raw_tamil_text,
            "message": "Gemini GenAI ready — awaiting API Key. Provide your GEMINI_API_KEY to enable AI contextual correction, modern Tamil transcription, and English translation."
        }

        # If API key is not yet configured, return baseline with notification
        if not self.is_configured:
            return baseline_result

        # Call Gemini GenAI for contextual post-processing
        try:
            return self._call_gemini_postprocessor(
                raw_tamil_text=raw_tamil_text,
                glyph_predictions=glyph_predictions,
                document_image=document_image,
                avg_confidence=avg_confidence,
                context_hints=context_hints
            )
        except Exception as e:
            print(f"Gemini GenAI call encountered error: {e}. Falling back to baseline.")
            baseline_result["status"] = "error"
            baseline_result["message"] = f"Gemini GenAI error: {str(e)}. Displaying CNN baseline."
            return baseline_result

    def _call_gemini_postprocessor(
        self,
        raw_tamil_text: str,
        glyph_predictions: List[Dict[str, Any]],
        document_image: Optional[np.ndarray],
        avg_confidence: float,
        context_hints: Optional[str]
    ) -> Dict[str, Any]:
        """Constructs prompt and calls Gemini GenAI."""

        # Format glyph sequence details for the prompt
        glyph_summary = []
        for g in glyph_predictions:
            top_alts = ", ".join([f"{c['tamil']}({c['confidence']}%)" for c in g.get("top_candidates", [])[:3]])
            glyph_summary.append(
                f"Line {g.get('line_idx', 0)+1}, Glyph {g.get('glyph_idx', 0)+1}: "
                f"Pred='{g.get('tamil')}' (Class {g.get('predicted_class')}, Conf: {g.get('confidence')}%), "
                f"Alternatives=[{top_alts}]"
            )
        glyph_details_str = "\n".join(glyph_summary[:60])  # Cap to prevent excessive prompt size

        prompt = f"""
You are an expert epigraphist, paleographer, and classical Tamil linguist specializing in ancient Tamil Palm-Leaf Manuscripts (ஓலைச்சுவடி) and Vatteluttu (வட்டெழுத்து) inscriptions.

We have scanned an ancient historical Tamil palm leaf document and passed it through a CNN Optical Character Recognition (OCR) model.
Because historical manuscripts suffer from ink fading, missing pulli (dots), archaic ligatures, and OCR noise, the raw character-by-character predictions need your contextual post-correction.

RAW RECOGNIZED TAMIL CHARACTERS (From CNN Model):
\"\"\"
{raw_tamil_text}
\"\"\"

DETAILED GLYPH OCR PREDICTIONS & TOP CANDIDATES:
\"\"\"
{glyph_details_str}
\"\"\"

ADDITIONAL CONTEXT / HINTS:
{context_hints or "Historical Tamil palm leaf manuscript / Vatteluttu document."}

YOUR TASK:
1. Contextual Post-OCR Correction: Using your deep knowledge of Classical/Historical Tamil grammar, vocabulary, and Vatteluttu orthography (e.g. restoring implicit pulli dots, correcting OCR confusions like ர/க or ப/வ based on the alternative candidate list), reconstruct the intended readable text.
2. Modern Tamil Transcription: Provide the clean, standardized Modern Tamil Unicode transcription (நவீன தமிழ் வடிவம்).
3. Transliteration: Provide standard romanized phonetic transliteration (ISO 15919 or clean English phonetics).
4. English Translation: Provide an accurate English translation conveying the meaning of the passage.
5. Historical & Epigraphical Commentary: Explain the domain (e.g. Sangam literature, Shaivite/Vaishnavite hymn, medical treatise/Siddha, astrological palm leaf, royal grant, or temple inscription), estimated linguistic era, and notable terms.
6. Corrections Log: List the key character corrections made from raw OCR to final text with brief rationale.

You MUST respond strictly with a valid JSON object matching this exact schema:
{{
  "modern_tamil": "<Clean standard modern Tamil text>",
  "transliteration": "<Romanized transliteration>",
  "english_translation": "<English translation of text>",
  "historical_commentary": "<Historical context, genre, and linguistic notes>",
  "ocr_corrections": [
    {{"original": "<char>", "corrected": "<char>", "reason": "<why corrected>"}}
  ],
  "evaluated_confidence": <number between 0 and 100 representing overall reliability>
}}
"""

        # Prepare multimodal contents if document image is available
        contents = []
        if document_image is not None:
            # Convert image to JPEG bytes
            success, encoded_img = cv2.imencode(".jpg", document_image)
            if success:
                from google.genai import types
                image_bytes = encoded_img.tobytes()
                contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))

        contents.append(prompt)

        # Call Gemini GenAI SDK
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents
        )

        response_text = response.text.strip()
        # Clean markdown codeblocks if enclosed in ```json ... ```
        if response_text.startswith("```"):
            lines = response_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            response_text = "\n".join(lines).strip()

        data = json.loads(response_text)

        modern_tamil = data.get("modern_tamil", raw_tamil_text)
        translit = data.get("transliteration", "")
        english = data.get("english_translation", "")
        commentary = data.get("historical_commentary", "")
        corrections = data.get("ocr_corrections", [])
        eval_conf = float(data.get("evaluated_confidence", avg_confidence))

        return {
            "status": "completed",
            "gemini_active": True,
            "raw_tamil": raw_tamil_text,
            "raw_transliteration": "\n".join([g.get("translit", "") for g in glyph_predictions]),
            "modern_tamil": modern_tamil,
            "transliteration": translit,
            "english_translation": english,
            "historical_commentary": commentary,
            "corrections_made": corrections,
            "overall_confidence": round(eval_conf, 2),
            "searchable_text": modern_tamil,
            "message": "Successfully processed via Google Gemini GenAI!"
        }
