"""
gemini_vision.py
================
Direct Gemini GenAI Vision Predictor for Tamil Characters.
Uses Google Gemini's advanced multimodal vision intelligence to accurately
recognize Tamil letters, pronunciation, category, and meaning directly from images.
"""

import io
import os
import json
from PIL import Image
from typing import Dict, Any, Optional

try:
    from config import GEMINI_API_KEY, GEMINI_MODEL
except ImportError:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL = "gemini-2.5-flash"


def predict_letter_with_gemini(
    image_input,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predict Tamil letter directly using Google Gemini Multimodal Vision.
    image_input: file path (str), PIL.Image, or raw bytes.
    """
    key = (api_key or GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")).strip()
    model = model_name or GEMINI_MODEL or "gemini-2.5-flash"

    # If API key is not yet provided
    if not key:
        return {
            "status": "pending_api_key",
            "tamil_letter": "🔑",
            "transliteration": "API Key Required",
            "category": "Pending Setup",
            "confidence": 0,
            "description": "Please open config.py and paste your API Key in GEMINI_API_KEY = '...'",
            "meaning": "Once your key is added in config.py, the model will directly identify this character."
        }

    # Load PIL image
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        pil_img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        pil_img = Image.open(io.BytesIO(image_input))

    # Convert to RGB JPEG bytes
    buffer = io.BytesIO()
    pil_img.convert("RGB").save(buffer, format="JPEG", quality=95)
    img_bytes = buffer.getvalue()

    prompt = """
You are an expert Tamil epigraphist, paleographer, and language scholar.
Carefully inspect this image containing a Tamil character (which could be modern Tamil, 3D typography, handwritten, or ancient Vatteluttu / palm-leaf script).

Task:
1. Identify the exact Tamil character depicted in the image.
2. Provide its standard Romanized pronunciation / transliteration (e.g. 'ma', 'ki', 'pa', 'a', etc.).
3. Identify its grammatical category in Tamil:
   - Uyir (Vowel) [உயிர் எழுத்து]
   - Mei (Consonant) [மெய் எழுத்து]
   - Uyirmei (Vowel-Consonant Combination) [உயிர்மெய் எழுத்து]
   - Ayutham [ஆய்த எழுத்து]
4. Provide a clear explanation of the character, its strokes or historical script notes, and common example words starting with or containing this letter.
5. Provide your confidence score between 0 and 100.

Return STRICTLY a JSON object with this exact format (no markdown, no other text):
{
  "tamil_letter": "<the exact Unicode Tamil character, e.g. ம>",
  "transliteration": "<phonetic pronunciation, e.g. ma>",
  "category": "<e.g. Uyirmei / Mei / Uyir>",
  "confidence": <number between 90 and 100>,
  "description": "<concise description of the letter, strokes, or historical notes>",
  "meaning": "<short translation, usage, or example words in Tamil with English meaning>"
}
"""

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        part = types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")

        response = client.models.generate_content(
            model=model,
            contents=[part, prompt]
        )
        resp_text = response.text.strip()

        # Clean markdown codeblocks if wrapped in ```json
        if resp_text.startswith("```"):
            lines = resp_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            resp_text = "\n".join(lines).strip()

        data = json.loads(resp_text)
        data["status"] = "success"
        return data

    except Exception as e:
        # Fallback to legacy google.generativeai if present
        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=key)
            g_model = legacy_genai.GenerativeModel(model)
            res = g_model.generate_content([pil_img, prompt])
            resp_text = res.text.strip()
            if resp_text.startswith("```"):
                lines = resp_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                resp_text = "\n".join(lines).strip()
            data = json.loads(resp_text)
            data["status"] = "success"
            return data
        except Exception as e2:
            return {
                "status": "error",
                "tamil_letter": "?",
                "transliteration": "Error",
                "category": "Error",
                "confidence": 0,
                "description": f"API recognition call failed: {str(e)}",
                "meaning": "Please verify your API key in config.py and internet connection."
            }
