"""
predict.py
==========
Direct Gemini GenAI Single-Letter Predictor for Tamil Characters.
Usage:
  python predict.py test_images/test1.png
  python predict.py test_images/test3.jpg
  python predict.py
"""

import os
import sys

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from gemini_vision import predict_letter_with_gemini
from config import GEMINI_API_KEY


def run_prediction(image_path: str):
    """Predicts a single Tamil character using Gemini GenAI Vision."""
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return

    print("=" * 60)
    print(f"Analyzing Character: {os.path.basename(image_path)}")
    print("=" * 60)

    result = predict_letter_with_gemini(image_path)

    if result.get("status") == "pending_api_key":
        print("\n⚠️ API KEY MISSING!")
        print("Please open 'config.py' and paste your API key in line 11:")
        print("  GEMINI_API_KEY = \"your_actual_api_key_here\"\n")
        return

    if result.get("status") == "error":
        print(f"\n❌ Error during prediction: {result.get('description')}")
        print("Please check your API key in config.py.\n")
        return

    print(f"  PREDICTED TAMIL LETTER : {result.get('tamil_letter')}")
    print(f"  PRONUNCIATION          : {result.get('transliteration')}")
    print(f"  GRAMMATICAL CATEGORY   : {result.get('category')}")
    print(f"  CONFIDENCE SCORE       : {result.get('confidence')}%")
    print("-" * 60)
    print(f"  DESCRIPTION            : {result.get('description')}")
    print(f"  MEANING / EXAMPLES     : {result.get('meaning')}")
    print("=" * 60)


def main():
    if len(sys.argv) > 1:
        run_prediction(sys.argv[1])
    else:
        test_folder = "test_images"
        if not os.path.exists(test_folder):
            print(f"Folder '{test_folder}' not found.")
            return

        images = [f for f in os.listdir(test_folder) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        if not images:
            print("No test images found.")
            return

        for img in images:
            run_prediction(os.path.join(test_folder, img))
            print()


if __name__ == "__main__":
    main()