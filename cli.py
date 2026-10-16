"""
cli.py
======
Command-Line Interface for the Historical Document Transcription Pipeline
Usage:
  python cli.py --image test_images/palmleaf_sample1.png
  python cli.py --image test_images/test1.png --api-key YOUR_GEMINI_API_KEY
  python cli.py --image test_images/palmleaf_sample2.png --output result.json
"""

import os
import sys
import json
import argparse

# Ensure standard output supports UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pipeline import HistoricalDocumentPipeline, format_searchable_output

try:
    from config import GEMINI_API_KEY, GEMINI_MODEL
except ImportError:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL = "gemini-2.5-flash"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ancient Tamil Palm-Leaf Manuscript & Historical Document Transcription Pipeline"
    )
    parser.add_argument(
        "--image", "-i",
        type=str,
        default="test_images/palmleaf_sample1.png",
        help="Path to input digitized document or character crop (default: test_images/palmleaf_sample1.png)"
    )
    parser.add_argument(
        "--api-key", "-k",
        type=str,
        default=GEMINI_API_KEY,
        help="Google Gemini API key (defaults to config.py setting)"
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default=GEMINI_MODEL,
        help="Gemini model to use (defaults to config.py setting)"
    )
    parser.add_argument(
        "--binarize",
        type=str,
        choices=["sauvola", "otsu", "adaptive"],
        default="sauvola",
        help="Binarization algorithm (default: sauvola)"
    )
    parser.add_argument(
        "--denoise",
        type=str,
        choices=["bilateral", "nlm", "median", "gaussian"],
        default="bilateral",
        help="Denoising filter (default: bilateral)"
    )
    parser.add_argument(
        "--no-deskew",
        action="store_true",
        help="Disable automatic skew angle correction"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Simulate historical degradation (stains, warping, jitter) for robustness testing"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to save output report (JSON or TXT)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.image):
        print(f"Error: File not found: {args.image}")
        sys.exit(1)

    print(f"\nProcessing Document: {args.image}")
    pipeline = HistoricalDocumentPipeline(
        gemini_api_key=args.api_key,
        gemini_model=args.gemini_model
    )

    results = pipeline.run(
        image_input=args.image,
        binarize_method=args.binarize,
        denoise_method=args.denoise,
        do_deskew=not args.no_deskew,
        simulate_augmentation=args.augment
    )

    report_str = format_searchable_output(results)
    print("\n" + report_str)

    if args.output:
        if args.output.lower().endswith(".json"):
            # Prepare serializable dict
            serializable = {
                "input_image": args.image,
                "skew_angle": results["skew_angle"],
                "total_lines": len(results["lines"]),
                "total_glyphs": len(results["segmented_glyphs"]),
                "glyph_predictions": results["glyph_predictions"],
                "final_transcription": results["final_transcription"]
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            print(f"\nJSON results saved to: {args.output}")
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(report_str)
            print(f"\nReport saved to: {args.output}")


if __name__ == "__main__":
    main()
