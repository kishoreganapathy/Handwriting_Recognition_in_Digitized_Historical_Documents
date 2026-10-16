"""
config.py
=========
Central configuration file.
Paste your Google Gemini API key here:
"""

import os

# =====================================================================
# 🔑 PASTE YOUR GEMINI GENAI API KEY HERE (Between the quotes):
# =====================================================================
GEMINI_API_KEY =

# (Optional fallback to environment variable if empty)
if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# Gemini Model Selection
GEMINI_MODEL = "gemini-2.5-flash"
