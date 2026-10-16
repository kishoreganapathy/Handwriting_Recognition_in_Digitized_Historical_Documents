# 📜 Tamil Character AI Predictor (Google Gemini GenAI)

A clean, minimal project that uses **Google Gemini GenAI Multimodal Vision** to directly recognize and translate Tamil characters (modern fonts, 3D typography, handwritten, or ancient palm-leaf script).

---

## 🔑 Where to Give Your Gemini API Key

Open **[`config.py`](file:///D:/nnproject/config.py)** and paste your API key on line 11:

```python
# In config.py:
GEMINI_API_KEY = "AIzaSy..."  # <-- Paste your Gemini API key here
```

Save the file, and predictions will immediately work!

---

## 🚀 How to Predict

### 1. Command-Line (CLI) Prediction
Predict any single letter image directly:

```powershell
# Predict test1.png (ம):
python predict.py test_images/test1.png

# Predict test3.jpg (கீ):
python predict.py test_images/test3.jpg

# Predict test2.jpg (பூ):
python predict.py test_images/test2.jpg
```

### 2. Streamlit Web App
Launch the web interface:

```powershell
streamlit run app.py
```
Open `http://localhost:8501`. Upload any image or choose a sample, and click **Predict Character with Gemini**!

---

## 📁 Minimal Project Structure

```
d:/nnproject/
├── config.py         # 🔑 Paste your GEMINI_API_KEY here
├── gemini_vision.py  # Gemini Multimodal Vision prediction engine
├── predict.py        # CLI single-letter prediction script
├── app.py            # Streamlit web UI
├── test_images/      # Test images (test1.png, test2.jpg, test3.jpg)
└── README.md         # Documentation
```
