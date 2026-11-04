"""Run this once after installing: python scripts/check_setup.py

Checks every library imports, then runs PaddleOCR on a generated test image.
The first PaddleOCR run downloads its models (~100 MB), so it can take a while.
"""
import importlib
import os
import sys

LIBS = [
    ("pymupdf", "PyMuPDF"), ("cv2", "opencv-python"), ("numpy", "numpy"),
    ("pydantic", "pydantic"), ("yaml", "pyyaml"), ("dateparser", "dateparser"),
    ("rapidfuzz", "rapidfuzz"), ("faker", "faker"), ("transformers", "transformers"),
    ("torch", "torch"), ("sentence_transformers", "sentence-transformers"),
    ("langchain", "langchain"), ("langchain_groq", "langchain-groq"),
    ("faiss", "faiss-cpu"), ("reportlab", "reportlab"), ("streamlit", "streamlit"),
    ("paddleocr", "paddleocr"), ("dotenv", "python-dotenv"),
]

ok = True
print(f"Python {sys.version.split()[0]}")
for mod, pkg in LIBS:
    try:
        m = importlib.import_module(mod)
        print(f"  OK   {pkg:22s} {getattr(m, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  FAIL {pkg:22s} {type(e).__name__}: {e}")

if not ok:
    sys.exit("\nSome packages failed. Run: pip install -r requirements.txt")

# --- PaddleOCR smoke test ---
import cv2
import numpy as np

img = np.full((120, 700, 3), 255, np.uint8)
cv2.putText(img, "FormSense test 12345", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)

from paddleocr import PaddleOCR

print("\nRunning PaddleOCR (first run downloads models)...")
try:  # PaddleOCR 3.x
    ocr = PaddleOCR(lang="en", use_doc_orientation_classify=False,
                    use_doc_unwarping=False, use_textline_orientation=False,
                    enable_mkldnn=False)
    res = ocr.predict(img)[0]
    pairs = list(zip(res["rec_texts"], res["rec_scores"]))
except TypeError:  # PaddleOCR 2.x
    ocr = PaddleOCR(lang="en", use_angle_cls=False, show_log=False)
    pairs = [(t, s) for _, (t, s) in ocr.ocr(img)[0]]

for text, score in pairs:
    print(f"  read: {text!r}  confidence={score:.2f}")

# --- Groq key ---
from dotenv import load_dotenv

load_dotenv()
print("\nGROQ_API_KEY found in .env" if os.getenv("GROQ_API_KEY")
      else "\nGROQ_API_KEY not set yet (add it to .env when you need the LLM)")
print("\nSetup looks good.")
