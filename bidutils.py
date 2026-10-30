import pymupdf as fitz
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io

# Windows la Tesseract install pannina apram idha uncomment pannunga:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

REQUIRED_CLAUSES = [
    "penalty", "performance guarantee", "termination",
    "payment terms", "delivery schedule", "eligibility",
    "arbitration", "warranty"
]

FRAUD_KEYWORDS = [
    "sole source", "single vendor", "urgent basis",
    "relaxation", "without tender", "nomination basis"
]


def extract_pages(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        text = page.get_text()
        if len(text.strip()) < 50:  # scanned page na OCR use pannum
            try:
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                text = pytesseract.image_to_string(gray)
            except Exception:
                text = ""  # Tesseract illa na skip pannum
        pages.append(text)
    return pages


def find_clauses(text):
    t = text.lower()
    return {c: (c in t) for c in REQUIRED_CLAUSES}


def clause_page_matrix(pages):
    """Each clause ku each page la irukka (1) illa (0)"""
    matrix = []
    for c in REQUIRED_CLAUSES:
        matrix.append([1 if c in p.lower() else 0 for p in pages])
    return matrix


def trust_score(clause_result, signals):
    found = sum(clause_result.values())
    score = int((found / len(clause_result)) * 100)
    score -= 10 * len(signals)  # each fraud signal ku 10 points kurayum
    return max(0, score)


def risk_level(score):
    if score >= 75:
        return "LOW RISK"
    if score >= 40:
        return "MEDIUM RISK"
    return "HIGH RISK"


def fraud_signals(text):
    t = text.lower()
    return [k for k in FRAUD_KEYWORDS if k in t]