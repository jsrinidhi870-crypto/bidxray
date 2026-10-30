import io
import os
import time
from datetime import date

import streamlit as st
import plotly.graph_objects as go
import pymupdf
import pytesseract
import cv2
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

# Streamlit: set_page_config must be the first st.* call
st.set_page_config(page_title="Bid X-Ray", page_icon="🔍", layout="wide")

# ====== SETTINGS ======
# API key: .env file la  GEMINI_API_KEY=unga_key  nu ezhudhunga (key-a code la podaadheenga)
API_KEY = ""
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass
API_KEY = os.getenv("GEMINI_API_KEY", API_KEY)

# First model fail aana next model try pannum
MODELS = ["gemini-2.5-flash", "gemini-3.8-flash"]
# Windows la Tesseract install pannina apram idha uncomment pannunga:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# ======================

REQUIRED_CLAUSES = [
    "penalty", "performance guarantee", "termination",
    "payment terms", "delivery schedule", "eligibility",
    "arbitration", "warranty",
]
FRAUD_KEYWORDS = [
    "sole source", "single vendor", "urgent basis",
    "relaxation", "without tender", "nomination basis",
]

# ---------------------------------------------------------------- languages
LANG_NAME = {
    "English": "English",
    "Tamil": "Tamil (தமிழ்)",
    "Hindi": "Hindi (हिन्दी, Devanagari script)",
}

T = {
    "English": {
        "tagline": "See Beyond Documents. Detect Risk Before Approval.",
        "sub": "Upload a tender or bid PDF. Bid X-Ray checks the clauses, flags suspicious signals and scores the risk in seconds.",
        "sidebar_info": "Upload a bid PDF. The app checks its clauses and reports the risk.",
        "upload": "Upload bid PDF",
        "analyzing": "Analyzing document...",
        "trust": "Trust Score", "signals": "Fraud Signals", "risk": "Risk Level", "pages": "Pages",
        "LOW": "LOW RISK", "MEDIUM": "MEDIUM RISK", "HIGH": "HIGH RISK",
        "tab_heat": "Heatmap", "tab_alerts": "Risk Alerts", "tab_clause": "Clause Finder",
        "tab_summary": "Summary", "tab_chat": "Ask Bid X-Ray",
        "heat_title": "Procurement Risk Heatmap (clause x page)",
        "heat_caption": "Green: clause appears on this page. Red: not found.",
        "no_risk": "No major risks found.",
        "missing": "Missing clause", "signal": "Suspicious signal found",
        "clause_title": "Clause Finder", "clause_col": "Clause", "status_col": "Status", "page_col": "Found on page",
        "found": "Found", "notfound": "Missing",
        "search_label": "Search any word or clause in the document",
        "search_none": "Not found in this document.",
        "summary_title": "30-second Executive Summary",
        "summary_spin": "Writing summary...",
        "chat_title": "Ask Bid X-Ray about this document",
        "chat_q": "Type your question",
        "chat_spin": "Thinking...",
        "download": "Download compliance report",
        "no_key": "Gemini API key not set. Create a .env file with GEMINI_API_KEY=your_key and restart the app.",
        "busy": "AI is busy or the key has a problem. Wait a moment and try again.",
        "f1t": "Clause check", "f1d": "8 mandatory clauses checked, with the page where each one appears.",
        "f2t": "Fraud signals", "f2d": "Flags phrases such as sole source, nomination basis and urgent basis.",
        "f3t": "Ask in your language", "f3d": "Summary and chat answers in English, Tamil or Hindi.",
        "empty": "Upload a bid PDF to start the analysis.",
    },
    "Tamil": {
        "tagline": "ஆவணங்களுக்கு அப்பால் பாருங்கள். ஒப்புதலுக்கு முன் ஆபத்தைக் கண்டறியுங்கள்.",
        "sub": "டெண்டர் அல்லது ஏல PDF-ஐ பதிவேற்றுங்கள். Bid X-Ray விதிகளை சரிபார்த்து, சந்தேக அறிகுறிகளை காட்டி, ஆபத்தை மதிப்பிடும்.",
        "sidebar_info": "ஏல PDF-ஐ பதிவேற்றுங்கள். செயலி விதிகளை சரிபார்த்து ஆபத்தை சொல்லும்.",
        "upload": "ஏல PDF பதிவேற்றுங்கள்",
        "analyzing": "ஆவணம் பகுப்பாய்வு செய்யப்படுகிறது...",
        "trust": "நம்பிக்கை மதிப்பெண்", "signals": "மோசடி அறிகுறிகள்", "risk": "ஆபத்து நிலை", "pages": "பக்கங்கள்",
        "LOW": "குறைந்த ஆபத்து", "MEDIUM": "நடுத்தர ஆபத்து", "HIGH": "அதிக ஆபத்து",
        "tab_heat": "ஹீட்மேப்", "tab_alerts": "ஆபத்து எச்சரிக்கைகள்", "tab_clause": "விதி தேடல்",
        "tab_summary": "சுருக்கம்", "tab_chat": "Bid X-Ray-யிடம் கேளுங்கள்",
        "heat_title": "கொள்முதல் ஆபத்து ஹீட்மேப் (விதி x பக்கம்)",
        "heat_caption": "பச்சை: இந்தப் பக்கத்தில் விதி உள்ளது. சிவப்பு: இல்லை.",
        "no_risk": "பெரிய ஆபத்துகள் எதுவும் இல்லை.",
        "missing": "விடுபட்ட விதி", "signal": "சந்தேக அறிகுறி கண்டறியப்பட்டது",
        "clause_title": "விதி தேடல்", "clause_col": "விதி", "status_col": "நிலை", "page_col": "உள்ள பக்கம்",
        "found": "உள்ளது", "notfound": "இல்லை",
        "search_label": "ஆவணத்தில் எந்த சொல் அல்லது விதியையும் தேடுங்கள்",
        "search_none": "இந்த ஆவணத்தில் கிடைக்கவில்லை.",
        "summary_title": "30 வினாடி நிர்வாகச் சுருக்கம்",
        "summary_spin": "சுருக்கம் எழுதப்படுகிறது...",
        "chat_title": "இந்த ஆவணத்தைப் பற்றி கேளுங்கள்",
        "chat_q": "உங்கள் கேள்வியை எழுதுங்கள்",
        "chat_spin": "யோசிக்கிறது...",
        "download": "அறிக்கையைப் பதிவிறக்குங்கள்",
        "no_key": "Gemini API key அமைக்கப்படவில்லை. .env கோப்பில் GEMINI_API_KEY=உங்கள்_key என எழுதி செயலியை மீண்டும் தொடங்குங்கள்.",
        "busy": "AI பிஸியாக உள்ளது அல்லது key-யில் சிக்கல். சிறிது நேரம் கழித்து மீண்டும் முயற்சிக்கவும்.",
        "f1t": "விதி சரிபார்ப்பு", "f1d": "8 கட்டாய விதிகள் சரிபார்க்கப்பட்டு, எந்தப் பக்கத்தில் உள்ளன என்று காட்டப்படும்.",
        "f2t": "மோசடி அறிகுறிகள்", "f2d": "sole source, nomination basis, urgent basis போன்ற சொற்றொடர்களை குறிக்கும்.",
        "f3t": "உங்கள் மொழியில் கேளுங்கள்", "f3d": "சுருக்கமும் பதில்களும் ஆங்கிலம், தமிழ், இந்தியில் கிடைக்கும்.",
        "empty": "பகுப்பாய்வைத் தொடங்க ஏல PDF-ஐ பதிவேற்றுங்கள்.",
    },
    "Hindi": {
        "tagline": "दस्तावेज़ों के पार देखें। मंज़ूरी से पहले जोखिम पहचानें।",
        "sub": "टेंडर या बोली की PDF अपलोड करें। Bid X-Ray क्लॉज़ जाँचता है, संदिग्ध संकेत दिखाता है और जोखिम का स्कोर देता है।",
        "sidebar_info": "बोली की PDF अपलोड करें। ऐप क्लॉज़ जाँचकर जोखिम बताएगा।",
        "upload": "बोली की PDF अपलोड करें",
        "analyzing": "दस्तावेज़ का विश्लेषण हो रहा है...",
        "trust": "ट्रस्ट स्कोर", "signals": "धोखाधड़ी के संकेत", "risk": "जोखिम स्तर", "pages": "पेज",
        "LOW": "कम जोखिम", "MEDIUM": "मध्यम जोखिम", "HIGH": "उच्च जोखिम",
        "tab_heat": "हीटमैप", "tab_alerts": "जोखिम चेतावनी", "tab_clause": "क्लॉज़ खोज",
        "tab_summary": "सारांश", "tab_chat": "Bid X-Ray से पूछें",
        "heat_title": "खरीद जोखिम हीटमैप (क्लॉज़ x पेज)",
        "heat_caption": "हरा: क्लॉज़ इस पेज पर है। लाल: नहीं मिला।",
        "no_risk": "कोई बड़ा जोखिम नहीं मिला।",
        "missing": "गायब क्लॉज़", "signal": "संदिग्ध संकेत मिला",
        "clause_title": "क्लॉज़ खोज", "clause_col": "क्लॉज़", "status_col": "स्थिति", "page_col": "किस पेज पर मिला",
        "found": "मिला", "notfound": "गायब",
        "search_label": "दस्तावेज़ में कोई भी शब्द या क्लॉज़ खोजें",
        "search_none": "इस दस्तावेज़ में नहीं मिला।",
        "summary_title": "30 सेकंड का कार्यकारी सारांश",
        "summary_spin": "सारांश लिखा जा रहा है...",
        "chat_title": "इस दस्तावेज़ के बारे में पूछें",
        "chat_q": "अपना सवाल लिखें",
        "chat_spin": "सोच रहा है...",
        "download": "रिपोर्ट डाउनलोड करें",
        "no_key": "Gemini API key सेट नहीं है। .env फ़ाइल में GEMINI_API_KEY=आपकी_key लिखें और ऐप दोबारा चलाएँ।",
        "busy": "AI व्यस्त है या key में समस्या है। थोड़ी देर बाद फिर कोशिश करें।",
        "f1t": "क्लॉज़ जाँच", "f1d": "8 ज़रूरी क्लॉज़ जाँचे जाते हैं, साथ में पेज नंबर भी।",
        "f2t": "धोखाधड़ी के संकेत", "f2d": "sole source, nomination basis, urgent basis जैसे वाक्यांश पकड़ता है।",
        "f3t": "अपनी भाषा में पूछें", "f3d": "सारांश और जवाब अंग्रेज़ी, तमिल या हिन्दी में मिलते हैं।",
        "empty": "विश्लेषण शुरू करने के लिए बोली की PDF अपलोड करें।",
    },
}

LEVEL_COLOR = {"LOW": "#3FB68B", "MEDIUM": "#F2B544", "HIGH": "#E5534B"}

# ---------------------------------------------------------------- voice verdict
GTTS_CODE = {"English": "en", "Tamil": "ta", "Hindi": "hi"}
VOICE = {
    "English": {
        "btn": "Play voice verdict",
        "install": "Voice needs the gTTS package and internet. Run: pip install gtts",
        "intro": "Bid analysis complete. Trust score is {score} out of 100. ",
        "HIGH": "Risk level is high. {n_missing} important clauses are missing and {n_sig} suspicious signals were found. This bid is not safe to approve without review.",
        "MEDIUM": "Risk level is medium. {n_missing} clauses are missing and {n_sig} suspicious signals were found. This bid can be used after the missing clauses are fixed.",
        "LOW": "Risk level is low. This bid looks safe and can be used.",
    },
    "Tamil": {
        "btn": "குரலில் முடிவு கேளுங்கள்",
        "install": "குரலுக்கு gTTS மற்றும் இணையம் தேவை. இதை இயக்குங்கள்: pip install gtts",
        "intro": "ஏல ஆவண பகுப்பாய்வு முடிந்தது. நம்பிக்கை மதிப்பெண் நூற்றுக்கு {score}. ",
        "HIGH": "ஆபத்து நிலை அதிகம். {n_missing} முக்கிய விதிகள் இல்லை, {n_sig} சந்தேக அறிகுறிகள் உள்ளன. மறுஆய்வு இல்லாமல் ஒப்புதல் தருவது பாதுகாப்பானது அல்ல.",
        "MEDIUM": "ஆபத்து நிலை நடுத்தரம். {n_missing} விதிகள் இல்லை, {n_sig} சந்தேக அறிகுறிகள் உள்ளன. விடுபட்ட விதிகளை சரிசெய்த பிறகு இந்த ஏலத்தை பயன்படுத்தலாம்.",
        "LOW": "ஆபத்து நிலை குறைவு. இந்த ஏலம் பாதுகாப்பானது, பயன்படுத்தலாம்.",
    },
    "Hindi": {
        "btn": "आवाज़ में नतीजा सुनें",
        "install": "आवाज़ के लिए gTTS पैकेज और इंटरनेट चाहिए। चलाएँ: pip install gtts",
        "intro": "बोली का विश्लेषण पूरा हुआ। ट्रस्ट स्कोर सौ में से {score} है। ",
        "HIGH": "जोखिम स्तर उच्च है। {n_missing} ज़रूरी क्लॉज़ गायब हैं और {n_sig} संदिग्ध संकेत मिले हैं। बिना समीक्षा के इसे मंज़ूर करना सुरक्षित नहीं है।",
        "MEDIUM": "जोखिम स्तर मध्यम है। {n_missing} क्लॉज़ गायब हैं और {n_sig} संदिग्ध संकेत मिले हैं। गायब क्लॉज़ ठीक करने के बाद इस बोली का उपयोग किया जा सकता है।",
        "LOW": "जोखिम स्तर कम है। यह बोली सुरक्षित लगती है और इसका उपयोग किया जा सकता है।",
    },
}


def voice_text(lang, lk, score, n_missing, n_sig):
    v = VOICE[lang]
    return (v["intro"] + v[lk]).format(score=score, n_missing=n_missing, n_sig=n_sig)


@st.cache_data(show_spinner=False)
def tts_bytes(text, code):
    buf = io.BytesIO()
    gTTS(text=text, lang=code).write_to_fp(buf)
    return buf.getvalue()

# ---------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=Noto+Sans+Devanagari:wght@400;600&family=Noto+Sans+Tamil:wght@400;600&display=swap');

:root {
  --ink: #0E1830; --panel: #16223F; --panel2: #1D2B4D; --line: #2C3D66;
  --text: #E9EEF9; --muted: #9FB0D3; --amber: #F2B544;
}
html, body, .stApp, [class*="css"] {
  font-family: 'IBM Plex Sans', 'Noto Sans Devanagari', 'Noto Sans Tamil', sans-serif;
}
.stApp { background: var(--ink); color: var(--text); }
.stApp p, .stApp label, .stApp span, .stApp li, .stApp div { color: inherit; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
.block-container { padding-top: 1.6rem; max-width: 1180px; }

.hero {
  display: grid; grid-template-columns: 1.4fr 1fr; gap: 2rem; align-items: center;
  background: linear-gradient(120deg, #16223F 0%, #1B2C57 100%);
  border: 1px solid var(--line); border-radius: 18px; padding: 2.2rem 2.4rem; margin-bottom: 1.4rem;
}
.hero h1 { font-family: 'Fraunces', serif; font-size: 2.9rem; margin: 0 0 .3rem 0; color: var(--text); letter-spacing: -0.5px; }
.hero .tag { font-size: 1.12rem; color: var(--amber); font-weight: 600; margin-bottom: .6rem; }
.hero .sub { color: var(--muted); max-width: 52ch; line-height: 1.55; }

/* the "x-ray": a document with a scan line over it, risky lines in red */
.doc { position: relative; background: #F4F6FB; border-radius: 10px; padding: 18px 18px 14px; transform: rotate(2deg);
       box-shadow: 0 14px 34px rgba(0,0,0,.45); overflow: hidden; }
.doc i { display: block; height: 8px; border-radius: 4px; background: #C9D1E4; margin-bottom: 11px; }
.doc i.w1 { width: 92%; } .doc i.w2 { width: 78%; } .doc i.w3 { width: 86%; } .doc i.w4 { width: 60%; }
.doc i.risk { background: #E5534B; } .doc i.warn { background: #F2B544; }
.doc .scan { position: absolute; left: 0; right: 0; height: 3px; background: #3FB68B; box-shadow: 0 0 14px 3px rgba(63,182,139,.7);
             animation: scan 3.2s ease-in-out infinite alternate; }
@keyframes scan { from { top: 8%; } to { top: 90%; } }
@media (prefers-reduced-motion: reduce) { .doc .scan { animation: none; top: 50%; } }

.kpi { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.2rem; height: 100%; }
.kpi .l { color: var(--muted); font-size: .88rem; margin-bottom: .2rem; }
.kpi .v { font-family: 'Fraunces', serif; font-size: 2.1rem; font-weight: 700; line-height: 1.15; }

.feat { background: var(--panel); border: 1px solid var(--line); border-left: 4px solid var(--amber);
        border-radius: 12px; padding: 1.1rem 1.2rem; height: 100%; }
.feat b { font-size: 1.05rem; }
.feat p { color: var(--muted); margin: .35rem 0 0 0; line-height: 1.5; }

.alert { border-radius: 10px; padding: .7rem 1rem; margin-bottom: .5rem; border: 1px solid var(--line); background: var(--panel); }
.alert.miss { border-left: 4px solid #E5534B; }
.alert.sig { border-left: 4px solid #F2B544; }
.alert.ok { border-left: 4px solid #3FB68B; }

.snip { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: .6rem .9rem; margin-bottom: .5rem; color: var(--muted); }
.snip b { color: var(--amber); }

.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { color: var(--muted); padding: .5rem 1rem; }
.stTabs [aria-selected="true"] { color: var(--amber) !important; }
[data-testid="stFileUploader"] section { background: var(--panel); border: 1px dashed var(--line); border-radius: 12px; }
.stButton > button, .stDownloadButton > button {
  background: var(--amber); color: #1B1405; border: 0; border-radius: 10px; font-weight: 600; padding: .55rem 1.2rem;
}
.stTextInput input { background: var(--panel); color: var(--text); border: 1px solid var(--line); }
@media (max-width: 800px) { .hero { grid-template-columns: 1fr; } .hero h1 { font-size: 2.2rem; } .doc { display: none; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- analysis
def extract_pages(pdf_bytes):
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page in doc:
        text = page.get_text()
        if len(text.strip()) < 50:  # scanned page na OCR
            try:
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                text = pytesseract.image_to_string(gray)
            except Exception:
                text = ""
        pages.append(text)
    return pages


def find_clauses(text):
    t = text.lower()
    return {c: (c in t) for c in REQUIRED_CLAUSES}


def clause_pages(pages):
    """clause -> list of page numbers where it appears"""
    return {c: [i + 1 for i, p in enumerate(pages) if c in p.lower()] for c in REQUIRED_CLAUSES}


def fraud_signals(text):
    t = text.lower()
    return [k for k in FRAUD_KEYWORDS if k in t]


def trust_score(clause_result, signals):
    score = int(sum(clause_result.values()) / len(clause_result) * 100)
    return max(0, score - 10 * len(signals))


def risk_key(score):
    if score >= 75:
        return "LOW"
    if score >= 40:
        return "MEDIUM"
    return "HIGH"


@st.cache_data(show_spinner=False)
def analyze(pdf_bytes):
    pages = extract_pages(pdf_bytes)
    text = "\n".join(pages)
    clauses = find_clauses(text)
    signals = fraud_signals(text)
    score = trust_score(clauses, signals)
    return pages, text, clauses, signals, score, clause_pages(pages)


def search_text(pages, word, limit=10):
    word = word.lower().strip()
    hits = []
    if not word:
        return hits
    for i, p in enumerate(pages):
        low = p.lower()
        pos = low.find(word)
        while pos != -1 and len(hits) < limit:
            s = max(0, pos - 80)
            e = min(len(p), pos + len(word) + 120)
            hits.append((i + 1, p[s:e].replace("\n", " ")))
            pos = low.find(word, pos + len(word) + 120)
        if len(hits) >= limit:
            break
    return hits


# ---------------------------------------------------------------- AI
client = genai.Client(api_key=API_KEY) if API_KEY else None


def ask_ai(prompt, lang):
    if client is None:
        return "⚠️ " + T[lang]["no_key"]
    cache = st.session_state.setdefault("ai_cache", {})
    key = (lang, prompt)
    if key in cache:
        return cache[key]
    # language rule goes in the system instruction, so the model cannot ignore it
    config = types.GenerateContentConfig(
        system_instruction=(
            f"Write your entire answer in {LANG_NAME[lang]}. "
            f"Do not answer in any other language, even if the document is in English."
        )
    )
    last_error = ""
    for model in MODELS:
        for attempt in range(2):
            try:
                res = client.models.generate_content(model=model, contents=prompt, config=config)
                cache[key] = res.text
                return res.text
            except Exception as e:
                last_error = str(e)
                if "404" in last_error or "NOT_FOUND" in last_error:
                    break  # model name wrong: go to next model
                time.sleep(2 * (attempt + 1))
    return f"⚠️ {T[lang]['busy']}\n\nError: {last_error[:300]}"


# ---------------------------------------------------------------- UI
lang = st.sidebar.selectbox("Language / மொழி / भाषा", ["English", "Tamil", "Hindi"])
t = T[lang]
st.sidebar.info(t["sidebar_info"])

st.markdown(
    f"""
<div class="hero">
  <div>
    <h1>Bid X-Ray</h1>
    <div class="tag">{t['tagline']}</div>
    <div class="sub">{t['sub']}</div>
  </div>
  <div class="doc">
    <i class="w1"></i><i class="w2"></i><i class="w3 risk"></i><i class="w1"></i>
    <i class="w4"></i><i class="w2 warn"></i><i class="w3"></i><i class="w1"></i><i class="w4"></i>
    <div class="scan"></div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

file = st.file_uploader(t["upload"], type="pdf")

if not file:
    st.caption(t["empty"])
    f1, f2, f3 = st.columns(3)
    for col, k in zip((f1, f2, f3), ("f1", "f2", "f3")):
        col.markdown(
            f'<div class="feat"><b>{t[k + "t"]}</b><p>{t[k + "d"]}</p></div>',
            unsafe_allow_html=True,
        )
else:
    with st.spinner(t["analyzing"]):
        pages, text, clauses, signals, score, cp = analyze(file.getvalue())
    lk = risk_key(score)
    level_label = t[lk]
    color = LEVEL_COLOR[lk]

    # --- KPI row: gauge + 3 cards
    g, k1, k2, k3 = st.columns([1.3, 1, 1, 1])
    with g:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            number={"font": {"size": 44, "color": "#E9EEF9"}, "suffix": "/100"},
            title={"text": t["trust"], "font": {"size": 15, "color": "#9FB0D3"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9FB0D3"},
                "bar": {"color": color},
                "bgcolor": "#1D2B4D", "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#3A2A3F"},
                    {"range": [40, 75], "color": "#3A3550"},
                    {"range": [75, 100], "color": "#24444A"},
                ],
            },
        ))
        gauge.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=10),
                            paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E9EEF9"})
        st.plotly_chart(gauge, use_container_width=True)
    k1.markdown(f'<div class="kpi"><div class="l">{t["risk"]}</div><div class="v" style="color:{color}">{level_label}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi"><div class="l">{t["signals"]}</div><div class="v">{len(signals)}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi"><div class="l">{t["pages"]}</div><div class="v">{len(pages)}</div></div>', unsafe_allow_html=True)

    # --- Voice verdict
    n_missing = sum(1 for ok in clauses.values() if not ok)
    if gTTS is None:
        st.caption(VOICE[lang]["install"])
    elif st.button("🔊 " + VOICE[lang]["btn"]):
        try:
            audio = tts_bytes(voice_text(lang, lk, score, n_missing, len(signals)), GTTS_CODE[lang])
            st.audio(audio, format="audio/mp3", autoplay=True)
        except Exception as e:
            st.warning(f"Voice error (internet check pannunga): {str(e)[:150]}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [t["tab_heat"], t["tab_alerts"], t["tab_clause"], t["tab_summary"], t["tab_chat"]]
    )

    # --- Heatmap
    with tab1:
        st.subheader(t["heat_title"])
        matrix = [[1 if p in cp[c] else 0 for p in range(1, len(pages) + 1)] for c in REQUIRED_CLAUSES]
        fig = go.Figure(go.Heatmap(
            z=matrix,
            x=[f"{i + 1}" for i in range(len(pages))],
            y=[c.title() for c in REQUIRED_CLAUSES],
            colorscale=[[0, "#B8423B"], [1, "#3FB68B"]],
            zmin=0, zmax=1, showscale=False, xgap=3, ygap=3,
        ))
        fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={"color": "#E9EEF9"}, xaxis_title=t["pages"], margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(t["heat_caption"])

    # --- Risk alerts
    with tab2:
        missing = [c for c, ok in clauses.items() if not ok]
        if not missing and not signals:
            st.markdown(f'<div class="alert ok">{t["no_risk"]}</div>', unsafe_allow_html=True)
        for c in missing:
            st.markdown(f'<div class="alert miss"><b>{t["missing"]}:</b> {c}</div>', unsafe_allow_html=True)
        for s in signals:
            st.markdown(f'<div class="alert sig"><b>{t["signal"]}:</b> "{s}"</div>', unsafe_allow_html=True)

    # --- Clause finder (exact page reference)
    with tab3:
        st.subheader(t["clause_title"])
        rows = [
            {
                t["clause_col"]: c.title(),
                t["status_col"]: t["found"] if cp[c] else t["notfound"],
                t["page_col"]: ", ".join(str(p) for p in cp[c]) if cp[c] else "-",
            }
            for c in REQUIRED_CLAUSES
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
        word = st.text_input(t["search_label"], key="clause_search")
        if word:
            hits = search_text(pages, word)
            if not hits:
                st.caption(t["search_none"])
            for pg, snippet in hits:
                st.markdown(f'<div class="snip"><b>{t["page_col"]}: {pg}</b><br>...{snippet}...</div>', unsafe_allow_html=True)

    # --- Executive summary
    with tab4:
        st.subheader(t["summary_title"])
        summary_prompt = (
            "Summarize this government bid document for a busy approver. "
            "Give 5 short bullet points that can be read in 30 seconds: what is being bought, "
            "key deadlines, key conditions, and the main risks.\n\n" + text[:15000]
        )
        with st.spinner(t["summary_spin"]):
            summary = ask_ai(summary_prompt, lang)
        st.write(summary)

    # --- Ask Bid X-Ray
    with tab5:
        st.subheader(t["chat_title"])
        q = st.text_input(t["chat_q"], key="chat_q")
        if q:
            with st.spinner(t["chat_spin"]):
                ans = ask_ai(
                    f"Document:\n{text[:15000]}\n\nQuestion: {q}\n"
                    f"Answer only from the document. If the document does not say, say so.",
                    lang,
                )
            st.write(ans)

    # --- Report
    report = (
        f"BID X-RAY REPORT\nDate: {date.today()}\nFile: {file.name}\n\n"
        f"Trust Score: {score}/100\nRisk Level: {level_label}\n"
        f"Fraud Signals: {len(signals)}\n\nCLAUSE CHECK\n"
        + "\n".join(
            f"- {c}: " + (f"FOUND (page {', '.join(map(str, cp[c]))})" if ok else "MISSING")
            for c, ok in clauses.items()
        )
        + "\n\nFRAUD SIGNALS\n"
        + ("\n".join(f"- {s}" for s in signals) if signals else "None")
        + f"\n\nEXECUTIVE SUMMARY\n{summary}\n"
    )
    st.download_button(t["download"], report, file_name="bidxray_report.txt")