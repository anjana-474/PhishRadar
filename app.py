#app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import torch
import time
import re
from datetime import datetime
import plotly.graph_objects as go

from urllib.parse import urlparse
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from rapidfuzz import fuzz, distance

import os
import zipfile
import gdown
import joblib


# -----------------------------
# Download using gdown (BEST for Google Drive)
# -----------------------------
def download_file(url, output_path):
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

# -----------------------------
# Extract zip safely
# -----------------------------
def extract_zip(zip_path, extract_to):
    if not os.path.exists("models/text_phishing_model"):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

# -----------------------------
# Load models
# -----------------------------
@st.cache_resource
def load_models():

    url_model_link = "https://drive.google.com/uc?id=1LQvuRC-i4OhmtssiERhUrnZaml3U2kBP"
    text_model_link = "https://drive.google.com/uc?id=1Cg6SfCPndw3M1DUY7x8KiWSDA379XJpS"

    # Download models
    download_file(url_model_link, "url_model.pkl")
    download_file(text_model_link, "text_model.zip")

    # Extract zip
    extract_zip("text_model.zip", "models")

    # Load models
    url_model = joblib.load("url_model.pkl")

    tokenizer = DistilBertTokenizer.from_pretrained("models/text_phishing_model")
    text_model = DistilBertForSequenceClassification.from_pretrained("models/text_phishing_model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model.to(device)
    text_model.eval()

    return url_model, tokenizer, text_model, device

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="PhishRadar",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown(
"""
<style>
    /* Add a nice subtle gradient to the header */
    .main-header {
        background: -webkit-linear-gradient(45deg, #4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        padding-bottom: 0;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.2rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    /* Threat Cards */
    .threat-card {
        background-color: rgba(43, 43, 54, 0.4);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
        border-left: 6px solid #2ecc71;
    }
    .threat-card.danger {
        border-left-color: #e74c3c;
        background-color: rgba(59, 40, 41, 0.4);
    }
    .threat-icon {
        font-size: 1.5rem;
        margin-right: 10px;
        vertical-align: middle;
    }
    .threat-title {
        font-size: 1.1rem;
        font-weight: 600;
        vertical-align: middle;
    }
    .threat-desc {
        font-size: 0.95rem;
        color: #a0a0b0;
        margin-top: 8px;
        margin-left: 38px;
    }
    
    /* Hover Animation for Cards */
    .threat-card {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .threat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.3);
    }
    .threat-card.danger:hover {
        box-shadow: 0 8px 15px rgba(231, 76, 60, 0.2);
    }
    .threat-card.safe:hover {
        box-shadow: 0 8px 15px rgba(46, 204, 113, 0.1);
    }

    /* Style Streamlit Inputs & Buttons */
    div[data-baseweb="input"] > div {
        background-color: rgba(30, 30, 40, 0.8) !important;
        border: 1px solid #333 !important;
        border-radius: 8px !important;
        transition: border 0.3s ease, box-shadow 0.3s ease;
    }
    div[data-baseweb="input"] > div:focus-within {
        border: 1px solid #00f2fe !important;
        box-shadow: 0 0 10px rgba(0, 242, 254, 0.3) !important;
    }
    
    /* Hide default Streamlit form wrappers if border=False isn't natively supported */
    [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
    }

    /* Primary buttons */
    button[kind="primary"], button[kind="primaryFormSubmit"] {
        background: linear-gradient(45deg, #ff416c, #ff4b2b) !important;
        color: #ffffff !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 8px !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(255, 65, 108, 0.4) !important;
    }

    /* Modern Segmented Control "Pill" Tabs */
    div[data-baseweb="tab-list"] {
        background-color: rgba(20, 20, 28, 0.6) !important;
        border-radius: 12px !important;
        padding: 6px !important;
        gap: 10px !important;
        border: 1px solid #333 !important;
        width: 100% !important;
    }
    button[data-baseweb="tab"] {
        flex: 1 !important;
        background-color: transparent !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        color: #a0a0b0 !important;
        padding: 12px 24px !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #000000 !important;
        background: linear-gradient(45deg, #4facfe, #00f2fe) !important;
        box-shadow: 0 4px 12px rgba(0, 242, 254, 0.3) !important;
    }
    /* Hide the default underline highlight and border completely */
    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
        display: none !important;
    }
</style>

<div class='main-header'>🛡️ PhishRadar</div>
<div class='sub-header'>Real-Time Phishing URL & Text Classification Using Transformer Models</div>
""",
unsafe_allow_html=True
)

st.divider()

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Gauge Chart Function
# -----------------------------
def create_gauge_chart(score, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={'suffix': "%", 'font': {'size': 24, 'color': "white"}},
        title={'text': title, 'font': {'size': 18, 'color': "white"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.2)"},
            'bar': {'color': "rgba(0,0,0,0)"}, 
            'steps': [
                {'range': [0, 20], 'color': "#2ecc71"},   
                {'range': [20, 50], 'color': "#f1c40f"}, 
                {'range': [50, 100], 'color': "#e74c3c"}  
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score * 100
            }
        }
    ))
    fig.update_layout(
        height=250, 
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"}
    )
    return fig


with st.spinner("Loading AI models..."):
    url_model, tokenizer, text_model, device = load_models()
    feature_names = url_model.feature_names_in_


# -----------------------------
# Popular Domains
# -----------------------------
@st.cache_data
def load_popular_domains():
    try:
        df = pd.read_csv("dataset/top-1m.csv", nrows=10000, header=None, names=["rank", "domain"])
        # Some rows might be NA
        return df["domain"].dropna().astype(str).tolist()
    except Exception as e:
        return [
            "google.com","paypal.com","facebook.com","amazon.com","apple.com",
            "microsoft.com","instagram.com","linkedin.com","twitter.com","x.com",
            "snapchat.com","tiktok.com","gmail.com","outlook.com","yahoo.com","office.com",
            "office365.com", "microsoft365.com", "proton.me","stripe.com","paytm.com","phonepe.com","razorpay.com",
            "flipkart.com","ebay.com","shopify.com","aliexpress.com",
            "github.com","gitlab.com","dropbox.com","cloudflare.com",
            "netflix.com","spotify.com","youtube.com", "walmart.com",
            "sbi.co.in","icicibank.com","hdfcbank.com","axisbank.com"
        ]

popular_domains = load_popular_domains()
popular_roots = [d.split(".")[0] for d in popular_domains]
popular_roots_set = set(popular_roots)


# -----------------------------
# Suspicious Keywords
# -----------------------------
suspicious_keywords = [
    "login","verify","update","secure","account",
    "bank","password","confirm","signin","authentication"
]

def get_domain(url):

    # add scheme if missing
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)

    domain = parsed.netloc.lower()

    # remove www
    domain = domain.replace("www.", "")

    return domain

def is_valid_url(url):

    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    parsed = urlparse(url)

    # must contain domain and dot
    if parsed.netloc == "" or "." not in parsed.netloc:
        return False

    # avoid extremely short domains
    domain = parsed.netloc.replace("www.","")
    if len(domain) < 4:
        return False

    return True

# -----------------------------
# Homograph Detection
# -----------------------------
def normalize_domain(domain):

    replacements = {
        "0": "o",
        "1": "l",
        "3": "e",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
        "|": "l"
    }

    # single character replacements
    for k, v in replacements.items():
        domain = domain.replace(k, v)

    # visual spoof patterns
    visual_spoofs = {
        "rn": "m",
        "vv": "w",
        "cl": "d",
        "ci": "d",
        "ii": "n",
        "oo": "o",
        "l0": "lo",
        "0l": "ol",
        "I": "l"
    }

    for k, v in visual_spoofs.items():
        domain = domain.replace(k, v)

    return domain

# -----------------------------
# URL Feature Extraction
# -----------------------------
def extract_url_features(url):

    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    parsed = urlparse(url)

    domain = parsed.netloc
    path = parsed.path

    features = {}

    features["url_length"] = len(url)
    features["num_dots"] = url.count(".")
    features["num_hyphens"] = url.count("-")
    features["num_digits"] = sum(c.isdigit() for c in url)
    features["has_at"] = 1 if "@" in url else 0
    features["https"] = 1 if parsed.scheme == "https" else 0
    features["num_subdomains"] = domain.count(".")
    features["path_length"] = len(path)

    # suspicious keyword count
    features["suspicious_words"] = sum(word in url.lower() for word in suspicious_keywords)

    # ip detection
    features["has_ip"] = 1 if re.match(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", domain) else 0

    return features


# -----------------------------
# Prepare DataFrame
# -----------------------------
def prepare_url_features(url):

    features = extract_url_features(url)

    df = pd.DataFrame([features])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    return df


# -----------------------------
# Brand Detection
# -----------------------------
# Precompute normalized roots for speed
normalized_popular_roots = [normalize_domain(r) for r in popular_roots]

def detect_impersonation(url):

    domain = get_domain(url)

    normalized = normalize_domain(domain)

    original_root = domain.split(".")[0]
    normalized_root = normalized.split(".")[0]

    # Stop exact legit domains from being flagged as impersonating themselves
    if domain in popular_domains or original_root in popular_roots_set:
        return False, None

    for brand, brand_root, norm_brand_root in zip(popular_domains, popular_roots, normalized_popular_roots):

        # detect homograph attack
        if normalized_root == norm_brand_root and original_root != brand_root:
            return True, brand

        # minor optimization to speed up 10,000 matches
        if abs(len(normalized_root) - len(norm_brand_root)) > 3:
            continue

        similarity = fuzz.ratio(normalized_root, norm_brand_root)

        if similarity > 85 and normalized_root != norm_brand_root:
            return True, brand

    return False, None

def detect_typosquatting(url):

    domain = get_domain(url)
    domain_root = normalize_domain(domain).split(".")[0]

    popular_roots = [d.split(".")[0] for d in popular_domains]

    # exact brand match → not typosquatting
    if domain_root in popular_roots:
        return False, None

    for brand in popular_domains:

        brand_root = brand.split(".")[0]

        # 1️⃣ similarity detection
        similarity = fuzz.ratio(domain_root, brand_root)

        # 2️⃣ edit distance detection
        edit_dist = distance.Levenshtein.distance(domain_root, brand_root)

        # 3️⃣ repeated letter attack detection
        simplified = re.sub(r'(.)\1+', r'\1', domain_root)

        repeated_attack = simplified == brand_root and domain_root != brand_root

        if (similarity > 88 and edit_dist <= 3) or repeated_attack:
            return True, brand

    return False, None
    

def detect_phishing_keywords(url):

    phishing_words = [
        "login","verify","update","secure",
        "account","bank","password","confirm",
        "billing","security"
    ]

    found = []

    for word in phishing_words:
        if word in url.lower():
            found.append(word)

    return found

def detect_ip_url(url):

    if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", url):
        return True

    return False

# -----------------------------
# Text Prediction
# -----------------------------
def predict_text(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    inputs = {k: v.to(device) for k,v in inputs.items()}

    with torch.no_grad():
        outputs = text_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    return probs[0][1].item()


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h2 style='margin-bottom: 5px; color: #ffffff;'>🛡️ PhishRadar</h2>
            <p style='color: #a0a0b0; font-size: 0.95rem; margin-top: 0;'>AI-Powered Phishing Detection</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<h3 style='font-size: 1.1rem; margin-bottom: 15px;'>🕒 Recent Scans</h3>", unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("No scans yet in this session.")
    else:
        # Display the last 5 scans in reverse order (newest first)
        for item in reversed(st.session_state.history[-5:]):
            
            if item['score'] >= 0.7:
                color_icon = "🔴"
                score_color = "#e74c3c"
            elif item['score'] >= 0.35:
                color_icon = "🟡"
                score_color = "#f1c40f"
            else:
                color_icon = "🟢"
                score_color = "#2ecc71"
                
            type_label = "URL" if item['type'] == "url" else "MSG"
            
            card_html = f"""
            <div style="background-color: rgba(43, 43, 54, 0.4); padding: 12px; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid {score_color};">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <div>
                        <span style="font-size: 13px; margin-right: 4px;">{color_icon}</span>
                        <span style="font-weight: bold; font-size: 12px; color: #eee;">{type_label}</span>
                    </div>
                    <span style="font-size: 11px; color: #888;">{item['time']}</span>
                </div>
                <div style="font-family: monospace; font-size: 12px; color: #4ECDC4; word-break: break-all; margin-bottom: 6px; background-color: rgba(0,0,0,0.2); padding: 6px; border-radius: 4px;">
                    {item['target']}
                </div>
                <div style="font-size: 12px; color: #bbb;">
                    Risk Score: <span style="color: {score_color}; font-weight: bold;">{item['score']*100:.1f}%</span>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<br><div style='text-align: center; color: #666; font-size: 12px;'>Cybersecurity ML Project</div>", unsafe_allow_html=True)


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["🔗 URL Detection", "✉️ Email / Text Detection"])


# =====================================================
# URL Detection
# =====================================================
with tab1:
    st.markdown("### 🔍 Analyze Suspicious URL")
    st.write("Paste any link below to see if it's safe, malicious, or pretending to be a known brand.")

    with st.form("url_scan_form", clear_on_submit=False, border=False):
        col1, col2 = st.columns([4, 1])

        with col1:
            url_input = st.text_input(
                "Enter URL",
                placeholder="https://secure-login-paypal.com",
                label_visibility="collapsed"
            )
        
        with col2:
            scan_btn = st.form_submit_button("🛡️ Scan URL", use_container_width=True, type="primary")

   
        if scan_btn:

            url_input = url_input.strip()

            if url_input == "":
                st.warning("⚠️ Please enter a URL to scan.")
                st.stop()

            if not is_valid_url(url_input):
                st.error("❌ Invalid URL format. Please enter a valid website.")
                st.stop()

            if not url_input.startswith(("http://", "https://")):
                url_input = "http://" + url_input

            with st.spinner("Analyzing URL vectors and AI heuristics..."):
                time.sleep(0.5)

                df = prepare_url_features(url_input)
                url_score = url_model.predict_proba(df)[0][1]
                impersonation, brand = detect_impersonation(url_input)
                typo, typo_brand = detect_typosquatting(url_input)
                keywords = detect_phishing_keywords(url_input)
                ip_flag = detect_ip_url(url_input)

            # Combine AI + heuristic indicators
            heuristic_flag = impersonation or typo or ip_flag or keywords
            
            # Increase risk if strong heuristic detected
            final_score = max(url_score, 0.7) if heuristic_flag else url_score

            st.markdown("---")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:

                if final_score >= 0.7 or heuristic_flag:
                    risk_label = "High Risk"
                    risk_color = "#e74c3c"
                elif final_score >= 0.35:
                    risk_label = "Moderate Risk"
                    risk_color = "#f1c40f"
                else:
                    risk_label = "Low Risk"
                    risk_color = "#2ecc71"

                st.markdown("##### AI Confidence Score", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align:center; color:{risk_color}; margin-top:0px;'>{final_score*100:.2f}%</h1>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='text-align:center; color:{risk_color};'><b>{risk_label}</b></h4>", unsafe_allow_html=True)
                
                fig = create_gauge_chart(final_score, "Threat Level")
                st.plotly_chart(fig, use_container_width=True)

                # Append to session history
                st.session_state.history.append({
                    "type": "url",
                    "target": url_input[:30] + "..." if len(url_input) > 30 else url_input,
                    "score": final_score,
                    "time": datetime.now().strftime("%H:%M:%S")
                })

            with res_col2:
                if final_score >= 0.7 or heuristic_flag:
                    st.error("### 🚨 High Risk Phishing URL Detected!")
                elif final_score >= 0.35:
                    st.warning("### ⚠️ Moderate Risk URL Detected")
                else:
                    st.success("### ✅ Safe URL")
                st.markdown("#### 🕵️ Threat Intelligence Panel")
                st.write("Identified risk vectors and heuristics:")
                
                # Create stylish HTML cards for each threat attribute
                def make_card(title, desc, is_danger):
                    status_class = "danger" if is_danger else "safe"
                    icon = "🚨" if is_danger else "✅"
                    return f'''
                    <div class="threat-card {status_class}">
                        <span class="threat-icon">{icon}</span>
                        <span class="threat-title">{title}</span>
                        <div class="threat-desc">{desc}</div>
                    </div>
                    '''
                    
                tc1, tc2 = st.columns(2)
                with tc1:
                    st.markdown(make_card("Brand Impersonation", f"Spoofing target: <b>{brand}</b>" if impersonation else "Not detected", impersonation), unsafe_allow_html=True)
                    st.markdown(make_card("Suspicious Keywords", f"Found: <b>{', '.join(keywords)}</b>" if keywords else "None found", bool(keywords)), unsafe_allow_html=True)
                with tc2:
                    st.markdown(make_card("Typosquatting", f"Targeting: <b>{typo_brand}</b>" if typo else "Not detected", typo), unsafe_allow_html=True)
                    st.markdown(make_card("IP Address Use", "Detected (High Risk)" if ip_flag else "Not detected", ip_flag), unsafe_allow_html=True)

                if url_score > 0.35 and not (impersonation or typo or ip_flag or keywords):
                    st.warning("⚠️ **AI Model Flag:** Although no specific heuristic threat was matched, the AI structural score strongly suggests a malicious link.")
                    
# =====================================================
# Text Detection
# =====================================================
with tab2:
    st.markdown("### ✉️ Extract & Analyze Email Content")
    st.write("Check if a given email, SMS, or direct message contains manipulative phishing language.")

    with st.form("text_scan_form", clear_on_submit=False, border=False):
        user_text = st.text_input(
            "Paste message content",
            placeholder="Your account will be locked. Click here to verify immediately..."
        )

        t_col1, t_col2, t_col3 = st.columns([1, 1, 1])
        with t_col1:
            analyze_btn = st.form_submit_button("🔍 Analyze Message", use_container_width=True, type="primary")

    if analyze_btn:
        if user_text.strip() == "":
            st.warning("⚠️ Please enter message text to analyze.")
        else:
            with st.spinner("Running AI NLP analysis..."):
                time.sleep(0.5)
                text_score = predict_text(user_text)

            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:

                if text_score > 0.6:
                    risk_label = "High Risk"
                    risk_color = "#e74c3c"

                elif text_score > 0.3:
                    risk_label = "Moderate Risk"
                    risk_color = "#f1c40f"

                else:
                    risk_label = "Low Risk"
                    risk_color = "#2ecc71"
                st.markdown("##### Phishing Probability", unsafe_allow_html=True)
                st.markdown(f"<h1 style='text-align:center; color:{risk_color}; margin-top:0px;'>{text_score*100:.2f}%</h1>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='text-align:center; color:{risk_color};'><b>{risk_label}</b></h4>", unsafe_allow_html=True)
                
                fig = create_gauge_chart(text_score, "Spam / Phish Risk")
                st.plotly_chart(fig, use_container_width=True)

                # Append to session history
                st.session_state.history.append({
                    "type": "text",
                    "target": user_text[:30] + "..." if len(user_text) > 30 else user_text.replace('\n', ' '),
                    "score": text_score,
                    "time": datetime.now().strftime("%H:%M:%S")
                })

            with res_col2:
                # Same card generator format from Tab 1
                def make_text_card(title, desc, is_danger):
                    status_class = "danger" if is_danger else "safe"
                    icon = "🚨" if is_danger else "✅"
                    return f'''
                    <div class="threat-card {status_class}">
                        <span class="threat-icon">{icon}</span>
                        <span class="threat-title">{title}</span>
                        <div class="threat-desc">{desc}</div>
                    </div>
                    '''

                st.markdown("#### �️ DistilBERT AI Analysis")
                st.write("Transformer-based natural language threat evaluation:")
                st.markdown("<br>", unsafe_allow_html=True)
                
                if text_score > 0.5:
                    st.markdown(make_text_card("High Risk Message", "Our DistilBERT model identifies this message as a likely social engineering or phishing attempt.", True), unsafe_allow_html=True)
                else:
                    st.markdown(make_text_card("Legitimate Message", "No significant manipulative or malicious language patterns detected inside the textual structure.", False), unsafe_allow_html=True)


# -----------------------------
# Footer
# -----------------------------
st.divider()

st.markdown(
"""
<center>
🛡️ PhishRadar — AI Phishing Detection System
</center>
""",
unsafe_allow_html=True
)