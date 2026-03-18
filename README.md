<h1 align="center">🛡️ PhishRadar</h1>

<p align="center">
 # Real-Time Phishing URL & Text Detection using ML + DistilBERT
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue" />
  <img src="https://img.shields.io/badge/Streamlit-App-red" />
  <img src="https://img.shields.io/badge/MachineLearning-Scikit--learn-green" />
  <img src="https://img.shields.io/badge/NLP-DistilBERT-orange" />
  <img src="https://img.shields.io/badge/PyTorch-DeepLearning-red" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" />
</p>

<p align="center">
  🌐 <b>Live Demo:</b><br>
  <a href="https://phishradar.streamlit.app/">https://phishradar.streamlit.app/</a>
</p>

<p align="center">
  <a href="https://phishradar.streamlit.app/">
    <img src="https://img.shields.io/badge/Live-App-brightgreen?style=for-the-badge&logo=streamlit">
  </a>
</p>

<hr>

## 📌 Overview

🚀 PhishRadar is an AI-powered cybersecurity system that detects phishing attacks in real time by analyzing both **URLs** and **textual content** using **Machine Learning, DistilBERT (Transformer NLP), and heuristic-based techniques**.

PhishRadar combines multiple approaches to detect phishing:
* 🔗 **URL-based Machine Learning model** (structural feature analysis)
* ✉️ **DistilBERT-based NLP model** (text classification)
* 🧠 **Heuristic techniques** (impersonation, typosquatting, keyword detection)

It provides **real-time risk scores and interpretable insights** through an interactive Streamlit dashboard.

---

## ✨ Features

* 🔍 Real-time **URL phishing detection**
* ✉️ **Email / message phishing detection**
* 🧠 **DistilBERT-based NLP classification**
* 🏷️ **Brand impersonation detection**
* 🔤 **Typosquatting & homograph detection**
* ⚠️ Suspicious keyword analysis
* 🌐 IP-based URL detection
* 📊 Risk score visualization (Low / Moderate / High)
* 📜 Scan history tracking
* 🎨 Interactive UI using Streamlit

---

## 🏗️ System Architecture

1. User inputs URL or text  
2. URL → Feature extraction → ML model prediction  
3. Text → Tokenization → DistilBERT inference  
4. Heuristic analysis applied  
5. Risk score computed  
6. Results displayed with explanation  

---

## 🧠 Tech Stack

* **Frontend:** Streamlit  
* **Machine Learning:** Scikit-learn  
* **NLP Model:** DistilBERT (HuggingFace Transformers)  
* **Deep Learning:** PyTorch  
* **Data Processing:** Pandas, NumPy  
* **Similarity Matching:** RapidFuzz  
* **Visualization:** Plotly  

---

## 📊 Dataset Sources

* 📧 Phishing Email Dataset (Kaggle):  
  https://www.kaggle.com/datasets/subhajournal/phishingemails  

* 🔗 Phishing URL Dataset (UCI):  
  https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset  

* 🌐 Top Domains (Tranco):  
  https://tranco-list.eu/  

---

## ⚠️ Note

Datasets and trained models are not included due to GitHub size limitations.

👉 Download them from the above links and place them in the `dataset/` and `models/` folders.

---

## 📁 Project Structure

```
PhishRadar/
│── app.py
│── predict.py
│── requirements.txt
│── dataset/
│── models/
│── notebooks/
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone https://github.com/anjana-474/PhishRadar.git
cd PhishRadar
```

### 2️⃣ Create virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Run the application

```
streamlit run app.py
```

---

## 🎯 How It Works

### 🔗 URL Detection
* Extracts structural features (length, digits, symbols, etc.)
* Applies ML model for classification
* Uses heuristics:
  * Brand impersonation
  * Typosquatting
  * IP-based URLs
  * Suspicious keywords

### ✉️ Text Detection
* Uses DistilBERT transformer model
* Detects phishing language patterns like:

  * Urgency
  * Threats
  * Credential requests

---

## 📸 Demo

🌐 Live App:  
https://phishradar.streamlit.app/

---

## 🚀 Future Enhancements

* 🌐 Browser extension integration
* ☁️ Cloud deployment (AWS / GCP)
* 📱 Mobile application
* 🔗 Real-time email scanning integration
  
---
## ⭐ If you like this project

Give it a ⭐ on GitHub!
