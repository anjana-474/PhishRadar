import joblib
import torch
import numpy as np
import pandas as pd

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification


# ==============================
# Load URL Model
# ==============================

url_model = joblib.load("models/url_model.pkl")

# Store feature names (to avoid sklearn warning)
url_feature_names = url_model.feature_names_in_


# ==============================
# Load Text Model
# ==============================

text_model_path = "models/text_phishing_model"

tokenizer = DistilBertTokenizer.from_pretrained(text_model_path)

text_model = DistilBertForSequenceClassification.from_pretrained(text_model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_model.to(device)

text_model.eval()


# ==============================
# URL Prediction
# ==============================

def predict_url(url_features):

    # Convert to dataframe with feature names
    url_features = pd.DataFrame(url_features, columns=url_feature_names)

    prob = url_model.predict_proba(url_features)[0][1]

    return prob


# ==============================
# Text Prediction
# ==============================

def predict_text(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        outputs = text_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    phishing_prob = probs[0][1].item()

    return phishing_prob


# ==============================
# Combined Risk Score
# ==============================

def combined_risk(url_score, text_score):

    # Text slightly stronger signal
    final_score = 0.4 * url_score + 0.6 * text_score

    return final_score


# ==============================
# Full Prediction Pipeline
# ==============================

def predict_phishing(url_features, text):

    url_score = predict_url(url_features)

    text_score = predict_text(text)

    final_score = combined_risk(url_score, text_score)

    result = {
        "url_score": round(url_score, 4),
        "text_score": round(text_score, 4),
        "final_score": round(final_score, 4),
        "prediction": "Phishing" if final_score > 0.5 else "Safe"
    }

    return result


# ==============================
# Example Test Run
# ==============================

if __name__ == "__main__":

    # Dummy URL feature vector
    example_url_features = np.zeros((1, len(url_feature_names)))

    example_text = "Your account has been suspended. Click the link to verify immediately."

    result = predict_phishing(example_url_features, example_text)

    print("\n===== PhishRadar Result =====")

    print("URL Risk Score  :", result["url_score"])
    print("Text Risk Score :", result["text_score"])
    print("Final Score     :", result["final_score"])
    print("Prediction      :", result["prediction"])