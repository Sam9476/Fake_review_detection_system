# app.py
import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
import pandas as pd

# -----------------------------
# NLTK setup
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# -----------------------------
# Load model & vectorizer
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return model, tfidf

model, tfidf = load_model()

# -----------------------------
# Preprocessing
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    cleaned_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

# -----------------------------
# Predict single review
# -----------------------------
def predict_single(review, threshold=0.5):
    cleaned = clean_text(review)
    features = tfidf.transform([cleaned])
    proba = model.predict_proba(features)[0, 1]
    label = "FAKE" if proba >= threshold else "REAL"
    return label, proba

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fake Review Detector", layout="centered")
st.title("🛡️ Fake Review Detection System")

review_text = st.text_area("Enter your review here:", height=200)
threshold = st.slider("Fake Probability Threshold", 0.0, 1.0, 0.5)

if st.button("Predict"):
    if review_text.strip() == "":
        st.warning("Please enter a review.")
    else:
        label, proba = predict_single(review_text, threshold)
        st.subheader("Prediction Result")
        st.write(f"**Label:** {label}")
        st.write(f"**Probability of being fake:** {proba:.3f}")

        # Optional: top contributing tokens
        try:
            features = tfidf.transform([clean_text(review_text)])
            coefs = model.coef_[0]
            feature_array = np.array(tfidf.get_feature_names_out())
            contrib = features.toarray()[0] * coefs
            top_idx = np.argsort(contrib)[-10:][::-1]
            top_tokens = feature_array[top_idx]
            top_scores = contrib[top_idx]
            st.write("Top contributing tokens (approx.):")
            st.table(pd.DataFrame({"Token": top_tokens, "Score": top_scores}))
        except Exception as e:
            st.write("Could not compute token contributions:", e)
