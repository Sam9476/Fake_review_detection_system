import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import joblib
import numpy as np

# -----------------------------
# NLTK setup
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# -----------------------------
# Load models with caching
# -----------------------------
@st.cache_resource
def load_model_files():
    tfidf = joblib.load("tfidf.joblib")
    model = joblib.load("model.joblib")
    return tfidf, model

tfidf, model = load_model_files()

# -----------------------------
# Text preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

# -----------------------------
# Prediction function
# -----------------------------
def predict_single(review, threshold=0.5):
    cleaned = clean_text(review)
    vector = tfidf.transform([cleaned])
    proba = model.predict_proba(vector)[0][1]  # Probability of class 1 (fake)
    label = "Fake" if proba >= threshold else "Real"
    return label, proba

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fake Review Detection", page_icon="🕵️‍♂️")
st.title("Fake Review Detection System")

review_text = st.text_area("Enter Review Text Here:")

threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.05)

if st.button("Predict"):
    if not review_text.strip():
        st.warning("Please enter a review text.")
    else:
        label, proba = predict_single(review_text, threshold)
        st.success(f"Prediction: {label}")
        st.info(f"Probability of being fake: {proba:.2f}")
