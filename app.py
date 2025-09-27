# app.py

import streamlit as st
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# -----------------------------
# NLTK setup
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# -----------------------------
# Load models
# -----------------------------
@st.cache_data
def load_model_files():
    tfidf = joblib.load("tfidf.joblib")
    model = joblib.load("model.joblib")
    return tfidf, model

tfidf, model = load_model_files()

# -----------------------------
# Text preprocessing
# -----------------------------
def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and stem
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# -----------------------------
# Prediction
# -----------------------------
def predict_single(review, threshold=0.5):
    cleaned = clean_text(review)
    vect = tfidf.transform([cleaned])
    proba = model.predict_proba(vect)[0][1]
    label = "Fake" if proba >= threshold else "Genuine"
    return label, proba

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Fake Review Detection System")

review_text = st.text_area("Enter a review for analysis:")

threshold = st.slider("Fake Review Probability Threshold:", 0.0, 1.0, 0.5, 0.01)

if st.button("Predict"):
    if review_text.strip() == "":
        st.warning("Please enter a review.")
    else:
        label, proba = predict_single(review_text, threshold)
        st.write(f"Prediction: **{label}**")
        st.write(f"Probability of being fake: **{proba:.2f}**")
