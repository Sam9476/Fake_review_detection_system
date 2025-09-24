import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import joblib
import os
import tempfile

# -----------------------------import streamlit as st
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
import joblib

# -----------------------------
# NLTK setup
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# -----------------------------
# Load model
# -----------------------------
@st.cache_data
def load_model(path="fake_review_model.joblib"):
    return joblib.load(path)

model = load_model()

# -----------------------------
# Text preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # remove punctuation/numbers
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# -----------------------------
# Prediction function
# -----------------------------
def predict_single(review, threshold=0.5):
    cleaned = clean_text(review)
    # Assuming model expects a list of texts
    proba = model.predict_proba([cleaned])[0][1]  # probability of being fake
    label = "FAKE" if proba >= threshold else "REAL"
    return label, proba

# -----------------------------
# Streamlit app
# -----------------------------
st.title("Fake Review Detection System")
st.write("Enter a review and check if it is REAL or FAKE:")

review_text = st.text_area("Review Text", "")

threshold = st.slider("Probability threshold", 0.0, 1.0, 0.5)

if st.button("Predict"):
    if review_text.strip() == "":
        st.warning("Please enter a review text!")
    else:
        label, proba = predict_single(review_text, threshold)
        st.success(f"Prediction: {label}")
        st.info(f"Probability of being FAKE: {proba:.2f}")

# NLTK setup for deployment
# -----------------------------
# Use a temporary directory for NLTK data
nltk_data_dir = os.path.join(tempfile.gettempdir(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download resources into temp directory if not already present
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_dir)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_dir)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# -----------------------------
# Load model and TF-IDF safely
# -----------------------------
try:
    model = joblib.load("model.pkl")
    tfidf = joblib.load("tfidf.pkl")
except Exception as e:
    st.error(f"Error loading model or TF-IDF: {e}")
    st.stop()

# -----------------------------
# Text preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# -----------------------------
# Prediction function
# -----------------------------
def predict_single(review, threshold=0.5):
    cleaned = clean_text(review)
    features = tfidf.transform([cleaned])
    proba = model.predict_proba(features)[0][1]  # probability for "fake"
    label = "FAKE" if proba >= threshold else "REAL"
    return label, proba

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Fake Review Detection System")

review_text = st.text_area("Enter a review for analysis:")

threshold = st.slider("Fake Probability Threshold", min_value=0.0, max_value=1.0, value=0.5)

if st.button("Predict"):
    if not review_text.strip():
        st.warning("Please enter a review!")
    else:
        label, proba = predict_single(review_text, threshold)
        st.success(f"Prediction: {label}")
        st.info(f"Probability of being fake: {proba:.2f}")
