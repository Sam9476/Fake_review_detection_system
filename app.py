import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# -----------------------------
# NLTK setup
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# -----------------------------
# Helper functions
# -----------------------------
def clean_text(text):
    """
    Clean and preprocess text:
    - Lowercase
    - Remove non-alphabetic chars
    - Tokenize
    - Remove stopwords
    - Apply stemming
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@st.cache_data
def load_model_files():
    """
    Load tfidf vectorizer and ML model from pickle files.
    """
    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return tfidf, model

def predict_single(review, threshold=0.5):
    """
    Predict whether a single review is fake or real.
    Returns label and probability.
    """
    cleaned = clean_text(review)
    vector = tfidf.transform([cleaned])
    proba = model.predict_proba(vector)[0][1]  # probability of fake review
    label = "Fake" if proba >= threshold else "Real"
    return label, proba

# -----------------------------
# Streamlit app
# -----------------------------
st.set_page_config(page_title="Fake Review Detection", layout="centered")
st.title("Fake Review Detection System")
st.write("Enter a review below to check if it is real or fake:")

# Load model and vectorizer
tfidf, model = load_model_files()

review_text = st.text_area("Your Review:", height=150)
threshold = st.slider("Fake Probability Threshold", 0.0, 1.0, 0.5, 0.01)

if st.button("Predict"):
    if review_text.strip() == "":
        st.warning("Please enter a review text!")
    else:
        label, proba = predict_single(review_text, threshold)
        st.success(f"Prediction: **{label}**")
        st.info(f"Fake Review Probability: **{proba:.2f}**")
