import streamlit as st
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# -----------------------------
# NLTK setup
# -----------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# -----------------------------
# Load model and vectorizer
# -----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# -----------------------------
# Text preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    tokens = word_tokenize(text)  # tokenize
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]  # remove stopwords + stem
    return ' '.join(tokens)

# -----------------------------
# Prediction
# -----------------------------
def predict_single(review, threshold=0.5):
    cleaned = clean_text(review)
    features = tfidf_vectorizer.transform([cleaned])
    proba = model.predict_proba(features)[0][1]  # probability of being FAKE
    label = "FAKE" if proba >= threshold else "REAL"
    return label, proba

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Fake Review Detection System")
st.write("Enter a review below to check if it is REAL or FAKE:")

review_text = st.text_area("Your Review:")

threshold = st.slider("FAKE Probability Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if st.button("Predict"):
    if review_text.strip() == "":
        st.warning("Please enter a review first!")
    else:
        label, proba = predict_single(review_text, threshold)
        st.success(f"Prediction: {label}")
        st.info(f"Probability of being FAKE: {proba:.2f}")
