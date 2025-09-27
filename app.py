import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import joblib

# -----------------------------
# Load model and vectorizer
# -----------------------------
@st.cache_resource
def load_model_files():
    tfidf = joblib.load("tfidf.joblib")
    model = joblib.load("model.joblib")
    return tfidf, model

tfidf, model = load_model_files()

# -----------------------------
# NLTK setup (English only)
# -----------------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# -----------------------------
# Text preprocessing function
# -----------------------------
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # remove non-letters
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# -----------------------------
# Prediction function
# -----------------------------
def predict_single(review_text, threshold=0.5):
    cleaned = clean_text(review_text)
    vector = tfidf.transform([cleaned])
    proba = model.predict_proba(vector)[0][1]
    label = "Fake" if proba >= threshold else "Genuine"
    return label, proba

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Fake Review Detection System")
review_text = st.text_area("Enter the review text:")

threshold = st.slider("Threshold for classifying as Fake", 0.0, 1.0, 0.5)

if st.button("Predict"):
    if review_text.strip() == "":
        st.warning("Please enter a review!")
    else:
        label, proba = predict_single(review_text, threshold)
        st.write(f"**Prediction:** {label}")
        st.write(f"**Probability of being Fake:** {proba:.2f}")
