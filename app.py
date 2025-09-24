import streamlit as st
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# -----------------------------
# NLTK setup (download only if missing)
# -----------------------------
nltk_data_path = './nltk_data'
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# -----------------------------
# Load model and vectorizer
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model_files():
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return tfidf, model

tfidf, model = load_model_files()

# -----------------------------
# Text preprocessing
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation/numbers
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# -----------------------------
# Prediction
# -----------------------------
def predict_single(text, threshold=0.5):
    cleaned = clean_text(text)
    X = tfidf.transform([cleaned])
    proba = model.predict_proba(X)[0][1]  # probability of being fake
    label = "Fake" if proba >= threshold else "Genuine"
    return label, proba

# -----------------------------
# Streamlit app
# -----------------------------
st.title("Fake Review Detection System")
review_text = st.text_area("Enter a review to check:")

threshold = st.slider("Set probability threshold", 0.0, 1.0, 0.5, 0.01)

if st.button("Predict"):
    if review_text.strip() == "":
        st.warning("Please enter a review first!")
    else:
        label, proba = predict_single(review_text, threshold)
        st.write(f"Prediction: **{label}**")
        st.write(f"Probability of being fake: **{proba:.2f}**")
