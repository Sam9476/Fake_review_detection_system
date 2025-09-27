import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os

# -----------------------------
# Set NLTK data path to local folder
# -----------------------------
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))

# -----------------------------
# Initialize stemmer and stop words
# -----------------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# -----------------------------
# Load model and TF-IDF vectorizer
# -----------------------------
@st.cache_resource
def load_model_files():
    try:
        tfidf = joblib.load("tfidf.joblib")
        model = joblib.load("model.joblib")
        return tfidf, model
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None

tfidf, model = load_model_files()

if tfidf is None or model is None:
    st.stop()

# -----------------------------
# Preprocessing function
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    tokens = word_tokenize(text)
    cleaned_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

# -----------------------------
# Prediction function
# -----------------------------
def predict_single(review):
    cleaned = clean_text(review)
    vect = tfidf.transform([cleaned])
    prediction = model.predict(vect)[0]
    proba = model.predict_proba(vect).max()
    return prediction, proba

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fake Review Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ Fake Review Detection System")
st.markdown("Enter a review below to check if it is genuine or fake.")

review_text = st.text_area("Enter review text here:", height=150)

threshold = st.slider("Prediction confidence threshold", 0.0, 1.0, 0.5)

if st.button("Analyze Review"):
    if review_text.strip() != "":
        label, proba = predict_single(review_text)
        is_fake = label == 'deceptive' and proba >= threshold
        
        st.markdown("---")
        st.subheader("Prediction Results")
        st.write(f"**Label:** {label.capitalize()}")
        st.write(f"**Confidence:** {proba:.2%}")
        
        if is_fake:
            st.error("🚨 This review is likely **FAKE**.")
        else:
            st.success("✅ This review is likely **GENUINE**.")
    else:
        st.warning("Please enter some review text to analyze!")
