import streamlit as st
import joblib
import re
import numpy as np

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
# Text preprocessing function
# -----------------------------
def preprocess_text(text):
    """Lowercase, remove punctuation, extra whitespace"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # remove punctuation
    text = re.sub(r'\d+', ' ', text)      # remove numbers
    text = ' '.join(text.split())         # remove extra spaces
    return text

# -----------------------------
# Prediction function
# -----------------------------
def predict_review(text):
    cleaned_text = preprocess_text(text)
    vect_text = tfidf.transform([cleaned_text])
    pred = model.predict(vect_text)[0]
    proba = model.predict_proba(vect_text).max()
    return pred, proba

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fake Review Detection", layout="wide")
st.title("🛡️ Fake Review Detection System")

review_text = st.text_area("Enter the review text:", height=150)

threshold = st.slider(
    "Prediction threshold (confidence level to classify as fake)", 0.0, 1.0, 0.5, 0.05
)

if st.button("Predict"):
    if review_text.strip() == "":
        st.warning("Please enter a review text to analyze.")
    else:
        label, proba = predict_review(review_text)
        is_fake = label == 1  # assuming 1 = fake, 0 = genuine

        st.markdown("---")
        st.subheader("Prediction Result")

        if is_fake:
            st.error(f"🚨 Fake Review Detected! (Confidence: {proba:.2%})")
        else:
            st.success(f"✅ Genuine Review (Confidence: {proba:.2%})")

        st.markdown("**Processed Text:**")
        st.text(preprocess_text(review_text))
