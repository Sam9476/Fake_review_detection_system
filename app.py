import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# -----------------------------
# Download NLTK resources
# -----------------------------
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass

download_nltk_data()

# -----------------------------
# Load model and TF-IDF
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
# Text preprocessing (same as notebook)
# -----------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Join back to string
    return ' '.join(tokens)

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
