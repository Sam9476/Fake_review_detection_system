import streamlit as st
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# -----------------------------
# Download NLTK resources safely
# -----------------------------
@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    except:
        pass

# -----------------------------
# Load model safely
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.joblib')  # your saved pipeline
        return model
    except FileNotFoundError:
        st.error("Model file not found! Please check model.joblib in the repo.")
        return None

# -----------------------------
# Text preprocessing function
# -----------------------------
def preprocess_text(text):
    text = text.lower()
    
    # Remove URLs, emails, and numbers
    text = re.sub(r'http\S+|www\S+|https\S+', ' URL ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = re.sub(r'\d+', ' NUMBER ', text)
    
    # Remove punctuation except sentence structure
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stopwords safely
    try:
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        text = ' '.join([t for t in tokens if t not in stop_words])
    except:
        pass
    
    return text

# -----------------------------
# Prediction function
# -----------------------------
def predict_review(text, model):
    cleaned_text = preprocess_text(text)
    pred = model.predict([cleaned_text])[0]
    prob = model.predict_proba([cleaned_text]).max()
    return pred, prob

# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(
        page_title="Fake Review Detection",
        page_icon="🛡️",
        layout="wide"
    )
    
    st.title("🛡️ Fake Review Detection System")
    st.markdown("---")
    
    # Download NLTK data
    download_nltk_data()
    
    # Load model
    model = load_model()
    if model is None:
        st.stop()
    
    # Text input
    review_text = st.text_area(
        "Enter your review text here:",
        height=150,
        placeholder="Paste or type your review..."
    )
    
    if st.button("Analyze Review"):
        if not review_text.strip():
            st.error("Please enter a review to analyze.")
        else:
            with st.spinner("Analyzing..."):
                pred, prob = predict_review(review_text, model)
                is_fake = pred == 'deceptive' or pred == 'fake'  # adapt according to your labels
                
                # Display result
                st.markdown("---")
                st.subheader("🔍 Prediction Result")
                if is_fake:
                    st.error(f"🚨 Fake Review Detected! Confidence: {prob:.1%}")
                else:
                    st.success(f"✅ Legitimate Review. Confidence: {prob:.1%}")
                
                # Optionally show cleaned text
                with st.expander("Show preprocessed text"):
                    st.write(preprocess_text(review_text))

if __name__ == "__main__":
    main()
