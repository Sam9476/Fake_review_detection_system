import streamlit as st
import joblib
import re
import os
import nltk

# ----------------------------
# 1. NLTK CONFIGURATION
# ----------------------------

# Point NLTK to the uploaded 'nltk_data' folder
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

# Import NLTK components safely
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# ----------------------------
# 2. TEXT PREPROCESSING
# ----------------------------
def clean_text(text):
    """Cleans and stems text exactly like during training."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    tokens = word_tokenize(text)
    cleaned_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

# ----------------------------
# 3. LOAD MODEL AND VECTORIZER
# ----------------------------
@st.cache_resource
def load_assets():
    """Load the trained model and TF-IDF vectorizer."""
    try:
        model = joblib.load('model.joblib')
        vectorizer = joblib.load('tfidf.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: Model or Vectorizer files not found. Ensure 'model.joblib' and 'tfidf.joblib' are uploaded.")
        return None, None

model, tfidf_vectorizer = load_assets()

# ----------------------------
# 4. STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Fake Review Detector", layout="wide")
st.title("🛡️ Web Hosting Fake Review Detection System")
st.markdown("Classify reviews as Genuine (OR) or Fake/Deceptive (CG).")
st.markdown("---")

if model and tfidf_vectorizer:
    st.header("Analyze a Review")
    
    review_input = st.text_area(
        "Paste the review text here (English reviews only):", 
        height=200, 
        placeholder="e.g., 'This service is fast and reliable!' or 'Worst host ever, totally deceptive.'"
    )

    if st.button("Detect Fake/Genuine Review"):
        if review_input.strip():
            with st.spinner("Analyzing review..."):
                # Preprocess
                cleaned_input = clean_text(review_input)
                
                # Vectorize
                vectorized_input = tfidf_vectorizer.transform([cleaned_input])
                
                # Predict
                prediction = model.predict(vectorized_input)[0]
                
                # Probability / confidence
                try:
                    probability = model.predict_proba(vectorized_input)[0][prediction]
                except:
                    probability = None
                
                # Display result
                st.markdown("### Analysis Result:")
                if prediction == 1:
                    st.error("❌ PREDICTION: FAKE/DECEPTIVE REVIEW (CG)")
                else:
                    st.success("✅ PREDICTION: GENUINE REVIEW (OR)")
                
                if probability is not None:
                    st.metric(label="Confidence Score", value=f"{probability * 100:.2f}%")
        else:
            st.warning("👈 Please enter a review to begin the analysis.")

st.markdown("---")
st.caption("Model: Logistic Regression | Features: TF-IDF with Porter Stemming")
