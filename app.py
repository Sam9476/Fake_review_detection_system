import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
from langdetect import detect, LangDetectException
import os

# --- NLTK Configuration (Uses Local Folder) ---
# Set the NLTK data path to the local directory
nltk_data_path = 'nltk_data'
nltk.data.path.append(nltk_data_path)

# Initialize stemmer and stop words (English specific)
try:
    # Stop words are loaded from the local folder
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
except LookupError:
    st.error("NLTK data not found locally. Ensure the 'nltk_data' folder is uploaded.")
    stop_words = set() # Set empty to prevent immediate crash


def clean_text(text):
    """Applies the exact cleaning and stemming logic used during training."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words and stem
    cleaned_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

# --- Load Assets (Cached for Efficiency) ---
@st.cache_resource
def load_assets():
    """Loads the trained model and TF-IDF vectorizer."""
    try:
        model = joblib.load('model.joblib')
        vectorizer = joblib.load('tfidf.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: Model or Vectorizer files not found. Ensure 'model.joblib' and 'tfidf.joblib' are in the same directory.")
        return None, None

model, tfidf_vectorizer = load_assets()

# --- Streamlit Application Interface ---
st.set_page_config(page_title="Fake Review Detector", layout="wide")
st.title("🛡️ Web Hosting Fake Review Detection System")
st.markdown("A Machine Learning model trained to classify English reviews as Genuine (OR) or Fake (CG).")
st.markdown("---")

if model and tfidf_vectorizer:
    
    st.header("Analyze a Review")
    
    review_input = st.text_area(
        "Paste the review text below (English only):", 
        height=200, 
        placeholder="e.g., 'This service is fast and reliable, five stars all the way!' or 'Worst host ever, zero support, totally deceptive.'"
    )

    if st.button("Detect Spam/Fake Review", type="primary"):
        if review_input:
            
            # --- Language Check ---
            try:
                language = detect(review_input)
                if language != 'en':
                    st.warning(f"⚠️ **Language Warning:** Detected language is '{language}'. The model is trained ONLY on English and predictions may be unreliable.")
            except LangDetectException:
                st.warning("⚠️ Could not reliably detect the language of the review. Proceeding.")
            
            # --- Prediction Logic ---
            with st.spinner('Analyzing review...'):
                
                # Preprocess input using the exact training function
                cleaned_input = clean_text(review_input)
                
                # Vectorize (Transform using the fitted vectorizer)
                vectorized_input = tfidf_vectorizer.transform([cleaned_input])
                
                # Predict (1=Fake/CG, 0=Genuine/OR)
                prediction = model.predict(vectorized_input)[0]
                
                # Get probability for confidence score
                try:
                    probability = model.predict_proba(vectorized_input)[0][prediction]
                except:
                    probability = None
                    
                st.markdown("### Analysis Result:")

                if prediction == 1:
                    st.error(f"❌ **PREDICTION: FAKE/DECEPTIVE REVIEW** (CG)")
                else:
                    st.success(f"✅ **PREDICTION: GENUINE REVIEW** (OR)")
                
                if probability is not None:
                    st.metric(
                        label="Confidence Score", 
                        value=f"{probability * 100:.2f}%"
                    )

        else:
            st.warning("👈 Please enter a review to begin the analysis.")

st.markdown("---")
st.caption("Model: Logistic Regression | Features: TF-IDF with Porter Stemming")
