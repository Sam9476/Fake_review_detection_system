import streamlit as st
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- NLTK Configuration (Uses Local Folder for speed) ---

# Set the path to the uploaded 'nltk_data' folder
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

# Check and Initialize NLTK components
try:
    # Explicitly find 'stopwords' resource
    nltk.data.find('corpora/stopwords')
    
    # Initialize stemmer and stop words
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    is_nltk_ready = True

except LookupError:
    st.error(
        "NLTK DATA ERROR: Failed to locate 'stopwords' in the 'nltk_data' folder. "
        "Prediction is disabled. Please ensure the folder is uploaded and structured correctly."
    )
    stemmer = None
    stop_words = set()
    is_nltk_ready = False
    
# --- Preprocessing Function (Aligned with Notebook) ---

def clean_text(text):
    """
    Applies the exact cleaning and stemming logic from the notebook,
    replacing word_tokenize with a safe alternative (.split()).
    """
    if not is_nltk_ready:
        return text 

    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove punctuation (From Notebook: text = re.sub(r'[^\w\s]', '', text))
    # We replace with a space to ensure separate words remain separate
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 3. Tokenize (REPLACED word_tokenize with robust .split())
    tokens = text.split() 
    
    # 4. Remove stop words and stem
    cleaned_tokens = [
        stemmer.stem(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 1
    ]
    
    return ' '.join(cleaned_tokens)

# --- Load Assets (Cached for Efficiency) ---
@st.cache_resource
def load_assets():
    """Loads the trained model and TF-IDF vectorizer."""
    try:
        # Assumes model is 'model.joblib' and vectorizer is 'tfidf.joblib'
        model = joblib.load('model.joblib')
        vectorizer = joblib.load('tfidf.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: Model or Vectorizer files not found. Ensure 'model.joblib' and 'tfidf.joblib' are in the same directory.")
        return None, None

# Load assets only if NLTK data was successfully initialized
if 'is_nltk_ready' in locals() and is_nltk_ready:
    model, tfidf_vectorizer = load_assets()
else:
    model, tfidf_vectorizer = None, None

# --- Streamlit Application Interface ---
st.set_page_config(page_title="Fake Review Detector", layout="wide")
st.title("🛡️ Web Hosting Review Detection System")
st.markdown("A Machine Learning model trained to classify reviews as **Genuine (OR)** or **Fake/Deceptive (CG)**.")
st.markdown("---")

if model and tfidf_vectorizer:
    
    st.header("Analyze a Review")
    
    review_input = st.text_area(
        "Paste the review text below:", 
        height=200, 
        placeholder="e.g., 'This service is fast and reliable, five stars all the way!' or 'Worst host ever, zero support, totally deceptive.'"
    )
    
    st.caption("⚠️ Note: Model trained on English data. Results may be unreliable for other languages.")

    if st.button("Detect Spam/Fake Review", type="primary"):
        if review_input:
            
            # --- Prediction Logic (Classification Only) ---
            with st.spinner('Analyzing review...'):
                
                cleaned_input = clean_text(review_input)
                vectorized_input = tfidf_vectorizer.transform([cleaned_input])
                
                # Predict (The prediction is the actual label: 'cg' or 'og')
                prediction = model.predict(vectorized_input)[0]
                
                # Determine display variables based on the predicted label
                is_fake = prediction.lower() == 'cg'
                status_emoji = "❌" if is_fake else "✅"
                status_text = "FAKE/DECEPTIVE REVIEW (CG)" if is_fake else "GENUINE REVIEW (OR)"
                
                st.markdown("### Analysis Result:")
                
                # --- Display Prediction without Percentage ---
                if is_fake:
                    st.error(f"{status_emoji} **PREDICTION: {status_text}**")
                else:
                    st.success(f"{status_emoji} **PREDICTION: {status_text}**")
        else:
            st.warning("👈 Please enter a review to begin the analysis.")

st.markdown("---")
st.caption("Model: Logistic Regression | Features: TF-IDF with Porter Stemming")
