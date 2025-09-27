import streamlit as st
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- NLTK Configuration (Uses Local Folder) ---

# 1. Set the path to the uploaded 'nltk_data' folder
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

# 2. Check and Initialize NLTK components
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
        "Please ensure the folder contains the correct structure (e.g., 'nltk_data/corpora/stopwords')."
    )
    # Disable prediction logic
    stemmer = None
    stop_words = set()
    is_nltk_ready = False
    
# --- Preprocessing Function (Manual Tokenization) ---

def clean_text(text):
    """
    Applies cleaning, stemming, and uses manual tokenization 
    to avoid the problematic nltk.word_tokenize (punkt dependency).
    """
    if not is_nltk_ready:
        return text 

    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Aggressive Cleaning: Remove anything that is NOT a letter or space.
    text = re.sub(r'[^a-z\s]', '', text) 
    
    # 3. Tokenize using split() 
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
st.markdown("A Machine Learning model trained to classify reviews as Genuine (OR) or Fake/Deceptive (CG).")
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
            
            # --- Prediction Logic ---
            with st.spinner('Analyzing review...'):
                
                cleaned_input = clean_text(review_input)
                
                # Vectorize 
                vectorized_input = tfidf_vectorizer.transform([cleaned_input])
                
                # Predict (1=Fake/CG, 0=Genuine/OR)
                prediction = model.predict(vectorized_input)[0]
                
                # Get probabilities for both classes
                try:
                    probabilities = model.predict_proba(vectorized_input)[0]
                    confidence_score = probabilities[prediction]
                    
                    # Prepare display variables
                    status_emoji = "❌" if prediction == 1 else "✅"
                    status_text = "FAKE/DECEPTIVE REVIEW (CG)" if prediction == 1 else "GENUINE REVIEW (OR)"
                    
                except:
                    confidence_score = None
                    status_emoji = ""
                    status_text = "Prediction Error"
                    
                st.markdown("### Analysis Result:")
                
                # --- FINAL DISPLAY FIX: Use st.metric for guaranteed percentage display ---
                
                if confidence_score is not None:
                    # Display the main prediction and confidence using st.metric
                    st.metric(
                        label=f"{status_emoji} **{status_text}**", 
                        value=f"{confidence_score * 100:.2f}%", 
                        delta_color="off" # Turn off delta color for neutral display
                    )
                    
                    # Display the counter-probability for full transparency
                    opposite_class = 1 - prediction
                    opposite_score = probabilities[opposite_class]
                    st.caption(f"Confidence for the {'Genuine (OR)' if opposite_class == 0 else 'Fake (CG)'} class: **{opposite_score * 100:.2f}%**")
                else:
                    st.warning("Could not calculate confidence score.")

        else:
            st.warning("👈 Please enter a review to begin the analysis.")

st.markdown("---")
st.caption("Model: Logistic Regression | Features: TF-IDF with Porter Stemming")
