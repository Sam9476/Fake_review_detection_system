import streamlit as st
import joblib
import re
import os
import nltk

# --- NLTK Configuration (CRITICAL FIX FOR LookupError) ---

# 1. Point NLTK to the uploaded 'nltk_data' folder
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

# 2. Explicitly load the required NLTK data resources before importing components.
# This ensures NLTK finds the models using the custom path.
try:
    # Attempt to find the necessary files. If they fail, the LookupError is raised here.
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    
    # Now import the components that rely on the found data
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    
    # Initialize stemmer and stop words
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    
    # Flag to indicate successful initialization
    is_nltk_ready = True

except LookupError:
    # Handle the case where the NLTK data is not found
    st.error("NLTK Data Error: Could not find 'punkt' or 'stopwords' in the 'nltk_data' folder. "
             "Please ensure the folder is complete and structured correctly (e.g., nltk_data/tokenizers/punkt).")
    
    # Disable prediction logic
    stemmer = None
    stop_words = set()
    is_nltk_ready = False
    
# --- Preprocessing Function ---

def clean_text(text):
    """Applies the exact cleaning and stemming logic used during training."""
    if not is_nltk_ready:
        return text # Return original text if cleaning failed

    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenize (This line caused the error, now protected by explicit find())
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

# Load assets only if NLTK data was successfully initialized
if is_nltk_ready:
    model, tfidf_vectorizer = load_assets()
else:
    model, tfidf_vectorizer = None, None

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
