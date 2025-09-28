import streamlit as st
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer # Using RegexpTokenizer is the closest robust alternative

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
    
    # Initialize tokenizer as close to word_tokenize as possible without punkt
    # Note: If your model was trained purely on word_tokenize after simple punctuation removal,
    # the RegexpTokenizer (or even a simple .split() after aggressive cleaning) is required here.
    # We will use .split() for maximum compatibility with the aggressive cleaning from your notebook.
    is_nltk_ready = True

except LookupError:
    st.error(
        "NLTK DATA ERROR: Failed to locate 'stopwords' in the 'nltk_data' folder. "
        "Prediction is disabled. Please ensure the folder is uploaded and structured correctly."
    )
    stemmer = None
    stop_words = set()
    is_nltk_ready = False
    
# --- Preprocessing Function (Manual Tokenization to avoid Punkt error) ---

def clean_text(text):
    """
    Applies the exact cleaning and stemming logic from the notebook,
    replacing word_tokenize with a safe alternative.
    """
    if not is_nltk_ready:
        return text 

    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Remove punctuation (THIS STEP FROM YOUR NOTEBOOK IS CRITICAL)
    # The regex r'[^\w\s]' removes all non-word characters (including punctuation)
    text = re.sub(r'[^\w\s]', ' ', text) # Replace with space to prevent word joining
    
    # 3. Tokenize (REPLACED word_tokenize with robust .split())
    # Your original code: tokens = word_tokenize(text)
    # Replacement:
    tokens = text.split() 
    
    # 4. Remove stop words and stem
    cleaned_tokens = [
        stemmer.stem(word) 
        for word in tokens 
        # Added len(word) > 1 check to remove single characters left after cleaning
        if word not in stop_words and len(word) > 1
    ]
    
    return ' '.join(cleaned_tokens)

# --- Load Assets (Cached for Efficiency) ---
@st.cache_resource
def load_assets():
    """Loads the trained model and TF-IDF vectorizer."""
    try:
        # Assumes model is 'model.joblib' and vectorizer is 'tfidf.joblib'
        # NOTE: Your notebook uses LogisticRegression() and TfidfVectorizer(). 
        # Ensure you saved the fitted vectorizer and the trained model using joblib.dump().
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
            
            # --- Prediction Logic ---
            with st.spinner('Analyzing review...'):
                
                cleaned_input = clean_text(review_input)
                
                # Transform the cleaned text using the fitted vectorizer
                vectorized_input = tfidf_vectorizer.transform([cleaned_input])
                
                # Predict (1=Fake/CG, 0=Genuine/OR)
                prediction = model.predict(vectorized_input)[0]
                
                # Get probabilities for both classes
                try:
                    probabilities = model.predict_proba(vectorized_input)[0]
                    confidence_score = probabilities[prediction]
                    
                    # Prepare display variables
                    is_fake = prediction == 1
                    status_emoji = "❌" if is_fake else "✅"
                    status_text = "FAKE/DECEPTIVE REVIEW (CG)" if is_fake else "GENUINE REVIEW (OR)"
                    
                except:
                    confidence_score = None
                    status_emoji = ""
                    status_text = "Prediction Error"
                    
                st.markdown("### Analysis Result:")
                
                # --- Display Prediction and Percentage using st.metric (Fix for display issue) ---
                
                if confidence_score is not None:
                    # Display the main prediction and confidence
                    st.metric(
                        # Label includes the emoji and status text
                        label=f"{status_emoji} **{status_text}**", 
                        # Value shows the percentage
                        value=f"{confidence_score * 100:.2f}%", 
                        delta_color="off" 
                    )
                    
                    # Display the counter-probability for full transparency
                    opposite_class = 1 - prediction
                    opposite_score = probabilities[opposite_class]
                    
                    opposite_label = 'Fake (CG)' if opposite_class == 1 else 'Genuine (OR)'
                    st.caption(f"Confidence for the opposing class ({opposite_label}): **{opposite_score * 100:.2f}%**")
                else:
                    st.warning("Could not calculate confidence score. Check your model's prediction output.")

        else:
            st.warning("👈 Please enter a review to begin the analysis.")

st.markdown("---")
st.caption("Model: Logistic Regression | Features: TF-IDF with Porter Stemming")
