import streamlit as st
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from datetime import datetime

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
    
    # 2. Remove punctuation 
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
        
        # Check if the model has predict_proba attribute
        has_proba = hasattr(model, 'predict_proba')
        
        return model, vectorizer, has_proba
        
    except FileNotFoundError:
        st.error("Error: Model or Vectorizer files not found. Ensure 'model.joblib' and 'tfidf.joblib' are in the same directory.")
        return None, None, False

# Load assets only if NLTK data was successfully initialized
if 'is_nltk_ready' in locals() and is_nltk_ready:
    model, tfidf_vectorizer, has_proba = load_assets()
else:
    model, tfidf_vectorizer, has_proba = None, None, False

# --- New Feature: CSS and UI Overhaul Function ---

def apply_custom_css():
    """Applies a clean, modern look with custom colors."""
    st.markdown("""
        <style>
        /* General Setup */
        .stApp {
            background-color: #f0f2f6; /* Light gray background */
        }
        
        /* Header Styling */
        h1 {
            color: #1e88e5; /* Blue accent for the main title */
            text-align: center;
            border-bottom: 2px solid #1e88e5;
            padding-bottom: 10px;
        }
        
        /* Main Content Container */
        .stTextArea textarea {
            border: 2px solid #ccc;
            border-radius: 8px;
            padding: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            background-color: #ffffff;
        }
        
        /* Primary Button Style */
        .stButton>button {
            background-color: #1e88e5;
            color: white !important;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #1565c0; /* Darker blue on hover */
        }

        /* Result Boxes (Success/Error/Warning) */
        .stSuccess, .stError, .stWarning {
            border-radius: 10px;
            padding: 15px;
            font-size: 1.1em;
            font-weight: bold;
            margin: 10px 0;
            border-left: 5px solid;
        }
        .stSuccess { border-left-color: #4CAF50; } /* Green */
        .stError { border-left-color: #F44336; } /* Red */
        .stWarning { border-left-color: #FF9800; } /* Orange */

        /* Custom Metrics Box Styling */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
            color: #1e88e5; /* Use blue for metric values */
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem;
            color: #616161;
        }
        </style>
        """, unsafe_allow_html=True)

# --- New Feature: Auxiliary Feature Extraction ---

def extract_auxiliary_features(text):
    """
    Extracts simple, notebook-aligned features for better insight.
    """
    word_count = len(text.split())
    char_count = len(text)
    
    # Highlighting features often found in fake reviews
    all_caps_count = len(re.findall(r'\b[A-Z]{3,}\b', text))
    exclam_count = text.count('!')
    
    # Check for specific suspicious words (case-insensitive)
    suspicious_words = ['scam', 'fraud', 'never buy', 'worst experience', 'best service ever', 'amazing']
    susp_word_count = sum(text.lower().count(w) for w in suspicious_words)
    
    return {
        'Word Count': word_count,
        'Character Count': char_count,
        'All Caps Words': all_caps_count,
        'Exclamation Marks': exclam_count,
        'Suspicious Keywords': susp_word_count
    }

# --- Streamlit Application Interface ---
apply_custom_css()
st.set_page_config(page_title="Fake Review Detector", layout="wide")
st.title("🛡️ Web Hosting Review Detection System")
st.markdown("A Machine Learning model trained to classify reviews as **Genuine (OR)** or **Fake/Deceptive (CG)**.")
st.markdown("---")

if not is_nltk_ready:
    # Stop the app if NLTK is not initialized
    st.stop()

if model and tfidf_vectorizer:
    
    # --- UI LAYOUT ---
    col_input, col_settings = st.columns([3, 1])
    
    with col_input:
        st.header("1. Enter Review Text")
        review_input = st.text_area(
            "Paste the review text below:", 
            height=200, 
            key="review_area",
            placeholder="e.g., 'This service is fast and reliable, five stars all the way!' or 'Worst host ever, zero support, totally deceptive.'"
        )
        
    with col_settings:
        st.header("2. Settings")
        show_features = st.checkbox("Show Auxiliary Features", value=True, help="Display counts for length, caps, and exclamation marks.")
        show_processed = st.checkbox("Show Processed Text", value=False, help="Display the text after cleaning and stemming.")
        st.markdown(f"**Model Status:** {'✅ Ready' if model else '❌ Missing'}")
        st.markdown(f"**Proba Status:** {'✅ Available' if has_proba else '⚠️ Missing'}")

    st.markdown("---")
    
    if st.button("🚀 DETECT FAKE REVIEW", type="primary", use_container_width=True):
        if review_input:
            
            # --- Prediction Logic ---
            with st.spinner('Analyzing review...'):
                
                cleaned_input = clean_text(review_input)
                vectorized_input = tfidf_vectorizer.transform([cleaned_input])
                
                # Predict
                prediction = model.predict(vectorized_input)[0]
                is_fake = prediction.lower() == 'cg'
                
                # Predict Probability (If available)
                probability = None
                if has_proba:
                    proba_array = model.predict_proba(vectorized_input)[0]
                    # Assuming 'cg' is class 0 and 'og' is class 1 (check your model)
                    # We'll calculate the probability of the predicted class
                    class_labels = model.classes_ 
                    
                    if 'cg' in class_labels and 'og' in class_labels:
                        
                        # Find the index of the predicted class and its probability
                        pred_index = np.where(class_labels == prediction)[0][0]
                        probability = proba_array[pred_index]
                    else:
                         # Fallback if class order is ambiguous
                        probability = np.max(proba_array)

                # --- Feature Extraction ---
                features = extract_auxiliary_features(review_input)
                
            # --- DISPLAY RESULTS ---
            
            st.markdown("## 3. Analysis Report")
            
            # 1. Main Prediction Box
            status_emoji = "🚨" if is_fake else "👍"
            status_text = "FAKE/DECEPTIVE REVIEW (CG)" if is_fake else "GENUINE REVIEW (OR)"
            status_color = "red" if is_fake else "green"
            
            st.markdown(f"""
                <div style="background-color: {'#ffebee' if is_fake else '#e8f5e9'}; 
                            padding: 20px; border-radius: 10px; 
                            border: 2px solid {status_color}; text-align: center;">
                    <h2 style="color: {status_color}; margin: 0;">{status_emoji} {status_text}</h2>
                    {'<p style="color: #F44336;">**Confidence:** ' + f'{probability:.2%}' + '</p>' if probability is not None else ''}
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Confidence and Insights")
            
            # 2. Confidence/Risk Metric
            col_conf, col_risk = st.columns(2)
            
            if probability is not None:
                risk_score = (probability * 100) if is_fake else (1 - probability) * 100
                risk_level = "High" if risk_score > 75 else "Moderate" if risk_score > 50 else "Low"

                col_conf.metric(
                    "Model Confidence", 
                    f"{probability:.2%}" if is_fake else f"{probability:.2%}",
                    delta=f"P({prediction.upper()})"
                )
                col_risk.metric(
                    "Deception Risk Score", 
                    f"{risk_score:.2f}%" if is_fake else f"{100-risk_score:.2f}%",
                    delta=risk_level
                )

            # 3. Auxiliary Feature Metrics
            if show_features:
                st.markdown("### Review Characteristics (Heuristics)")
                cols_feat = st.columns(5)
                
                cols_feat[0].metric("Word Count", features['Word Count'])
                cols_feat[1].metric("Char Count", features['Character Count'])
                
                # Highlight potentially suspicious metrics
                cols_feat[2].metric("ALL CAPS Words", features['All Caps Words'], delta_color="inverse")
                cols_feat[3].metric("Exclamation Marks", features['Exclamation Marks'], delta_color="inverse")
                cols_feat[4].metric("Suspicious Keywords", features['Suspicious Keywords'], delta_color="inverse")
            
            # 4. Processed Text Display
            if show_processed:
                st.markdown("### Text Processing Details")
                st.code(cleaned_input, language='text')
                
            # 5. Recommendation
            st.markdown("### Recommendation")
            if is_fake:
                st.warning("🚨 **ACTION REQUIRED:** This review exhibits high linguistic similarity to known deceptive/fake content. Investigate the reviewer's profile and history.")
            else:
                st.info("👍 **Looks Genuine:** The language patterns align with typical truthful reviews. No immediate action required.")
                
        else:
            st.warning("👈 Please enter a review to begin the analysis.")

st.markdown("---")
st.caption(f"App powered by Streamlit | Model: Logistic Regression | Last Analysis: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
