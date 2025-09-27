import streamlit as st
import joblib
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- Configuration for Local NLTK Data (Faster Startup) ---
# Ensure your 'nltk_data' folder is uploaded to the root of your repository.
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

# --- NLTK and Model Initialization ---
try:
    # Explicitly find 'stopwords' resource to check data availability
    nltk.data.find('corpora/stopwords')
    
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    is_nltk_ready = True
    
except LookupError:
    st.error(
        "NLTK DATA ERROR: Failed to locate 'stopwords'. Please ensure the 'nltk_data' folder is uploaded correctly."
    )
    is_nltk_ready = False
    stemmer = None
    stop_words = set()

# Load the trained model and vectorizer
@st.cache_resource
def load_assets():
    """Loads the trained model and TF-IDF vectorizer (renamed from your reference)."""
    if not is_nltk_ready:
        return None, None
    try:
        # Assuming model files are named model.joblib and tfidf.joblib as per previous context
        model = joblib.load('model.joblib')
        vectorizer = joblib.load('tfidf.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found! Ensure 'model.joblib' and 'tfidf.joblib' are in the same directory.")
        return None, None

# --- Preprocessing Function (Stable Logic from Previous Fixes) ---
def preprocess_text(text):
    """
    Applies cleaning and stemming using manual tokenization 
    to ensure deployment stability and match training features.
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

def main():
    # Set page config
    st.set_page_config(
        page_title="Fake Review Detection System",
        page_icon="🛡️",
        layout="wide"
    )
    
    # --- Load Assets ---
    model, tfidf_vectorizer = load_assets()
    
    # --- Check for load errors ---
    if not model or not tfidf_vectorizer:
        st.error("⚠️ Application halted due to missing model assets or NLTK data.")
        return

    # --- Header ---
    st.title("🛡️ Web Hosting Fake Review Detection System")
    st.markdown("A Machine Learning model trained to classify reviews as **Genuine (OR)** or **Fake/Deceptive (CG)**.")
    st.markdown("---")
    
    # --- Main interface ---
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📝 Enter Review Text to Analyze")
        
        user_text = st.text_area(
            "Paste the review text below:",
            height=150,
            placeholder="e.g., 'This host is incredible! Use my referral code NOW!' or 'The service was fine, no major issues.'"
        )
    
    with col2:
        st.subheader("⚙️ Settings")
        
        # NOTE: Rating is removed as it's not present in your preprocess_text
        st.caption("Model Type: Logistic Regression")
        st.caption("Features: TF-IDF with Porter Stemming")

        # Provide example button for quick testing
        if st.button("Load Sample Fake Review", key="sample_fake"):
            user_text = "THIS IS THE WORST HOST EVER! Complete scam. EVERYTHING IS BAD. I LOST ALL MY DATA. Don't use them, they are totally deceptive and criminal. Zero stars. Stay away. TERRIBLE!"
        if st.button("Load Sample Genuine Review", key="sample_genuine"):
            user_text = "I've been using this host for a year. Uptime is fantastic, only one 10-minute outage. Support is slow on weekends, but pricing is unbeatable. Highly recommend."
    
    # Analysis button and results
    if st.button("🔍 Classify Review", type="primary", use_container_width=True):
        if user_text.strip():
            with st.spinner("Analyzing review..."):
                
                # Preprocess text
                cleaned_text = preprocess_text(user_text)
                
                # Vectorize and Predict
                vectorized_input = tfidf_vectorizer.transform([cleaned_text])
                prediction = model.predict(vectorized_input)[0] # 1=CG (Fake), 0=OR (Genuine)
                probabilities = model.predict_proba(vectorized_input)[0]
                
                # Determine result based on prediction (0 or 1)
                is_fake = prediction == 1
                confidence_score = probabilities[prediction]
                
                # Prepare display variables
                status_emoji = "❌" if is_fake else "✅"
                status_label = "FAKE/DECEPTIVE REVIEW (CG)" if is_fake else "GENUINE REVIEW (OR)"
                
                st.markdown("---")
                st.subheader("📊 Analysis Result")
                
                # --- FINAL DISPLAY: Use st.metric for guaranteed percentage display ---
                
                # Display the main prediction and confidence
                st.metric(
                    label=f"{status_emoji} **{status_label}**", 
                    value=f"{confidence_score * 100:.2f}%", 
                    delta_color="off"
                )
                
                # Display the counter-probability for full transparency
                opposite_class = 1 - prediction
                opposite_score = probabilities[opposite_class]
                
                st.caption(
                    f"Confidence for the {'Genuine (OR)' if opposite_class == 0 else 'Fake (CG)'} class: **{opposite_score * 100:.2f}%**"
                )
                
                # Recommendations
                st.subheader("💡 Recommendations")
                if is_fake:
                    st.warning("⚠️ **Warning**: This review exhibits characteristics of being deceptive. Exercise caution.")
                else:
                    st.success("✅ **Legitimate**: This review appears to be genuine based on its features.")
        else:
            st.error("Please enter some text to analyze!")

    st.markdown("---")
    st.caption("Final Note: Classification accuracy depends entirely on how closely the current preprocessing matches the model's original training preprocessing.")

if __name__ == "__main__":
    main()
