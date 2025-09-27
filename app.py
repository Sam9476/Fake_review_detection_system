import streamlit as st
import joblib
import os
import re
import nltk

# ------------------ NLTK SETUP ------------------
# Use uploaded nltk_data folder
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    # Initialize stemmer and stopwords
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    is_nltk_ready = True
except LookupError:
    st.error("NLTK data not found. Ensure nltk_data folder is uploaded with tokenizers/punkt and corpora/stopwords.")
    stemmer = None
    stop_words = set()
    is_nltk_ready = False

# ------------------ TEXT CLEANING FUNCTION ------------------
def clean_text(text):
    """Preprocessing identical to notebook: lowercase, remove punctuation, tokenize, remove stopwords, stem."""
    if not is_nltk_ready:
        return text
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    cleaned = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned)

# ------------------ LOAD MODEL AND TF-IDF ------------------
@st.cache_resource
def load_assets():
    """Load trained model and TF-IDF vectorizer."""
    try:
        model = joblib.load('model.joblib')          # Your LogisticRegression model
        vectorizer = joblib.load('tfidf.joblib')    # TF-IDF vectorizer from notebook
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model or vectorizer files not found. Ensure 'model.joblib' and 'tfidf.joblib' exist in repo.")
        return None, None

if is_nltk_ready:
    model, tfidf_vectorizer = load_assets()
else:
    model, tfidf_vectorizer = None, None

# ------------------ STREAMLIT APP ------------------
st.set_page_config(page_title="Fake Review Detector", layout="wide")
st.title("🛡️ Web Hosting Fake Review Detection System")
st.markdown("Detect whether an English review is **Genuine** or **Fake/Deceptive**.")
st.markdown("---")

if model and tfidf_vectorizer:
    st.subheader("Enter a review for analysis")
    review_input = st.text_area(
        "Paste the review text here:", 
        height=200, 
        placeholder="e.g., 'This service is fast and reliable!'"
    )

    if st.button("Detect Review Type"):
        if review_input.strip():
            with st.spinner("Analyzing review..."):
                # Preprocess review exactly like notebook
                cleaned_input = clean_text(review_input)
                vectorized_input = tfidf_vectorizer.transform([cleaned_input])
                prediction = model.predict(vectorized_input)[0]

                # Get confidence
                try:
                    probability = model.predict_proba(vectorized_input)[0][prediction]
                except:
                    probability = None

                # Display results
                st.markdown("### Analysis Result:")
                if prediction == 1:
                    st.error("❌ **FAKE / DECEPTIVE REVIEW**")
                else:
                    st.success("✅ **GENUINE REVIEW**")
                if probability is not None:
                    st.metric("Confidence Score", f"{probability * 100:.2f}%")

        else:
            st.warning("👈 Please enter a review to analyze.")
