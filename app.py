import streamlit as st
import joblib
import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import base64

# --- NLTK Functions ---

# Download NLTK data if needed (Cached for fast re-runs)
@st.cache_resource
def download_nltk_data():
    try:
        # Downloads are done once per deployment
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
    except Exception:
        # Suppress error if NLTK fails to download on some environments
        pass

# --- Model Loading ---

# Load the trained model (Cached for fast re-runs)
@st.cache_resource
def load_model():
    try:
        # NOTE: Assumes the pipeline output is the vectorizer + final classifier
        model = joblib.load('spam_detection_model.pkl')
        with open('model_info.json', 'r') as f:
            model_info = json.load(f)
        return model, model_info
    except FileNotFoundError:
        # If files are missing, display an error
        st.error("Model files ('spam_detection_model.pkl' or 'model_info.json') not found! Please ensure they are uploaded.")
        return None, None

# --- Preprocessing Function (Reference) ---

# Enhanced text preprocessing function (same as training)
def preprocess_text(text):
    """Clean and preprocess text data with enhanced features"""
    text = text.lower()
    
    # Keep some punctuation patterns that might be useful for spam detection
    text = re.sub(r'!{2,}', ' MULTIPLE_EXCLAMATION ', text)
    text = re.sub(r'\?{2,}', ' MULTIPLE_QUESTION ', text)
    
    # Mark ALL CAPS words (common in spam)
    text = re.sub(r'\b[A-Z]{3,}\b', ' ALLCAPS_WORD ', text)
    
    # Mark URLs and emails
    text = re.sub(r'http[s]?://\S+', ' URL_LINK ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL_ADDRESS ', text)
    
    # Mark numbers but keep them as NUMBER token
    text = re.sub(r'\d+', ' NUMBER ', text)
    
    # Remove most punctuation but keep sentence structure
    text = re.sub(r'[^\w\s!?.]', ' ', text)
    text = re.sub(r'[!?.]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# --- Streamlit CSS/Styling ---

# Function to set background image with container
def set_background(image_path):
    """Set background image for the app with styled container and white text, black input text"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        # Use a fallback color if image not found
        encoded_string = ""

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(
                rgba(0, 0, 0, 0.5),   /* Dark overlay (50% opacity) */
                rgba(0, 0, 0, 0.5)
            ){(f", url(data:image/png;base64,{encoded_string})" if encoded_string else "#0e1117")};
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            color: white !important;
        }}
        .main-content {{
            background-color: rgba(255, 255, 255, 0.1);
            padding: 2rem;
            border-radius: 10px;
            margin: 1rem 0;
            color: white !important;
        }}
        .prediction-box {{
            background-color: rgba(255, 255, 255, 0.15);
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 5px solid #1f77b4;
            color: white !important;
        }}
        .spam-alert {{
            background-color: rgba(255, 0, 0, 0.2);
            border-left: 5px solid #ff4444;
            color: white !important;
        }}
        .safe-alert {{
            background-color: rgba(0, 255, 0, 0.2);
            border-left: 5px solid #44ff44;
            color: white !important;
        }}
        /* Make all text white */
        .stApp, .stApp * , h1, h2, h3, h4, h5, h6, p, div, span {{
            color: white !important;
        }}
        /* Reset sidebar text back to black */
        section[data-testid="stSidebar"] *, 
        section[data-testid="stSidebar"] div, 
        section[data-testid="stSidebar"] span {{
            color: black !important;
        }}
        /* Force input text (textarea, input, select) to black */
        .stTextArea textarea, 
        .stTextInput input, 
        .stSelectbox select {{
            color: black !important;
            background-color: rgba(255, 255, 255, 0.8) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Main Application Logic ---

def main():
    # Set page config
    st.set_page_config(
        page_title="Fake Review Detection System",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Set background (change "background.jpg" to your actual image path or remove)
    set_background("background.jpg")
    
    # Download NLTK data
    download_nltk_data()
    
    # Load model
    model, model_info = load_model()
    
    if model is None:
        st.stop()
        
    # Determine the class names from the model (assuming binary classification)
    try:
        class_names = model.classes_
        # Find the index of the 'deceptive' or 'spam' class
        spam_class_index = list(class_names).index('deceptive') if 'deceptive' in class_names else 1
    except Exception:
        # Fallback if model classes cannot be accessed (e.g., if model is a pipeline)
        class_names = ['safe', 'deceptive']
        spam_class_index = 1
        
    # Initialize the missing user_rating variable
    user_rating = None
        
    # Main content
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Header
    st.title("🛡️ Fake Review Detection System")
    st.markdown("---")
    
    # Sidebar with model info
    with st.sidebar:
        st.header("📊 Model Information")
        st.info(f"**Algorithm:** {model_info.get('model_name', 'Unknown')}")
        st.info(f"**Accuracy:** {model_info.get('accuracy', 0):.1%}")
        st.info(f"**Features:** {model_info.get('max_features', 'Unknown')}")
        
        # Add rating input to sidebar (optional, but needed for the recommendation logic)
        st.header("⭐ Optional Input")
        user_rating = st.slider("Provide an associated rating (1-5):", min_value=1, max_value=5, value=3)
        
        st.header("🎯 How it works")
        st.write("""
        1. **Input:** Enter your text/review
        2. **Processing:** Text is cleaned and vectorized
        3. **Prediction:** ML model analyzes patterns
        4. **Result:** Get spam probability with confidence
        """)
        
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Enter Text to Analyze")
        
        input_method = st.radio(
            "Choose input method:",
            ["Type text", "Upload file", "Use examples"]
        )
        
        user_text = ""
        
        if input_method == "Type text":
            user_text = st.text_area(
                "Enter your text here:",
                height=150,
                placeholder="Paste your text, review, or message here..."
            )
        
        # ... (File upload and Examples sections remain the same) ...
        
        elif input_method == "Upload file":
            uploaded_file = st.file_uploader(
                "Upload a text file",
                type=['txt'],
                help="Upload a .txt file to analyze"
            )
            if uploaded_file is not None:
                user_text = str(uploaded_file.read(), "utf-8")
                st.text_area("File content:", user_text, height=100, disabled=True)
        
        elif input_method == "Use examples":
            examples = {
                "Suspicious Review": "This hotel is AMAZING!!! Best deal ever! Book now and get 90% discount! Limited time offer!",
                "Genuine Review": "I stayed at this hotel last week. The room was clean and the staff was helpful. The location is convenient for downtown attractions.",
                "Spam-like": "URGENT! You won a million dollars! Click here now! Don't miss this incredible opportunity!",
                "Normal Text": "The conference was informative and well-organized. The speakers provided valuable insights into current industry trends."
            }
            
            selected_example = st.selectbox("Choose an example:", list(examples.keys()))
            user_text = examples[selected_example]
            st.text_area("Selected example:", user_text, height=100, disabled=True)

    with col2:
        st.subheader("⚙️ Analysis Settings")
        
        show_confidence = st.checkbox("Show confidence score", value=True)
        show_processing = st.checkbox("Show text processing", value=False)
        
        st.subheader("📊 Quick Stats")
        if user_text:
            st.metric("Characters", len(user_text))
            st.metric("Words", len(user_text.split()))
            st.metric("Lines", len(user_text.split('\n')))
            
    # Analysis button and results
    if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
        if user_text.strip():
            with st.spinner("Analyzing text..."):
                # Preprocess text
                cleaned_text = preprocess_text(user_text)
                
                # Make prediction
                prediction_label = model.predict([cleaned_text])[0] # e.g., 'deceptive' or 'safe'
                probabilities = model.predict_proba([cleaned_text])[0]
                
                # Determine result based on spam_class_index
                spam_prob = probabilities[spam_class_index]
                safe_prob = probabilities[1 - spam_class_index]
                
                is_spam = prediction_label == class_names[spam_class_index]
                
                confidence = max(probabilities)
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Main result box
                result_class = "spam-alert" if is_spam else "safe-alert"
                result_emoji = "🚨" if is_spam else "✅"
                result_text = "SPAM DETECTED" if is_spam else "LEGITIMATE TEXT"
                result_color = "red" if is_spam else "green"
                
                st.markdown(
                    f"""
                    <div class="prediction-box {result_class}">
                        <h2 style="color: {result_color}; text-align: center;">
                            {result_emoji} {result_text}
                        </h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Detailed results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Prediction",
                        prediction_label.title(),
                        delta="Spam" if is_spam else "Safe"
                    )
                
                with col2:
                    if show_confidence:
                        st.metric(
                            # Label is the specific spam probability
                            f"{class_names[spam_class_index].title()} Probability",
                            f"{spam_prob:.1%}",
                            # Delta shows the opposite class probability for comparison
                            delta=f"{safe_prob:.1%} safe probability"
                        )
                
                with col3:
                    risk_level = "HIGH" if spam_prob > 0.8 else "MEDIUM" if spam_prob > 0.5 else "LOW"
                    st.metric("Risk Level", risk_level)
                
                # Show processing details if requested
                if show_processing:
                    st.subheader("🔧 Text Processing Details")
                    st.text_area("Original text:", user_text, height=100, disabled=True)
                    st.text_area("Processed text:", cleaned_text, height=100, disabled=True)
                
                # Enhanced recommendations with rating insights
                st.subheader("💡 Recommendations")
                if is_spam:
                    st.warning("""
                    **⚠️ This text appears to be spam or deceptive. Consider:**
                    - Verify the source before trusting
                    - Look for unrealistic claims or urgent language
                    - Check for spelling/grammar issues
                    - Be cautious of unsolicited offers
                    - Suspicious patterns in rating vs. content
                    """)
                    
                    # Recommendation based on the user_rating input
                    if user_rating is not None and user_rating >= 4:
                        st.error("🚨 **HIGH ALERT**: High rating combined with spam-like content suggests fake review!")
                
                else:
                    st.success("""
                    **✅ This text appears to be legitimate. However:**
                    - Always use your judgment
                    - Verify important information from multiple sources
                    - Be aware that sophisticated spam can be harder to detect
                    - Consider the rating-content consistency
                    """)
                
        else:
            st.error("Please enter some text to analyze!")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            <p>🛡️ Spam Detection System | Built with Streamlit & Machine Learning</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
