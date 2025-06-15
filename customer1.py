import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import gdown
import os
import logging
import google.generativeai as genai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
GEMINI_API_KEY = "AIzaSyA-9-lTQTWdNM43YdOXMQwGKDy0SrMwo6c"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
genai.configure(api_key=GEMINI_API_KEY)

# Cache NLTK data to avoid repeated downloads
@st.cache_resource
def load_nltk_data():
    logger.info("Downloading NLTK data...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    return set(stopwords.words('english')), WordNetLemmatizer()

stop_words, lemmatizer = load_nltk_data()

# Function to clean text (same as in the notebook)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = ' '.join(lemmatizer.lemmatize(word, pos='v') for word in text.split() if word not in stop_words)
    return text

# Download model from Google Drive
model_path = 'bilstm_model.h5'
model_url = 'https://drive.google.com/uc?id=1FESvhtjQUcmtxDeY_o258JUZGgE2LeIp'

if not os.path.exists(model_path):
    try:
        st.info("Downloading model from Google Drive...")
        logger.info("Starting model download...")
        gdown.download(model_url, model_path, quiet=False)
        logger.info("Model downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        st.error(f"Error downloading model: {str(e)}")
        st.stop()

# Load the trained BiLSTM model
try:
    logger.info("Loading BiLSTM model...")
    model = load_model(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Load the tokenizer from local file
tokenizer_paths = [
    'tokenizer.pkl',  # Root directory
    'customer_compliants/tokenizer.pkl'  # Subdirectory
]
tokenizer = None
for path in tokenizer_paths:
    if os.path.exists(path):
        try:
            logger.info(f"Attempting to load tokenizer from {path}...")
            with open(path, 'rb') as handle:
                tokenizer = pickle.load(handle)
            logger.info(f"Tokenizer loaded successfully from {path}")
            break
        except Exception as e:
            logger.error(f"Error loading tokenizer from {path}: {str(e)}")
            continue

if tokenizer is None:
    logger.error("Failed to load tokenizer from all attempted paths")
    st.error("Error: Could not load tokenizer. Please check if tokenizer.pkl is in the repository.")
    st.stop()

# Define maximum sequence length (must match the value used in training)
max_length = 100  # Replace with the actual max_length from your notebook

# Define product categories
product_categories = ['credit_card', 'retail_banking', 'credit_reporting', 'mortgages_and_loans', 'debt_collection']

# Function to get Gemini LLM prediction and response
def get_gemini_prediction_and_response(complaint):
    try:
        logger.info("Querying Gemini LLM...")
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = f"""
        Classify the following bank customer complaint into one of these categories: {', '.join(product_categories)}.
        Complaint: {complaint}
        Provide the category in the format: [category]
        Then, provide a response addressing the user's complaint in a professional tone.
        """
        response = model.generate_content(prompt)
        response_text = response.text.strip()

        # Extract category (assumes [category] format)
        category_match = re.search(r'\[(.*?)\]', response_text)
        if category_match:
            predicted_category = category_match.group(1).lower()
            if predicted_category not in product_categories:
                logger.warning(f"Gemini returned invalid category: {predicted_category}")
                predicted_category = 'credit_reporting'  # Fallback category
        else:
            logger.warning("Gemini did not return a category in the expected format")
            predicted_category = 'credit_reporting'  # Fallback

        # Extract response (everything after the category)
        response_body = response_text.split('\n', 1)[1] if '\n' in response_text else "We apologize for the issue. Please contact our support team for assistance."

        logger.info(f"Gemini prediction: {predicted_category}")
        return predicted_category, response_body
    except Exception as e:
        logger.error(f"Error querying Gemini LLM: {str(e)}")
        st.error(f"Error querying Gemini LLM: {str(e)}")
        return 'credit_reporting', "We apologize for the issue. Please contact our support team for assistance."

# Streamlit app layout
st.title("Bank Customer Complaint Classifier")
st.write("Enter a customer complaint narrative below, and the model will classify it into a product category and provide a response.")

# Text input for user narrative
user_input = st.text_area("Enter the complaint narrative:", height=150)

# Button to trigger classification
if st.button("Classify Complaint"):
    if user_input:
        try:
            logger.info("Processing user input...")
            # Clean the input text for BiLSTM
            cleaned_input = clean_text(user_input)
            
            # BiLSTM prediction
            logger.info("Making BiLSTM prediction...")
            input_seq = tokenizer.texts_to_sequences([cleaned_input])
            input_pad = pad_sequences(input_seq, maxlen=max_length, padding='post')
            bilstm_prediction = model.predict(input_pad, verbose=0)
            bilstm_class = np.argmax(bilstm_prediction, axis=1)[0]
            bilstm_category = product_categories[bilstm_class]
            bilstm_confidence = bilstm_prediction[0][bilstm_class] * 100
            logger.info(f"BiLSTM prediction: {bilstm_category} ({bilstm_confidence:.2f}%)")

            # Gemini LLM prediction and response
            gemini_category, gemini_response = get_gemini_prediction_and_response(user_input)

            # Compare predictions
            if bilstm_category == gemini_category:
                logger.info("BiLSTM and Gemini predictions match")
                final_category = gemini_category
                final_response = gemini_response
            else:
                logger.info(f"BiLSTM ({bilstm_category}) and Gemini ({gemini_category}) predictions differ. Using Gemini.")
                final_category = gemini_category
                final_response = gemini_response

            # Display results
            st.success(f"**Predicted Category:** {final_category}")
            st.info(f"**BiLSTM Confidence Score:** {bilstm_confidence:.2f}%")
            st.write(f"**Response:** {final_response}")
            logger.info(f"Final prediction: {final_category}")
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            st.error(f"Error during prediction: {str(e)}")
    else:
        st.error("Please enter a complaint narrative.")

# Display model performance metrics
st.subheader("Model Performance")
st.write("The BiLSTM model was trained on a dataset of bank customer complaints and achieved the following performance on the test set:")
st.write("""
- **Accuracy**: 90.06%
- **Macro Average F1-Score**: 90.02%
- **Precision per class**:
  - Credit Card: 88.03%
  - Retail Banking: 89.71%
  - Credit Reporting: 86.48%
  - Mortgages and Loans: 91.66%
  - Debt Collection: 94.66%
""")
