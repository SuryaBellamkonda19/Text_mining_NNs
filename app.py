import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import re
import numpy as np
import os

# --- Paths ---
MODEL_PATH = "sentiment_cnn_model.h5"  # Use .h5 file instead of SavedModel folder
TOKENIZER_PATH = "tokenizer.pkl"

# --- Load model and tokenizer ---
if os.path.exists(MODEL_PATH):
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
else:
    st.error(f"Model file not found at {MODEL_PATH}. Please convert your SavedModel to .h5 format.")

if os.path.exists(TOKENIZER_PATH):
    try:
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
else:
    st.error(f"Tokenizer file not found at {TOKENIZER_PATH}")

# --- Preprocessing function ---
def preprocess_input(text):
    text = text.lower()
    # Remove URLs, mentions, hashtags
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", "", text)

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)
    return padded

# --- Prediction function ---
def predict_sentiment(text):
    try:
        processed_text = preprocess_input(text)
        prediction = model.predict(processed_text)[0][0]
        return "Positive" if prediction >= 0.5 else "Negative"
    except Exception as e:
        return f"Error during prediction: {e}"

# --- Streamlit App ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊", layout="centered")
st.title("📝 Sentiment Analyzer")

st.markdown("""
## 📖 User Guide

Enter a tweet or sentence below and click **Predict Sentiment**.  
The app will show whether the sentiment is **Positive** or **Negative**.
""")

user_text = st.text_area("Enter your text here...", height=100)

if st.button("Predict Sentiment"):
    if not user_text.strip():
        st.warning("Please enter some text to analyze!")
    else:
        result = predict_sentiment(user_text)
        
        if result == "Positive":
            st.success(f"😊 Positive Sentiment")
        elif result == "Negative":
            st.error(f"😞 Negative Sentiment")
        else:
            st.warning(result)  # Show error messages if any
        
        st.markdown(f"**Input Text:** {user_text}")
