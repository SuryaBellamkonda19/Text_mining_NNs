import streamlit as st
from keras.models import load_model
from keras.utils import pad_sequences
import pickle
import re
import numpy as np

# --- Load model and tokenizer ---
model = load_model("sentiment_cnn_model")  # replace with your path if needed
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# --- Preprocessing function ---
def preprocess_input(text):
    text = text.lower()
    # Remove URLs, mentions, hashtags
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    # Keep only letters and spaces
    text = re.sub(r"[^a-z\s]", "", text)

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)  # maxlen should match model input
    return padded

# --- Prediction function ---
def predict_sentiment(text):
    processed_text = preprocess_input(text)
    prediction = model.predict(processed_text)[0][0]  # model output
    if prediction >= 0.5:
        return "Positive"
    else:
        return "Negative"

# --- Streamlit App ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊", layout="centered")
st.title("📝 Sentiment Analyzer")
st.markdown("""
## 📖 User Guide

Enter a tweet or sentence below and click **Predict Sentiment**.  
The app will show whether the sentiment is **Positive** or **Negative**.
""")

# --- User Input ---
user_text = st.text_area("Enter your text here...", height=100)

# --- Prediction ---
if st.button("Predict Sentiment"):
    if not user_text.strip():
        st.warning("Please enter some text to analyze!")
    else:
        result = predict_sentiment(user_text)
        
        if result.lower() == "positive":
            st.success(f"😊 Positive Sentiment")
        else:
            st.error(f"😞 Negative Sentiment")
        
        st.markdown(f"**Input Text:** {user_text}")
