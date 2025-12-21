import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import os

MODEL_PATH = "sentiment_cnn_model.h5"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 100

model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=MAX_LEN)

def predict_sentiment(text):
    pred = model.predict(preprocess_text(text))[0][0]
    return "Positive" if pred >= 0.5 else "Negative"

st.title("Sentiment Analyzer (CNN)")
text = st.text_area("Enter text")

if st.button("Predict"):
    if text.strip():
        st.success(predict_sentiment(text))
    else:
        st.warning("Enter some text")
