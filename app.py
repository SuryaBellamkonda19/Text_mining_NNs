import streamlit as st

# Dummy predict function (replace with your actual model)
def predict_sentiment(text):
    # Example: return 'Positive', 'Negative', or 'Neutral'
    return "Positive"  # Replace with your prediction logic

# Streamlit App
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊", layout="centered")
st.title("📝 Sentiment Analyzer")
st.write("Enter a tweet or sentence below to predict its sentiment:")

user_text = st.text_area("Your text here...", height=100)

if st.button("Predict Sentiment"):
    if not user_text.strip():
        st.warning("Please enter some text to analyze!")
    else:
        result = predict_sentiment(user_text)
        
        # Display sentiment with colors and emojis
        if result.lower() == "positive":
            st.success(f"😊 Positive Sentiment")
        elif result.lower() == "negative":
            st.error(f"😞 Negative Sentiment")
        else:
            st.info(f"😐 Neutral Sentiment")
        
        # Optional: Show the input text
        st.markdown(f"**Input Text:** {user_text}")
