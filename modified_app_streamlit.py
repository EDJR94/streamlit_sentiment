import streamlit as st
import sentiment_analysis_model as sam

# Page configuration
st.set_page_config(page_title="🤖 Sentiment Analysis App 🤖", layout="centered", initial_sidebar_state="expanded", page_icon='🤖')

# Adding title with robot emojis
st.title("🤖 Sentiment Analysis App 🤖")
st.write("Welcome to the sentiment analysis application. Enter a review below to get the sentiment!")

# Instructions
st.sidebar.header("Instructions")
st.sidebar.text("1. Write a review in the text box.")
st.sidebar.text("2. Click on the 'Predict' button.")
st.sidebar.text("3. View the sentiment result.")

# Taking user input
user_input = st.text_area("Write a review here:")

# Adding a Predict button
if st.button('Predict'):
    # Feedback message while processing
    with st.spinner("Analyzing..."):
        sentiment = sam.predict_sentiment(user_input)

    # Displaying the result with emojis
    st.subheader("Sentiment Result:")
    if sentiment == "Positive":
        st.success(f"😊 {sentiment}")
    elif sentiment == "Negative":  # Fixed this part
        st.error(f"😞 {sentiment}")




