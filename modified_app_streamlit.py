import streamlit as st

import time

# URL da sua API Flask


#Image
#st.image('\\wsl.localhost\Ubuntu\home\edilson07\projects\sentiment_analysis\api\sentiment_robot.png', width=100)


st.title("Sentiment Analysis Prediction")

# Campo de texto para o usuário inserir a revisão
user_input = st.text_area("Enter your review:")

# Botão para fazer a previsão
if st.button("Predict"):
    # Enviar a revisão para a API Flask e obter a previsão
    sentiment = sam.predict_sentiment(user_input)
    with st.spinner(text="Predicting..."):
        time.sleep(2)    
    
    # Verifique se a resposta foi bem-sucedida
    if sentiment:
        json_response = response.json()
        #st.write(f"Response from API: {json_response}")  # Escreva a resposta completa para depuração
        
        # Tente obter a previsão
        try:
            prediction = sentiment
            if prediction == "Positive":
                st.markdown("**Sentiment:** :smile: Positive!")
            else:
                st.markdown("**Sentiment:** :disappointed: Negative!")
        except KeyError:
            st.write("Key 'sentiment' not found in the API response.")


