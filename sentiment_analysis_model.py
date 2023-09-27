import numpy as np
import re
import string
import pickle
import nltk
from nltk.corpus import stopwords

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Inicializando o lemmatizer e o conjunto de stopwords

english_stopwords = set(['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'])

# Carregar o modelo treinado e o tokenizer
MODEL_PATH = 'lstm_binary_model.h5'
TOKENIZER_PATH = 'tokenizer.pickle'

model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Constante
MAX_SEQUENCE_LENGTH = 512 

# Funções necessárias



def process_text(text):
    """Processa o texto."""
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    filtered_and_lemmatized_words = [word for word in words if word.lower() not in english_stopwords]
    text = ' '.join(filtered_and_lemmatized_words)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def predict_sentiment(text):
    """Prevê o sentimento de um texto com base em intervalos de probabilidade."""
    text = process_text(text)
    sequence_text = tokenizer.texts_to_sequences([text])
    X_text = pad_sequences(sequence_text, maxlen=MAX_SEQUENCE_LENGTH)
    y_pred = model.predict(X_text)[0][0]

    # Definir intervalos de sentimento
    if y_pred >= 0.6:
        sentiment = "Positive"
    elif y_pred <= 0.4:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment


