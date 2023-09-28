import numpy as np
import re
import string
import pickle
import nltk
import os
from nltk.corpus import stopwords
from nltk.corpus.reader import wordnet
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Inicializando o lemmatizer e o conjunto de stopwords

current_directory = os.path.dirname(os.path.realpath(__file__))
wordnet_path = os.path.join(current_directory, 'wordnet')
nltk.data.path.append(wordnet_path)

print(nltk.data.path)

english_stopwords = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've",
    "you'll", "you'd", "your", "yours", "yourself", "yourselves", "he", "him", "his",
    "himself", "she", "she's", "her", "hers", "herself", "it", "it's", "its", "itself",
    "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
    "that", "that'll", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but",
    "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't", "should", "should've",
    "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn", "couldn't",
    "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't",
    "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan",
    "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't",
    "wouldn", "wouldn't"])

# Carregar o modelo treinado e o tokenizer
MODEL_PATH = 'lstm_binary_model.h5'
TOKENIZER_PATH = 'tokenizer.pickle'

model = load_model(MODEL_PATH)
with open(TOKENIZER_PATH, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Constante
MAX_SEQUENCE_LENGTH = 512 

# Funções necessárias

lemmatizer = WordNetLemmatizer()

def lemmatize_word(word):
    lemma = lemmatizer.lemmatize(word, pos=wordnet.VERB)
    return lemmatizer.lemmatize(lemma, pos=wordnet.NOUN)


def process_text(text):
    """Processa o texto."""
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    #filtered_and_lemmatized_words = [word for word in words if word.lower() not in english_stopwords]
    filtered_and_lemmatized_words = [lemmatize_word(word) for word in words if word.lower() not in english_stopwords]
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
    if y_pred >= 0.66:
        sentiment = "Positive"
    elif y_pred <= 0.33:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return sentiment


