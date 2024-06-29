import streamlit as st
import pickle
from nltk.corpus import stopwords
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score
import numpy as np
# import models
tfidf = pickle.load(open('./models/vectorizer.pkl', 'rb'))
model = pickle.load(open('./models/model.pkl', 'rb'))

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# App
st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess the input text
    transformed_sms = transform_text(input_sms)
    # Vectorize the transformed text
    vector_input = tfidf.transform([transformed_sms])
    # Convert the sparse matrix to dense array
    vector_input_dense = vector_input.toarray()
    # Predict using the Voting Classifier
    result_voting_proba = model.predict_proba(vector_input_dense)[0]

    # Display results based on probability threshold
    if result_voting_proba[1] > 0.5:
        st.header("Voting Classifier: Spam")
    else:
        st.header("Voting Classifier: Not Spam")
