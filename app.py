import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

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

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("E-mail/SMS Spam Detector")

inputmsg = st.text_input("Enter the mail/message...")

if st.button('Predict'):

    transformed_text = transform_text(inputmsg)

    vector_text = tfidf.transform([transformed_text])

    result = model.predict(vector_text)[0]

    if result == 1:
        st.header("Spam!")
    else:
        st.header("Not Spam.")