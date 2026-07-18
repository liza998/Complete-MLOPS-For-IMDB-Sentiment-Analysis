from flask import Flask,redirect,url_for,render_template,request
import pickle 
import pandas as pd 
import os 
import logging
import yaml
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
import re
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer

with open("./Vector/tfidf.pkl","rb") as file:
    tfidf = pickle.load(file) 

with open("./models/LogisticRegression/LogisticRegression.pkl", "rb") as file:
    model = pickle.load(file)

def transform(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+'," ",text)
    text = re.sub(r'@\w+|#\w+'," ",text)
    text = "".join([char for char in text if char.isalpha() or char.isdigit() or char.isspace()])
    Stopwords = stopwords.words("english")
    lemmatizer = WordNetLemmatizer ()
    word = text.split()
    words = [lemmatizer.lemmatize(w) for w in word if w not in Stopwords]
    text = " ".join(words)
    return text


app=Flask(__name__)
@app.route('/',)
def home():
    return render_template('index.html')
@app.route('/upload',methods= ["POST"])
def input():
    if request.method == "POST":
        text = request.form["text"]
        process_text = transform(text)
        vector_text = tfidf.transform([process_text])
        predict = model.predict(vector_text)[0]
        if predict == 1:
            prediction = "POSITIVE Review"
        else:
            prediction = "NEGATIVE Review"
        return prediction


if __name__ == '__main__':
    #DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(host="0.0.0.0",port=5000,debug=True)