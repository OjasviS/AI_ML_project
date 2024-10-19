# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 22:05:04 2024

@author: ojiha
"""

import tkinter as tk
from tkinter import messagebox
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


def preprocessText(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in string.punctuation and word not in stopwords.words('english')]
    return tokens


with open("C:\\CS\\ML_lab\\models\\Logistic_sentiment_model.pkl", 'rb') as f1:
    model = pickle.load(f1)

with open("C:\\CS\\ML_lab\\models\\count_vectorizer.pkl", 'rb') as f2:
    vectorizer = pickle.load(f2)


def predict_sentiment():
    input_text = text_input.get() 
    if not input_text.strip():
        messagebox.showerror("Error", "Please enter a valid sentence.")
        return
    
    
    processed_input = preprocessText(input_text)
    processed_input_text = " ".join(processed_input)  # to convert tokens back to string

    
    input_vector = vectorizer.transform([processed_input_text])

    
    prediction = model.predict(input_vector)

   
    if prediction == 1:
        result_label.config(text="Sentiment: Positive", fg="green")
    else:
        result_label.config(text="Sentiment: Negative", fg="red")


root = tk.Tk()
root.title("Sentiment Analysis")

# input box
text_input = tk.Entry(root, width=50)
text_input.pack(pady=10)

# button
predict_button = tk.Button(root, text="Predict Sentiment", command=predict_sentiment)
predict_button.pack(pady=10)

# result feild
result_label = tk.Label(root, text="Sentiment will appear here.", font=("Helvetica", 14))
result_label.pack(pady=20)


root.mainloop()
