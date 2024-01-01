import numpy as np
import pandas as pd
import pickle
import streamlit as st
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import time

# loading the saved model
loaded_model = pickle.load(open('./movie_genre_classification_model.sav', 'rb'))

df_train = pd.read_csv("./train_data.txt",sep=':::',names=['Title', 'Genre', 'Description']).reset_index(drop=True)

def data_clean(txt):
    # Removing punctuations
    txt = re.sub(f'[{string.punctuation}]','',txt)
    
    # Removing numbers
    txt = re.sub(f'[{string.digits}]','',txt)
    
    # Removing single characters 
    txt = re.sub(r'\s+[a-zA-Z]\s+', ' ', txt)
    
    # Removing emails
    txt = re.sub(r'@\S+', '', txt)
    
    # Removing URLs
    txt = re.sub(r'http\S+', '', txt)
    
    return txt

# Applying the Data Cleaning function to the Description column of Train
df_train['Description'] = df_train['Description'].apply(data_clean)

tfidf = TfidfVectorizer(lowercase=True, stop_words='english',min_df=1)
tfidf.fit_transform(df_train['Description'])

def movie_genre_prediction(input_data, tfidf_vectorizer, nb_model, lr_model):
    # Cleaning the Input
    cleaned_desc = data_clean(input_data)
    
    # Vectorize the input
    movie_desc_vectorized = tfidf_vectorizer.transform([cleaned_desc])
    
    # Prediction of Naive Bayes Model:
    nb_model_pred = nb_model.predict(movie_desc_vectorized)
    nb_prediction_statement = f'Naive Bayes Model : {nb_model_pred[0].title()}'
    
    # Prediction of Logistic Regression Model:
    lr_model_pred = lr_model.predict(movie_desc_vectorized)
    lr_prediction_statement = f'Logistic Regression Model : {lr_model_pred[0].title()}'
    
    return nb_prediction_statement, lr_prediction_statement

def main():
    # giving a title
    st.title('Movie Genre Classifier')

    # getting the input data from the user
    Movie_Description = st.text_input('Enter the Movie Description below ...')

    # creating a button for Prediction
    if st.button('Predict Genre'):
      nb_prediction, lr_prediction = movie_genre_prediction(Movie_Description, tfidf, loaded_model[0], loaded_model[1])
      st.subheader('Movie Description')
      st.write(Movie_Description)
      
      st.subheader('Predictions')
      st.success(nb_prediction)
      st.success(lr_prediction)

if __name__ == '__main__':
    main()
