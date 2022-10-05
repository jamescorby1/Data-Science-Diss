from keras.saving.save import load_model
import streamlit as st
import re
import nltk
import string
import pickle 
import os
from keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

#Clean the text8 
def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split("\W+", text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text 

#Load Vectorizer and Tokenizer
vectorizer = pickle.load(open("/Users/jamescorby/Documents/MSc Data Science/Dissertation Files/Project Dissertation Code/webapp/ctiVectorizer.pickle", "rb"))
tokenizer = pickle.load(open("/Users/jamescorby/Documents/MSc Data Science/Dissertation Files/Project Dissertation Code/webapp/ctitokenizer.pickle", "rb"))

#Load Neural Network 
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()

#loaded_model = model_from_json(loaded_model_json)
#loaded_model.load_weights("model.h5")

#Models
def result(pred):
   if pred == 'Critical':
      return "This post could be related to a Cyber Threat!"
   else: 
      return "This post does not look like a Cyber Threat."

def main():
    st.title("CYBER THREAT INTELLIGENCE CLASSIFIER.")
    st.subheader("About: ")
    st.write("The purpose of this app is to showcase mutliple machine learning models that have been trained to detect cyber threats from open source forums such as Reddit and Twitter."
      " Each model calssifies the entered post into one of two categories; Critical and Non Critical. "
    )
    st.subheader("Lets classify a recent post! Try below: ")
    post_text = st.text_area("Enter text", "Copy and paste a social media post here")
    all_models = ["Random Forest", "Gradient Booster", "Multinomial Bayes", "Linear SVM", "Neural Network"]
    model_choice = st.selectbox('Please choose a Machine Learning model: ', all_models)

    if st.button("Classify"):
       st.text("The original post: \n{}".format(post_text))
       vect_text = vectorizer.transform([post_text])
       if model_choice == "Random Forest":
          predictor = pickle.load(open("/Users/jamescorby/Documents/MSc Data Science/Dissertation Files/Project Dissertation Code/webapp/ctiRandomForest_model.pickle", "rb"))
          prediction = predictor.predict(vect_text)
          #st.write(prediction)
       elif model_choice == "Gradient Booster":
          predictor = pickle.load(open("/Users/jamescorby/Documents/MSc Data Science/Dissertation Files/Project Dissertation Code/webapp/ctiGradientBoosting_model.pickle", "rb"))
          prediction = predictor.predict(vect_text)
          #st.write(prediction)
       elif model_choice == "Multinomial Bayes":
          predictor = pickle.load(open("/Users/jamescorby/Documents/MSc Data Science/Dissertation Files/Project Dissertation Code/webapp/ctiBayes_model.pickle", "rb"))
          prediction = predictor.predict(vect_text)
          #st.write(prediction)
       elif model_choice == "Linear SVM":
          predictor = pickle.load(open("/Users/jamescorby/Documents/MSc Data Science/Dissertation Files/Project Dissertation Code/webapp/ctiLinear_SVM_model.pickle", "rb"))
          prediction = predictor.predict(vect_text)
          #st.write(prediction)
       final_result = result(prediction)
       st.success(final_result)
    st.subheader("Disclaimer: ")
    st.write("Sometimes the models do not correctly classify input posts, therefore it is in the best interest that posts that are classified as Critical, are checked by an expert.")


if __name__ == '__main__':
    main()


