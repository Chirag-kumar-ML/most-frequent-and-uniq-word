import streamlit as st

import pandas as pd


import re

import collections


import nltk
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

STOP_WORDS = stopwords.words("english")

st.title("Know Most Frequemt amd Unique Word ")
st.subheader("Welcome")



def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

user_input = st.text_input("Enter Your text here", default_value_goes_here)


preprocessed_profiles = []

sentance=user_input
sentance = re.sub(r"http\S+", "", sentance)
sentance = BeautifulSoup(sentance, 'lxml').get_text()
sentance = decontracted(sentance)
sentance = re.sub("\S*\d\S*", "", sentance).strip()
sentance = re.sub('[^A-Za-z]+', ' ', sentance)
# https://gist.github.com/sebleier/554280
sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in STOP_WORDS)
preprocessed_profiles.append(sentance.strip())


list_text = preprocessed_profiles

from collections import Counter
freq_data = []
for i in list_text:
    counter = Counter(i.split())
    freq_words = [x for (x,_) in counter.most_common(5)]
    freq_data.append(", ".join(freq_words))
st.text("These are the 5 most FREQUENT Words",freq_data[0])




from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from nltk.stem.porter import PorterStemmer



# TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


tf_idf_vect = TfidfVectorizer(ngram_range=(1,2), min_df=1)
tf_idf_vect.fit(list_text)
#print("some sample features(unique words in the corpus)",tf_idf_vect.get_feature_names()[0:10])
print('='*50)
n_print=5
list_uniq_words = []
for line in list_text:
    final_tf_idf = tf_idf_vect.transform([line])
    feature_array = np.array(tf_idf_vect.get_feature_names())
    tfidf_sorting = np.argsort(final_tf_idf.toarray()).flatten()[::-1]
    top_n = feature_array[tfidf_sorting][:n_print]
    list_uniq_words.append(', '.join(top_n))
st.text("These are the 5 most FREQUENT Words",list_uniq_words[0])    
