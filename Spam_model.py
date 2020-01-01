# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:45:33 2019

@author: BAHAR
"""

#import numpy as np
import pandas as pd
#import string
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.model_selection import train_test_split
#import nltk
#from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
#import matplotlib.pyplot as plt
import pickle

#Run the below piece of code for the first time
#nltk.download('stopwords')

message_data = pd.read_csv("spam.csv",sep=",",encoding = "latin")
message_data = message_data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
message_data = message_data.rename(columns = {'v1':'Spam/Not_Spam','v2':'message'})

message_data.groupby('Spam/Not_Spam').describe()

message_data_copy = message_data['message'].copy()


def stemmer (text):
    text = text.split()
    words = ""
    for i in text:
            stemmer = SnowballStemmer("english")
            words += (stemmer.stem(i))+" "
    return words

message_data_copy = message_data_copy.apply(stemmer)
vectorizer = TfidfVectorizer("english")
message_mat = vectorizer.fit_transform(message_data_copy)

#message_train, message_test, spam_nospam_train, spam_nospam_test = train_test_split(message_mat, 
                                                        #message_data['Spam/Not_Spam'], test_size=0.3, random_state=20)

from sklearn.linear_model import LogisticRegression


message_train=message_mat
spam_nospam_train=message_data['Spam/Not_Spam']

Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(message_train, spam_nospam_train)

# Saving model to disk
pickle.dump(Spam_model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

#%%


def is_spam(inp):
    print(inp)
    inp=pd.Series(inp)
    print("series",inp)
    inp =inp.apply(stemmer)
    print("stemmer",inp)
    inp_test=vectorizer.transform(inp)
    print("inp_test",inp_test)
    inp_sonuc=model.predict(inp_test)
    print("tahmin",inp_sonuc)

    if inp_sonuc=='spam':
        return True
    else:
        return False
    
 
#print(is_spam(inp = ["""\
#Please call our customer service representative on 0800 169 6031 between 10am-9pm as you have WON a guaranteed å£1000 cash or å£5000 prize!
#"""]))

#inp1 = "Please call our customer service representative on 0800 169 6031 between 10am-9pm as you have prize!"  
#is_spam(inp1)
#model.predict(vectorizer.transform(inp)[0])