# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 22:07:14 2019

@author: BAHAR
"""

import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk #doğal dil işleme
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt

#
nltk.download('stopwords')



dataset=pd.read_csv("spam.csv",encoding = "latin")
#gereksiz kolonları silme
dataset=dataset.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
dataset = dataset.rename(columns = {'v1':'Spam/Ham','v2':'message'})

message_copy = dataset['message'].copy() #
#%%
#punctuation- noktalama işareti
#stopwords- atılacak-çok kullanılan kelimeler

def text_preprocess(text):
    #sözlük oluşturma-translate ile tercüme-noktalama işaretlerinden kurtulma
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return " ".join(text)
message_copy = message_copy.apply(text_preprocess)


# vektörize etme
vectorizer = TfidfVectorizer("english")
message_mat = vectorizer.fit_transform(message_copy)

X_train, X_test, y_train, y_test = train_test_split(message_mat, dataset['Spam/Ham'], test_size=0.3, random_state=20)
    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(X_train, y_train)
y_pred = Spam_model.predict(X_test)
acc1=accuracy_score(y_test,y_pred)
print("Accuracy 1:", acc1)

data='my name is tom'
data=pd.Series(data)

data = data.apply(text_preprocess)
vectorizer = TfidfVectorizer("english")
data_test = vectorizer.fit_transform(data)


print(Spam_model.predict(data_test))
#%%

#stemming and normalizing -köklerine ayırma
def stemmer (text):
    text = text.split()
    words = " "
    for i in text:
        stemmer = SnowballStemmer("english")
        words += (stemmer.stem(i))+" "
    return words

message_copy = message_copy.apply(stemmer)
vectorizer = TfidfVectorizer("english")
message_mat = vectorizer.fit_transform(message_copy)


X_train, X_test, y_train, y_test = train_test_split(message_mat, dataset['Spam/Ham'], test_size=0.3, random_state=20)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(X_train, y_train)
pred = Spam_model.predict(X_test)
acc2 =accuracy_score(y_test,pred)
print("Accuracy 2:", acc2)




#%%

#uzunluğa göre
dataset['length'] = dataset['message'].apply(len)

vectorizer = TfidfVectorizer("english")
message_mat = vectorizer.fit_transform(message_copy)

length = dataset['length'].as_matrix()
new_mat = np.hstack((message_mat.todense(),length[:, None]))

X_train, X_test, y_train, y_test = train_test_split(new_mat, dataset['Spam/Ham'], test_size=0.3, random_state=20)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

Spam_model = LogisticRegression(solver='liblinear', penalty='l1')
Spam_model.fit(X_train, y_train)
pred = Spam_model.predict(X_test)
acc3=accuracy_score(y_test,pred)
print("Accuracy 3:", acc3)


