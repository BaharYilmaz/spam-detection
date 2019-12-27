import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt

nltk.download('stopwords')

dataset=pd.read_csv("spam.csv",encoding = "latin")
#gereksiz kolonları silme
dataset=dataset.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)
dataset = dataset.rename(columns = {'v1':'Spam/Ham','v2':'message'})

message_copy = dataset['message'].copy()


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

from sklearn.linear_model import LogisticRegression
Spam_model = LogisticRegression(solver='liblinear', penalty='l1')

x = message_mat['message']
y = dataset['Spam/Ham']
Spam_model.fit(x, y)

# Save your model
from sklearn.externals import joblib
joblib.dump(Spam_model, 'model.pkl')
print("Model dumped!")

# Load the model that you just saved
Spam_model = joblib.load('model.pkl')

# Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")