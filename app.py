# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import Spam_model as sm 
from sklearn.externals import joblib
from flask import Flask, request, jsonify, render_template
import pickle
import json 

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    if sm:
        
        # json_=request.json
        # query=pd.get_dummies(pd.DataFrame(json_))
        # # query = query.reindex(columns=model_columns, fill_value=0)
        # sonuc=sm.is_spam(query)
        data=request.form['data']
        print(data)
        sonuc=sm.is_spam(data)
        print("sonuc",sonuc)
        print("json gtti")
        if sonuc==True:
            return render_template('index.html', predict='SPAM Mail')
        else:
            return render_template('index.html', predict='Not SPAM Mail')

        
    else:
        print ('Train the model first')
        return ('No model here to use')   
        

    
if __name__ == '__main__':
    
    # Spam_model = joblib.load("model.pkl") # Load "model.pkl"
    # print('model loaded')
    # model_columns=joblib.load("model_columns.pkl")
    # try:
               
    #     dump=np.load('filename.npz',encoding='bytes')
    #     dump=dict(dump[dump.files[0]].tolist())
    #     dump=dump = {str(k.decode('utf-8')): dump[k] for k in dump}
    # except:
    #     dump = np.load(open('filename.npz', 'rb'))
    #     dump = dict(dump[dump.files[0]].tolist())
    #     dump = {str(k): dump[k] for k in dump}
    
    app.run(debug=True)