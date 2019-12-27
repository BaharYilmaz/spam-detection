from flask import Flask, request, jsonify
from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
import json

# Your API definition
app = Flask(__name__,template_folder="templates")

@app.route('/predict', methods=['POST'])
def predict():
    if Spam_model:

        try:
            json_ = request.json
            print(json.dumps(json_))
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(Spam_model.predict(query))
            resp= jsonify({'prediction': str(prediction)})
            response=app.response_class(response=json.dumps(prediction),mimetype='application/json')
            #print str(prediction)
            return str(prediction)
        except:

            return jsonify({'trace': traceback.format_exc()})

    else:
        print ('Train the model first')
        return ('No model here to use')

@app.route('/')
def home():
    return templates('index.html')
if __name__ == '__main__':

    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    Spam_model = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    try: 
         # for py3
        dump = np.load('filename.npz', encoding='bytes')
        dump = dict(dump[dump.files[0]].tolist())
        dump = {str(k.decode('utf-8')): dump[k] for k in dump}
    except:  # for py2
        dump = np.load(open('filename.npz', 'rb'))
        dump = dict(dump[dump.files[0]].tolist())
        # dump = {str(k): dump[k] for k in dump}


    app.run(port=port, debug=True)