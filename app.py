import pickle
from flask import Flask, request, app, jsonify, url_for, render_template

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
reg_model = pickle.load(open("regmodel.pkl", 'rb'))
scaler = pickle.load(open("standard_scaler_pickle.pkl", 'rb')) 
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST'])

def predict():

    # capture the data on the prediction
    data = request.json['data'] 
    data_instance = np.array(list(data.values())).reshape(1,-1)
    data_instance = scaler.transform(data_instance)
    pred_val = reg_model.predict(data_instance)
    print(pred_val[0])
    return jsonify(pred_val[0])


if __name__ == "__main__":
    app.run(debug = True)

