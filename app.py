import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST' , 'GET'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    final_features = request.args.get('text')

    prediction = model.predict(final_features)



    return render_template('index.html', Class=Class, prediction_text='URL is {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
