import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Some utilites
from util import base64_to_pil
import numpy as np

from neural_network import *

# Declare a flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'model/uploaded_images'

# Load model
neural = NeuralNetwork()


print('Model loaded. Check http://127.0.0.1:5000/')


test_img = 'model/uploaded_images/test.jpg'
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = base64_to_pil(request.json)
        print(img.filename)
        path = 'model/uploaded_images/test2.jpg'
        img.save(path)
        pred_proba = 1
        result2 = neural.predict_breed(path)

        #print(result2)
        # Serialize the result, you can add additional fields
        return jsonify(result=result2, probability=pred_proba)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
