#!/usr/bin/env python
# coding: utf-8

import os
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from keras_image_helper import create_preprocessor

from flask import Flask
from flask import request
from flask import jsonify

from protobuf import np_to_protobuf


# Set the host configurable using environmental variable
host = os.getenv('TF_SERVING_HOST', 'localhost:8500')

# Create channel
channel = grpc.insecure_channel(host)
# Use channel to connect to prediction service
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel) # PredictionServiceStub is used to invoke localhost services
# Create input image preprocessor
preprocessor = create_preprocessor('xception', target_size=(229, 229))


# Function to prepare request
def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()

    pb_request.model_spec.name = 'clothing-model'
    pb_request.model_spec.signature_name = 'serving_default'
    pb_request.inputs['input_8'].CopyFrom(np_to_protobuf(X)) # convert numpy array to protobuf
    return pb_request


# List of class names
classes = [
    'dress',
    'hat',
    'longsleeve',
    'outwear',
    'pants',
    'shirt',
    'shoes',
    'shorts',
    'skirt',
    't-shirt'
]

# Function to prepare response
def prepare_response(pb_response):
    preds = pb_response.outputs['dense_7'].float_val
    return dict(zip(classes, preds))


# Function to make predictions from url
def predict(url):
    X = preprocessor.from_url(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    response = prepare_response(pb_response)
    return response

# Create flask app
app = Flask('gateway')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data['url']
    result = predict(url)
    return jsonify(result)


if __name__=='__main__':
    # # test locally without flask (with docker run)
    # img_url = 'http://bit.ly/mlbookcamp-pants'
    # response = predict(img_url)
    # print(response)
    # app.run(debug=True, host='0.0.0.0', port=9696)
