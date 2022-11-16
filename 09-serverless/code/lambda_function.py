#!/usr/bin/env python
# coding: utf-8

# Import required libraries
import tflite_runtime.interpreter as tflite
from keras_image_helper import create_preprocessor


# Create preprocessor for the model we used for training (i.e., Xception)
preprocessor = create_preprocessor('xception', target_size=(299,299))


# Initalize interpreter
interpreter = tflite.Interpreter(model_path='clothing-model.tflite')
# Allocate memory
interpreter.allocate_tensors()
# Get input index
input_index = interpreter.get_input_details()[0]['index']
# Get output index
output_index = interpreter.get_output_details()[0]['index']


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

# # Image url
# url = 'http://bit.ly/mlbookcamp-pants'

def predict(url):
    # Preprocess the image from url
    X = preprocessor.from_url(url)

    # Make prediction on image
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)[0]

    # Convert numpy array predictions into float type
    # for conversion we need to convert array to python list first
    float_preds = preds.tolist()
    
    return dict(zip(classes, float_preds))

# Create lambda function
def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result
