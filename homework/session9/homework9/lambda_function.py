# Import required libraries
import os
import numpy as np
import tflite_runtime.interpreter as tflite

from io import BytesIO
from urllib import request
from PIL import Image
from PIL.Image import Resampling


# Get model path as environment variable
MODEL_NAME = os.getenv('MODEL_NAME', 'dino-dragon-model.tflite')
# Image size
IMG_SIZE = (150, 150)


# Function to download image and read as PIL image file
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

# Function to convert image to color image (if not) and resize image
def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, resample=Resampling.NEAREST)
    return img

# Function to rescale image input
def rescale_img(x):
    return x / 255.0


# Instantiate tflite model using interpreter class
interpreter = tflite.Interpreter(model_path=MODEL_NAME)
# Load the weights from the model to memory
interpreter.allocate_tensors()

# Get the input and output index from the interpreter
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


# Function to make image prdiction from url
def predict(url):
    # Load image from url
    img = download_image(url)
    # Prepare image and resize it
    img = prepare_image(img, target_size=IMG_SIZE)

    
    # Convert image to numpy array
    x = np.array(img, dtype='float32')
    # Add batch dimension
    X = np.array([x])
    # Rescale image
    X = rescale_img(X)

    # Set image tensor and invoke interpreter
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    # Make prediction
    pred = interpreter.get_tensor(output_index)[0][0]
    
    return float(pred)

# Create lambda function
def lambda_handler(event, context=None):
    url = event['url']
    pred = predict(url)
    result = {'prediction': pred}

    return result
