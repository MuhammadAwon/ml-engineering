# Tensorflow Serving base image
FROM tensorflow/serving:2.7.0

# Copy model to 'models' dir with version number
COPY clothing-model /models/clothing-model/1
# Provide enviromental variable
ENV MODEL_NAME='clothing-model'