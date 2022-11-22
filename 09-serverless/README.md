# 9. Severless Deep Learning

## 9.1 Introduction to Severless

In this session we'll talk about deploying deep learning model on cloud using AWS Lambda and TensorFlow Lite. The model we'll use from the session 8 to classify the clothings on the website.

## 9.2 AWS Lambda

AWS Lambda is a severless compute service that runs our code in response to events and automatically manages the underlying compute resources for us for serving the model. The other benefit os Lambda is that we pay only as per request that means when there is no request we are not getting charged on anything.

TensorFlow is too big for deployment therefore we'll use TensorFlow Lite instead.

Following are the steps to create Lamda funtion:

- After login to AWS > select/search Lambda > from desktop select Create function
- Select Author from scratch > assign Function name, Runtime, and Architecture > Create function
- From dropdown menu next to `Test` button > select `Configure test event` > assign Event name, Event JSON > Save
- Select `deploy` (if changes are made in lambda_function.py) > Select `Test` to send the request

There are certain number of Lambda requests comes under free tier.

**Reference**:

- [More on AWS Free Tier plan can be found here](https://aws.amazon.com/free/?all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc&awsf.Free%20Tier%20Types=*all&awsf.Free%20Tier%20Categories=*all)

## 9.3 TensorFlow Lite

TensorFlow is a large library of 1.2gb size, due to such larget volume it is not feasible to deploy TensorFlow model in inference. There are also other mulitple reason we want to use TensoFlow Lite instead of usually TensorFlow:

- AWS Lambda limits 50mb per file (zip) and upto 10gb for docker image
- Large images are problematic:
  - it costs more cause of larger storage requirement
  - revoking lambda for the first time takes some time to initialize the function and with larger image it becomes even slower
  - TensorFlow is such a larger library and it needs to import many dependencies which also make the application slow

The solution to this is not to use TensorFlow but use lighter version of the libray which is TensorFlow Lite. TensorFlow Lite is not used to training the models and it only focuses on inference (i.e, model.predict(X)). To be able to use TensorFlow model we need to convert it to TensorFlow Lite.

To preprocess images and run the model without TensorFlow, we need to install the following to libraries:

- For image preprocess: `pip install keras-image-helper`
- TFlite runtime: `pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime`

**Reference**:

- [TensorFlow Lite documentation](https://www.tensorflow.org/lite)

## 9.4 Preparing the Code for Lambda

To use the code in deployment we need to convert jupyter notebook into python script, which can be done using the command `jupyter nbconvert --to script notebook_name.ipynb`.

Then we need to create prediction function and lambda_handler for lambda function.

## 9.5 Preparing a Docker Image

To build docker image for our application we need to prepare the `Dockerfile`:

```docker
FROM public.ecr.aws/lambda/python:3.9

RUN pip install keras-image-helper
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl

COPY clothing-model.tflite .
COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]
```

There are few things we need to take care of in the Dockerfile, the lambda base image is extracted from AWS ECR Public Gallery which contains the Amazon Linux Base operating system, whereas we built our application on Ubuntu and that's why we'll have runtime conflict cause of different OS architecture.

Since we are using Amazon Linux OS, we need to install different wheel python for tflite as well.

Before running the script to test the docker container we need to convert the prediction from numpy array to `float` data type in `lambda_function.py`:

```python
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
```

`test.py` contains the script to test the docker container:

```python
import requests


# Url to send request
url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# Data to send for request
data = {'url': 'http://bit.ly/mlbookcamp-pants'}

# Convert POST request of the data to JSON
result = requests.post(url, json=data).json()
print(result)
```

Now we can build the docker image using the command `docker build -t clothing-model .` and run the docker container with `docker run -it --rm -p 8080:8080 clothing-model:latest`. If everything is configured properly, we should see the predictions in the terminal with the command `python test.py`.

**Reference**:

- Download AWS lambda base image: https://gallery.ecr.aws/lambda/python
- Install tflite for AWS lambda: https://github.com/alexeygrigorev/tflite-aws-lambda/tree/main/tflite

## 9.6 Creating the Lambda Function

In order to deploy the docker container as lambda function we need to following these steps:

- **Publishing the image to AWS ECR**

  - Install awscli (`pip install awscli`) and configure with the credentials
  - Create ECR repository with the command `aws ecr create-repository --repository-name clothing-tflite-images` and copy the Repository URI address (need it for later)
  - Create a script of variables from Repository URI:
    - ```bash
      ACCOUNT=2278222782
      REGION=ap-south-1
      REGISTRY=clothing-tflite-images
      PREFIX=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY}

      TAG=clothing-model-xception-v4-001
      REMOTE_URI=${PREFIX}:${TAG}
      ```
  - Log in the registry using `aws ecr get-login --no-include-email | sed 's/[0-9a-zA-z=]\{20,\}/PASSWORD/g'` (hiding the password used sed and regex)
  - To be able to login the registry and to push the docker image, execute the output of the above command with `$(aws ecr get-login --no-include-email)`
  - Tag the docker image with URI `docker tag clothing-model:latest ${REMOTE_URI}`
  - Push docker image to ECR `docker push ${REMOTE_URI}`

- **Creating the Lambda Function**
  
  - Search for AWS Lambda from the sreach bar and select Create function
  - Select Container image > Function name > Container image URI using the Browse images (select the docker image) > Create function
  - Select Configuration > Increase Memory to 1024 and Timeout to 30 sec > Save
  - Select Test to test the function > Add image url in the body (JSON) > Test

**Reference**:

- AWS ECR to create container registry: https://aws.amazon.com/ecr/
- Create Lambda function: https://aws.amazon.com/lambda/
- AWS Lambda pricing: https://aws.amazon.com/lambda/pricing/

## 9.7 API Gateway: Exposing the Lambda Function

We create the lambda function for our docker image, now we need to expose it using AWS API Gateway:

- Search for api gateway > Create API > Select REST API > Settings (API name) > Create API
- Select Create Resource from the drop-down of Actions button > Resource Name (e.g., predict, this will be the endpoint of the url) > Create Resource
- Select Create Method from Actions > Select POST from drop-down of /predict > Select ok and then POST > Integration type (Lambda Function), Lambda Region, and Lambda Function name > Save
- Click on TEST > Request Body (url of the image for prediction) > Hit Test
- Select Deploy API from Actions > Deployment stage (New Stage), Stage name (test) > Deploy
- Copy the Invoke URL and past in the `test.py`, be sure to add /predict at the end of the url, e.g., `https://01coolapi10.execute-api.ap-south-1.amazonaws.com/test/predict`

**Reference**:

- Create AWS API Gateway: https://aws.amazon.com/api-gateway/

## 9.8 Summary

- AWS Lambda is way of deploying models without having to worry about servers
- TensorFlow Lite is a lightweight alternative to TensorFlow that only focuses on inference
- To deploy our code, package it in a Docker container
- Expose the lambda function via API Gateway

## 9.9 Explore More

- Try similar serverless services from Google Cloud and Microsoft Azure
- Deploy cats vs dogs and other Keras models with AWS Lambda
- AWS Lambda is also good for other libraries, not just TensorFlow. You can deploy Scikit-Learn and XGBoost models with it as well
