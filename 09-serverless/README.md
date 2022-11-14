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

**Reference**:

- [TensorFlow Lite documentation](https://www.tensorflow.org/lite)