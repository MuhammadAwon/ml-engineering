# 7. BentoML Production

## 7.1 Intro and Overview

The goal of this session is to build and deploy an ML service, customize our service to fit our use case, and make our service production-ready with the open-source library BentoML. For this, we'll be using the model we built in session 6.

What is production-ready?

- `Scalability`: it is the ability to increase or decrease the resources of the application according to the user demands.
- `Operationally efficiency`: it is being able to maintain the service by reducing the time, efforts and resources as much as possible without compromising the high-quality.
- `Repeatability (CI/CD)`: to update or modify the service as we need without having to do everything again from the scratch.
- `Fexibility`: to make it easy to react and apply changes to the issues in the production.
- `Resiliency`: to ensure even if the service is completely broke we are still able to reverse to the previous stable version.
- `Easy to use`: all the required frameworks should be easy to use.

We first focus should always be on getting the service to the production and rest will come later.

What is a BentoML? A typical machine learning application has various components. For instance, code, model(s), data, dependencies, configuration, deployment logic, and many more. BentoML packages all these componments into one deployable unit with ease.

What we will be convering in session 7?

- Building a prediction service
- Deploying our prediction service
- Anatomy of a BentoML service
- Customizing Bentos
- BentoML Production Deployment
- High Perfromance serving
- Custom Runner / Framework

## 7.2 Build Bento Service

We need BentoML model store to save an XGBoost model instance in the following way:

```python
# Save xgboost model
bentoml.xgboost.save_model('credit_risk_model',
                            model,
                            custom_objects={'DictVectorizer': dv})
```

BentoML will generator a directory in the home directory where the model will be kept.

Once the model is saved, we can create a `service.py` file that will be used to define the BentoML service:

```python
import bentoml
from bentoml.io import JSON


# Pull the model as model reference (it pulls all the associate metadata of the model)
model_ref = bentoml.xgboost.get('credit_risk_model:latest')
# Call DictVectorizer object using model reference
dv = model_ref.custom_objects['DictVectorizer']
# Create the model runner (it can also scale the model separately)
model_runner = model_ref.to_runner()

# Create the service 'credit_risk_classifier' and pass the model
svc = bentoml.Service('credit_risk_classifier', runners=[model_runner])


# Define an endpoint on the BentoML service
@svc.api(input=JSON(), output=JSON()) # decorate endpoint as in json format for input and output
def classify(application_data):
    # transform data from client using dictvectorizer
    vector = dv.transform(application_data)
    # make predictions using 'runner.predict.run(input)' instead of 'model.predict'
    prediction = model_runner.predict.run(vector)
    
    result = prediction[0] # extract prediction from 1D array
    print('Prediction:', result)

    if result > 0.5:
        return {'Status': 'DECLINED'}
    elif result > 0.3:
        return {'Status': 'MAYBE'}
    else:
        return {'Status': 'APPROVED'}
```

Once the service and the endpoint is created we can run the app using the command: `bentoml serve service:svc`, where ***service*** is the script and ***svc*** is the service name.

*There are some key things to consider at the time of bentoml version 1.0.7*:

- The final model has to be trained without providing feature names in the `DMatrix` constructor to prevent "ValueError" while building the BentoML application.
- When running bento service instead of using `http://0.0.0.0:3000/`, use `http://localhost:3000/`.
- On Windows systems `bentoml serve service:svc --reload` is not working. The issue has been added [here](https://github.com/bentoml/BentoML/issues/3111).

## 7.3 Deploy Bento Service

In this section we are going to look at BentoML cli and what operations BentoML is performing behind the scenes.

We can get a list of saved model in the terminal using the commmand `bentoml models list`. This command shows all the saved models and their tags, module, size, and the time they were created at. For instance:

```bash
 Tag                           Module           Size        Creation Time
 credit_risk_model:l652ugcqk…  bentoml.xgboost  197.77 KiB  2022-10-20 08:29:54
```

We can use `bentoml models list -o json|yaml|table` to display the output in one of the given format.

Running the command `bentoml models get credit_risk_model:l652ugcqkgefhd7k` displays the information about the model which looks like:

```yaml
name: credit_risk_model
version: l652ugcqkgefhd7k
module: bentoml.xgboost
labels: {}
options:
  model_class: Booster
metadata: {}
context:
  framework_name: xgboost
  framework_versions:
    xgboost: 1.6.2
  bentoml_version: 1.0.7
  python_version: 3.10.6
signatures:
  predict:
    batchable: false
api_version: v2
creation_time: '2022-10-20T08:29:54.706593+00:00'
```

Important thing to note here is that the version of the XGBoost in the `framework_versions` has to be same as the model was trained with otherwise we might get inconsistent results. The BentoML pulls these dependencies automatically and generates this file for convenience.

The next we want to do is, creating the file `bentofile.yaml`:

```yaml
service: "service.py:svc" # Specify entrypoint and service name
labels: # Labels related to the project for reminder (the provided labels are just for example)
  owner: bentoml-team
  project: gallery
include:
- "*.py" # A pattern for matching which files to include in the bento build
python:
  packages: # Additional pip packages required by the service
    - xgboost
    - sklearn
```

Once we have our `service.py` and `bentofile.yaml` files ready we can build the bento by running the command `bentoml build`. It will look in the service.py file to get all models being used and into bentofile.yaml file to get all the dependencies and creates one single deployable directory for us. The output will look something like this:

```bash
Successfully built Bento(tag="credit_risk_classifier:kdelkqsqms4i2b6d")
```

We can look into this directory by locating `cd ~/bentoml/bentos/credit_risk_classifier/kdelkqsqms4i2b6d/` and the file structure may look like this:

```bash
.
├── README.md # readme file
├── apis
│   └── openapi.yaml # openapi file to enable Swagger UI
├── bento.yaml # bento file to bind everything together
├── env # environment related directory
│   ├── docker # auto generate dockerfile (also can be customized)
│   │   ├── Dockerfile
│   │   └── entrypoint.sh
│   └── python # requirments for installation
│       ├── install.sh
│       ├── requirements.txt
│       └── version.txt
├── models # trained model(s)
│   └── credit_risk_model
│       ├── l652ugcqkgefhd7k
│       │   ├── custom_objects.pkl # custom objects (in our case DictVectorizer)
│       │   ├── model.yaml # model metadate
│       │   └── saved_model.ubj # saved model
│       └── latest
└── src
    └── service.py # bentoml service file for endpoint
```

The idea behind the structure like this is to provide standardized way that a machine learning service might required.

Now the last thing we need to do is to build the docker image. This can be done with `bentoml containerize credit_risk_classifier:kdelkqsqms4i2b6d`.

> Note: We need to have Docker installed before running this command.

Once the docker image is built successfully, we can run `docker run -it --rm -p 3000:3000 containerize credit_risk_classifier:kdelkqsqms4i2b6d` to see if everything is working as expected. We are exposing 3000 port to map with the service port which is also 3000 and this should take us to Swagger UI page again.

**Reference**:

- [Tutorial: Intro to BentoML](https://docs.bentoml.org/en/latest/tutorial.html)

## 7.4 Sending, Receiving and Validating Data

Data validation is another great feature on BentoML that ensures the data transferation is valid and reliable. We can integrate Python library Pydatic with BentoML for this purpose.

Pydantic can be installed with `pip install pydantic`, after that we need to import the `BaseModel` class from the library and create our custom class for data validation:

```python
# Create pydantic base class to create data schema for validation
class CreditApplication(BaseModel):
    seniority: int
    home: str
    time: int
    age: int
    marital: str
    records: str
    job: str
    expenses: int
    income: float
    assets: float
    debt: float
    amount: int
    price: int
```

Our model is trained on 13 features of different data types and the BaseModel will ensure that we are always recieving them for the model prediction.

Next we need to implement pass the class in our bentoml service:

```python
# Pass pydantic class in the application
@svc.api(input=JSON(pydantic_model=CreditApplication), output=JSON()) # decorate endpoint as in json format for input and output
def classify(credit_application):
    # transform pydantic class to dict to extract key-value pairs 
    application = credit_application.dict()
    # transform data from client using dictvectorizer
    vector = dv.transform(application)
    # make predictions using 'runner.predict.run(input)' instead of 'model.predict'
    prediction = model_runner.predict.run(vector) 
```

Along the `JSON()`, BentoML uses various other descriptors in the input and output specification of the service api, for example, NumpyNdarray(), PandasDataFrame(), Text(), and many more.

**Reference**:

- [Pydantic Manual](https://pydantic-docs.helpmanual.io/)
- [BentoML API IO Descriptors](https://docs.bentoml.org/en/latest/reference/api_io_descriptors.html)

## 7.5 High Performance Model Serving

BentoML can optimize the performance on our application where the model will have to make predictions on hundreds of requests per seconds. For this we need to install locust (`pip install locust`), which is a Python open-source library for load testing.

Once the locust is installed, we'll need to create `locustfile.py` and implement user flows for testing:

```python
import numpy as np
from locust import task
from locust import between
from locust import HttpUser


# Sample data to send
sample = {"seniority": 3,
 "home": "owner",
 "time": 36,
 "age": 26,
 "marital": "single",
 "records": "no",
 "job": "freelance",
 "expenses": 35,
 "income": 0.0,
 "assets": 60000.0,
 "debt": 3000.0,
 "amount": 800,
 "price": 1000
 }

# Inherit HttpUser object from locust
class CreditRiskTestUser(HttpUser):
    """
    Usage:
        Start locust load testing client with:
            locust -H http://localhost:3000, in case if all requests failed then load client with:
            locust -H http://localhost:3000 -f locustfile.py

        Open browser at http://0.0.0.0:8089, adjust desired number of users and spawn
        rate for the load test from the Web UI and start swarming.
    """

    # create mathod with task decorator to send request
    @task
    def classify(self):
        self.client.post("/classify", json=sample) # post request in json format with the endpoint 'classify'

    wait_time = between(0.01, 2) # set random wait time between 0.01-2 secs
```

This first optimization we can implement in our application is called *async* optimization. This will make the application to process the requests in parallel and the model will make predictions simultaneously:

```python
# Define an endpoint on the BentoML service
# pass pydantic class application
@svc.api(input=JSON(pydantic_model=CreditApplication), output=JSON()) # decorate endpoint as in json format for input and output
async def classify(credit_application): # parallelized requests at endpoint level (async)
    # transform pydantic class to dict to extract key-value pairs 
    application = credit_application.dict()
    # transform data from client using dictvectorizer
    vector = dv.transform(application)
    # make predictions using 'runner.predict.run(input)' instead of 'model.predict'
    prediction = await model_runner.predict.async_run(vector) # bentoml inference level parallelization (async_run)
```

Another optimization is to take advantage of micro-batching. This is another BentoML feature where it can combine the data coming from multiple users and combine them into **one array**, and then this array will be batched into smaller batches when the model prediction is called. There are few steps we need to do to enable this functionality, the first thing we have to save the model with bentoml `signatures` feature:

```python
# Save the model batchable settings for production efficiency
bentoml.xgboost.save_model('credit_risk_model',
                            model,
                            custom_objects={'DictVectorizer': dv},
                           signatures={  # model signatures for runner inference
                               'predict': { 
                                   'batchable': True, 
                                   'batch_dim': 0 # '0' means bentoml will concatenate request arrays by first dimension
                               }
                           })
```

Running `bentoml serve --production` will make the batchable model in serving, the `--production` flag will enable more than one process for our web workers.

We can also configure the batching parameters of the runner by creating `bentoconfiguration.yaml` file:

```python
# Config file controls the attributes of the runner
runners:
  batching:
    enabled: true
    max_batch_size: 100
    max_latency_ms: 500
```

> Note: In general, we are not supposed to be running the traffic generator on same machine that is serving the application requests because that takes away the CPU from the requests server.

**Resources**:

- [Locust documentation](https://locust.io/)
- [BentoML Adaptive Batching](https://docs.bentoml.org/en/latest/guides/batching.html)
- [BentoML Runners Usage](https://docs.bentoml.org/en/latest/concepts/runner.html)

## 7.6 Bento Production Deployment

In this lession we'll deploy our model to Amazon ECS (Elastic Container Service). This service makes it easy for us to deploy and scale our containerized applications.

First thing we need is to get our model tag that we want to used to create docker image, or we can simply run `bentoml build` to get the latest tag. Then we will create docker container using `bentoml containerize model:tag --platform=linux/amd64`, the option --plaform=linux/amd64 will prevent us from getting any deployment issues on ECS.

Once the container is built, we need to setup ECR container repository where we can store the docker image:

- Create Identity and Access Management (IAM) user.
- Get the Security credentials (if don't have already):
  - From top right drop-down menu > Security credentials > Access keys
- Install AWS CLI.
- Connect AWS with local machine by running `aws configure` command to provide credentials.
- Create Amazon Elastic Container Registry (ECR):
  - Click Get Started > General settings > Create repository
- Authenticate and push docker image to ECR:
  - Click on the repo name > View push commands > follow the instructions and tag the docker image built with bentoml (skip the step 2 because we have already built the docker image).

Now we need to setup Amazon Elastic Container Service (ECS) to run our docker image:

- Search and click Elastic Container Service in the search bar.
- Create and Configure Cluster:
  - Click Create Cluster > Networking only (CPU only) > follow the instructions.
- Create Task Definitions:
  - Click Create new Task Definition > FARGRATE > Task memory (0.5GB), Task CPU (0.25 vCPU) > Add container (follow instructions and paste the image URI of ECR repo into the *Image* section, also increase the *Soft limit* to 256 and set *Port mappings* to 3000) > create task
- Run the Task:
  - From ECS dashboard > Clusters > select the cluster we just created > Tasks > Run new Task > follow instructions (also select *Launch type* FARGATE, *Operating system family* Linux, *Security groups* set custom tcp to 3000) > create Task

Once the Task is running we can click on it to see all of the information including the *Public IP* which we can entry in the browser to access the service.

If we want to share the model or saving it to cloud, we can do so with `bentoml export model:tag path_to_store/modelname.bento` command and with this we can save the model in a local or push the model on save in the cloud (e.g., on Amazon S3 bucket). Beside the native .bento format, we can also save the model in `('tar')`, `tar.gz ('gz')`, `tar.xz ('xz')`, `tar.bz2 ('bz2')`, and `zip`.

In addition we can also import bentoml models from cloud or other sources using `bentoml import path_to_access_/model_name.bento`.

**References**:

- [ML Bookcamp article to Get Started with AWS and Creating IAM ROle](https://mlbookcamp.com/article/aws)
- [AWS CLI installation instructions](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)

## 7.7 (Optional) Advanced Example: Deploying Stable Diffusion Model

This lesson is about deploying an open-source model on Amazon Elastic Compute Cloud (EC2). The model is called Stable Diffusion which takes prompt as text and/or image from the user and transform it into an image.

BentoML has a nice user-friendly instructions on their GitHub page to use the model and deploy it on EC2.

**Link**: [Serving Stable Diffusion with BentoML](https://github.com/bentoml/stable-diffusion-bentoml)

## 7.8 Summary

Here's the summary of what we have gone through in the session 7:

- Building a prediction service
- Deploying our prediction service
- Sending, Receiving and Validating Data
- High performance serving
- Bento production deployment
- Advanced example: Deploying Stable Diffusion

**Link**: [BentoML GitHub page](https://github.com/bentoml/BentoML)
