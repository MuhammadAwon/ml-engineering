# 7. BentoML Production

## 7.1 Intro and Overview

The goal of this session is to build and deploy an ML service, customize our service to fit our use case, and make our service production ready with the open-source library BentoML. For this, we'll be using the model we built in session 6.

What is production ready?

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


# Pull the model as model reference (it pulls all the associate mata data of the model)
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

