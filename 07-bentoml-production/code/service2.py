# This script is used for developement

import bentoml
from pydantic import BaseModel
from bentoml.io import JSON


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

# # Pull the model as model reference (it pulls all the associate mata data of the model)
# model_ref = bentoml.xgboost.get('credit_risk_model:latest') # unbatchable model

# Specify 'batchable' model to run
model_ref = bentoml.xgboost.get('credit_risk_model:ggtmwrcrfw5ccp44')

# Call DictVectorizer object using model reference
dv = model_ref.custom_objects['DictVectorizer']

# Create the model runner (it can also scale the model separately)
model_runner = model_ref.to_runner()

# Create the service 'credit_risk_classifier' and pass the model
svc = bentoml.Service('credit_risk_classifier', runners=[model_runner])


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
    
    result = prediction[0] # extract prediction from 1D array
    print('Prediction:', result)

    if result > 0.5:
        return {'Status': 'DECLINED'}
    elif result > 0.3:
        return {'Status': 'MAYBE'}
    else:
        return {'Status': 'APPROVED'}

