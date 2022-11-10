# Basic script for bentoml integration

import bentoml
from bentoml.io import JSON

model_ref = bentoml.xgboost.get("credit_risk_model:latest")
dv = model_ref.custom_objects['dictVectorizer']

print('model_ref.custom_objects') 


model_runner = model_ref.to_runner()

svc = bentoml.Service("credit_risk_classifier", runners=[model_runner])

@svc.api(input=JSON(),output=JSON())

def classify(application_data):
    vector = dv.transform(application_data)
    prediction = model_runner.predict.run(vector)
    print(prediction)

    result = prediction[0]

    if result > 0.5:
        return {"status": "DECLINED"}
    elif result > 0.23:
        return{"status":"MAYBE"}
    else:        
        return {"status": "Approved"}