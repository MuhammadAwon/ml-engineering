# This script is for deployment
# credit_risk_classifier:lzbdkucrgwe4my5i (bento production model)

import bentoml
from bentoml.io import JSON


model_ref = bentoml.xgboost.get('credit_risk_model:ggtmwrcrfw5ccp44')
dv = model_ref.custom_objects['DictVectorizer']
model_runner = model_ref.to_runner()

svc = bentoml.Service('credit_risk_classifier', runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
async def classify(application_data):
    vector = dv.transform(application_data)
    prediction = await model_runner.predict.async_run(vector)
    
    result = prediction[0]
    print('Prediction:', result)

    if result > 0.5:
        return {'Status': 'DECLINED'}
    elif result > 0.3:
        return {'Status': 'MAYBE'}
    else:
        return {'Status': 'APPROVED'}