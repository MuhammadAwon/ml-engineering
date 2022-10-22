import numpy as np
import bentoml
from bentoml.io import NumpyNdarray


# Get sklearn model references and create runners
runner = bentoml.models.get('mlzoomcamp_homework:qtzdz3slg6mwwdu5').to_runner()
runner2 = bentoml.models.get('mlzoomcamp_homework:jsi67fslz6txydu5').to_runner()


# Create bentoml service
svc = bentoml.Service('cool_classifier', runners=[runner, runner2])

# Define endpoint
@svc.api(input=NumpyNdarray(shape=(-1, 4), enforce_shape=True), output=NumpyNdarray()) # numpy array as input and output
async def classify(vector):
    result = await runner.predict.async_run(vector)[0]
    print('RESULT:', result)
    return result
