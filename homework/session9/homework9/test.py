import requests


# Local host url 
url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# Test image prediction locally with docker
data = {'url': 'https://upload.wikimedia.org/wikipedia/en/e/e9/GodzillaEncounterModel.jpg'}


result = requests.post(url, json=data).json()
print(result)
