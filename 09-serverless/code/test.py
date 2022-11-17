import requests


# # Url to send request
# url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# API gateway url
url = 'https://01coolapi10.execute-api.ap-south-1.amazonaws.com/test/predict'

# Data to send for request
data = {'url': 'http://bit.ly/mlbookcamp-pants'}

# Convert POST request of the data to JSON
result = requests.post(url, json=data).json()
print(result)