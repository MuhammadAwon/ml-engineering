import requests


# Url to send request
url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# Data to send for request
data = {'url': 'http://bit.ly/mlbookcamp-pants'}

# Convert POST request of the data to JSON
result = requests.post(url, json=data).json()
print(result)