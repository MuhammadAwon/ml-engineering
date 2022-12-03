import requests

# url = 'http://localhost:9696/predict' # test with port 9696
# url = 'http://localhost:8080/predict' # test load balancer service on port 8080

# Test with remote url
url = 'http://af4bcd632235d4c7c879e73c10a48dc2-2014733034.ap-south-1.elb.amazonaws.com/predict'

data = {'url': 'http://bit.ly/mlbookcamp-pants'}

result = requests.post(url, json=data).json()
print(result)