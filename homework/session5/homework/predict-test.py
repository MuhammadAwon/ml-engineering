import requests


# Local host url

url = 'http://localhost:9696/predict'


# Client data to make prediction

client = {"reports": 0, "share": 0.245,
          "expenditure": 3.438, "owner": "yes"}


# Send POST request to the body of the json response

response = requests.post(url, json=client).json()

# Make credit card decision
credit_card_decision = response >= 0.5


# Get the probability and eligibility of the client

if (credit_card_decision >= 0.5) == True:
    print(f'Client probability is {response} and eligible for credit card!!')
else:
    print(f'Client probability is {response} and it is below credit card eligible!!')
