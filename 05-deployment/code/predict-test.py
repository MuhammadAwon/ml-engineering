from urllib import response
import requests


# Local host url

url = 'http://localhost:9696/predict'


# Customer data in dictionary format

customer_id = 'xyz-123' # dummy customer id

customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


# Send post request and see the body of the response using 'json()' function

response = requests.post(url, json=customer).json()
print(response)


# Decide to send email or not based on churn prediction

if response['churn'] == True:
    print(f'Sending promo email to {customer_id}')
else:
    print(f'Not sending promo email to {customer_id}')