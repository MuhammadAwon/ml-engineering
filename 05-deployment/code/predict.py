import pickle
from flask import Flask, request, jsonify


# Function to apply one-hot encoding feature to the customer data and make prediction

def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    
    return y_pred


# Load the saved model

model_file = 'model_C=1.0.bin' # saved model

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


# Create flask app

app = Flask('churn')


# Create POST request of make prediction on customer data

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json() # telling flask we are getting json object so turn it as python dictionary

    prediction = predict_single(customer, dv, model)
    churn = prediction >= 0.5

    # prepare our response to send back to the browser
    result = {
        'churn_probability': float(prediction),
        'churn': bool(churn),
    }

    return jsonify(result) # return result as json object


if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)