import pickle
from flask import Flask, request, jsonify


# # Client data

# client = {"reports": 0, "share": 0.001694,
#           "expenditure": 0.12, "owner": "yes"}

# File paths

model_path = './model1.bin'
dv_path = './dv.bin'


# Load files as binary

with open(model_path, 'rb') as f_model, \
    open(dv_path, 'rb') as f_dv:
    model = pickle.load(f_model)
    dv = pickle.load(f_dv)


# Function to make single prediction

def predict_single(client, dv, model):
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]

    return y_pred


# Create flask app

app = Flask('credit-card')


# Create POST request to make prediction on client data

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    prediction = predict_single(client, dv, model)

    # prepare response to send back to the browser
    result = float(round(prediction, 3))

    return jsonify(result)


if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)