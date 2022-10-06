import pickle


model_path = './model1.bin'
dv_path = './dv.bin'

with open(model_path, 'rb') as f_model, \
    open(dv_path, 'rb') as f_dv:
    model = pickle.load(f_model)
    dv = pickle.load(f_dv)

def predict_single(customer, dv, model):
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]

    return y_pred

def predict(customer):
    prediction = predict_single(customer, dv, model)
    card_decision = prediction >= 0.5

    result = {
        'credit_card_probability': round(prediction, 3),
        'credit_card_decision': card_decision
    }

    return result

customer = {"reports": 0, "share": 0.001694,
            "expenditure": 0.12, "owner": "yes"}


credit_card_decision = predict(customer)

print(credit_card_decision)