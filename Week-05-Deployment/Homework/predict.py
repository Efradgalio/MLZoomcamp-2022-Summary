import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = f'model1.bin'
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('credit-card')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0,1]

    credit = y_pred >= 0.5

    result = {
        "credit_probability": float(y_pred),
        "credit" : bool(credit)
    }

    return jsonify(result)