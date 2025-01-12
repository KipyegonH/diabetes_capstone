import pickle
import xgboost as xgb
from flask import Flask, request, jsonify

# Load the DictVectorizer and model
with open('dv.pkl', 'rb') as f_in:
    dv = pickle.load(f_in)

with open('xgb_model.pkl', 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('diabetes')

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()

    X = dv.transform([patient])

    dmatrix = xgb.DMatrix(X, feature_names=list(dv.get_feature_names_out()))

    y_pred = model.predict(dmatrix)[0]
    diabetic = y_pred >= 0.5

    result = {
        'diabeties_probability': float(y_pred),
        'diabetic': bool(diabetic)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
