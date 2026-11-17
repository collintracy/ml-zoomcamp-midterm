import pickle

import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify

from train import prepare_features


model_file = 'model.bin'

with open(model_file, 'rb') as f_in:
    pipeline, groups = pickle.load(f_in)

app = Flask('adoption')


@app.route('/predict', methods=['POST'])
def predict():
    animal = request.get_json()

    features = prepare_features(pd.DataFrame([animal]), groups)
    y_pred = pipeline.predict_proba(features.to_dict(orient='records'))[0, 1]

    result = {
        'adoption_probability': float(y_pred),
        'adopted': bool(y_pred >= 0.5),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
