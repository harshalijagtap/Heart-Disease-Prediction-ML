from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

FEATURE_COLUMNS = None
with open('feature_columns.pkl', 'rb') as f:
    FEATURE_COLUMNS = pickle.load(f)

COLUMNS_TO_SCALE = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        age        = float(data['age'])
        sex        = int(data['sex'])           # 0=Female, 1=Male
        cp         = int(data['cp'])            # 0-3
        trestbps   = float(data['trestbps'])
        chol       = float(data['chol'])
        fbs        = int(data['fbs'])           # 0 or 1
        restecg    = int(data['restecg'])       # 0-2
        thalach    = float(data['thalach'])
        exang      = int(data['exang'])         # 0 or 1
        oldpeak    = float(data['oldpeak'])
        slope      = int(data['slope'])         # 0-2
        ca         = int(data['ca'])            # 0-4
        thal       = int(data['thal'])          # 0-3

        # Build a base row dict with all dummy columns set to 0
        row = {col: 0 for col in FEATURE_COLUMNS}

        # Fill continuous features (will be scaled)
        row['age']      = age
        row['trestbps'] = trestbps
        row['chol']     = chol
        row['thalach']  = thalach
        row['oldpeak']  = oldpeak

        # One-hot dummies
        def set_dummy(prefix, value):
            key = f"{prefix}_{value}"
            if key in row:
                row[key] = 1

        set_dummy('sex', sex)
        set_dummy('cp', cp)
        set_dummy('fbs', fbs)
        set_dummy('restecg', restecg)
        set_dummy('exang', exang)
        set_dummy('slope', slope)
        set_dummy('ca', ca)
        set_dummy('thal', thal)

        # Build dataframe-like array in correct column order
        import pandas as pd
        df_row = pd.DataFrame([row], columns=FEATURE_COLUMNS)

        # Scale continuous columns
        df_row[COLUMNS_TO_SCALE] = scaler.transform(df_row[COLUMNS_TO_SCALE])

        X = df_row.values

        prediction = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        confidence = float(round(max(proba) * 100, 2))
        risk_score = float(round(proba[1] * 100, 2))  # probability of heart disease

        # Risk level
        if risk_score < 30:
            risk_level = "Low"
        elif risk_score < 60:
            risk_level = "Moderate"
        else:
            risk_level = "High"

        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'no_disease_prob': float(round(proba[0] * 100, 2)),
            'disease_prob': float(round(proba[1] * 100, 2)),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
