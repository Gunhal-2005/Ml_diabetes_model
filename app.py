from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('svc_gridsearch_best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Feature names (must match training order)
feature_names = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        input_df = pd.DataFrame([data], columns=feature_names)
        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        return jsonify({
            'prediction': int(prediction[0]),
            'prediction_probability_class_0': float(prediction_proba[0][0]),
            'prediction_probability_class_1': float(prediction_proba[0][1])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    print("Flask app running...")
    app.run(host='0.0.0.0', port=10000)
