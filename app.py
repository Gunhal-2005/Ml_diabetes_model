from flask import Flask, request, jsonify
import joblib
import pandas as pd

import joblib

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as scaler.pkl")

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('svc_gridsearch_best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the feature names in the correct order as used during training
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)

        # Convert input data to a pandas DataFrame
        # Ensure the order of columns matches the training data
        input_df = pd.DataFrame([data], columns=feature_names)

        # Scale the input features
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        # Return the prediction as JSON
        return jsonify({
            'prediction': int(prediction[0]),
            'prediction_probability_class_0': prediction_proba[0][0],
            'prediction_probability_class_1': prediction_proba[0][1]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400
      
     if __name__ == '__main__':
    print("Flask app running...")
    app.run(debug=True)
