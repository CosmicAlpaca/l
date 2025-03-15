from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS to allow Node.js front-end to communicate

# Load the trained model and preprocessing pipeline
model = load_model("model/diabetes_model.h5")
preprocessing_pipeline = joblib.load("model/preprocessing_pipeline.joblib")

# Define input columns
columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
           "DiabetesPedigreeFunction", "Age"]


@app.route('/api/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()
    input_data = [float(data[col]) for col in columns]

    # Create DataFrame
    input_df = pd.DataFrame([input_data], columns=columns)

    # Replace zeros with NaN
    cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    input_df[cols_with_zeros] = input_df[cols_with_zeros].replace(0, np.nan)

    # Preprocess input
    input_transformed = preprocessing_pipeline.transform(input_df)

    # Make prediction
    prediction = model.predict(input_transformed)
    probability = float(prediction[0][0])
    outcome = "Positive" if probability > 0.5 else "Negative"

    # Return JSON response
    return jsonify({
        "probability": round(probability * 100, 2),
        "outcome": outcome
    })


if __name__ == '__main__':
    app.run(port=5000, debug=True)