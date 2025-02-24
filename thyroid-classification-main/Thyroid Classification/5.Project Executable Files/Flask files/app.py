from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import logging

# Load the saved model and label encoder
model_path = 'random_forest_model.pkl'
label_encoder_path = 'label_encoder.pkl'

rf_model = joblib.load(model_path)
label_encoder = joblib.load(label_encoder_path)

# Diagnosis labels
diagnosis_labels = {
    'A': 'hyperthyroid',
    'B': 'T3 toxic',
    'C': 'toxic goitre',
    'D': 'secondary toxic',
    'E': 'hypothyroid',
    'F': 'primary hypothyroid',
    'G': 'compensated hypothyroid',
    'H': 'secondary hypothyroid',
    'I': 'increased binding protein',
    'J': 'decreased binding protein',
    'K': 'concurrent non-thyroidal illness',
    'L': 'consistent with replacement therapy',
    'M': 'underreplaced',
    'N': 'overreplaced',
    'O': 'antithyroid drugs',
    'P': 'I131 treatment',
    'Q': 'surgery',
    'R': 'discordant assay results',
    'S': 'elevated TBG',
    'T': 'elevated thyroid hormones',
    '-': 'no condition'
}

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON request data
        data = request.json
        logging.debug(f"Received data: {data}")

        # Extract the features from the request
        goitre = int(data.get('goitre', 0))
        tumor = int(data.get('tumor', 0))
        hypopituitary = int(data.get('hypopituitary', 0))
        psych = int(data.get('psych', 0))
        TSH = float(data.get('TSH', 0))
        T3 = float(data.get('T3', 0))
        TT4 = float(data.get('TT4', 0))
        T4U = float(data.get('T4U', 0))
        FTI = float(data.get('FTI', 0))
        TBG = float(data.get('TBG', 0))

        # Example of additional features that might be required
        # Ensure these features match the model's requirements
        additional_features = [0] * 18  # Replace with actual features as needed

        # Combine features into a single array
        features = np.array([[goitre, tumor, hypopituitary, psych, TSH, T3, TT4, T4U, FTI, TBG] + additional_features])
        logging.debug(f"Features: {features}")

        # Make a prediction using the Random Forest model
        prediction = rf_model.predict(features)
        predicted_label = label_encoder.inverse_transform(prediction)[0]
        predicted_diagnosis = diagnosis_labels.get(predicted_label, 'Unknown')
        logging.debug(f"Prediction: {predicted_diagnosis}")

        # Return the predicted diagnosis as a JSON response
        return jsonify({'predicted_diagnosis': predicted_diagnosis})
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
