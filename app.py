from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

# Load the saved model and scaler
rf_classifier = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# Initialize Flask app
app = Flask(__name__)

# Home route to render HTML
@app.route('/')
def index():
    return render_template('index.html')

# Predict route to get crop prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array([[ 
        float(data['N']), float(data['P']), float(data['K']),
        float(data['temperature']), float(data['humidity']),
        float(data['ph']), float(data['rainfall'])
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = rf_classifier.predict(input_scaled)[0]
    
    return jsonify({'crop': prediction})

if __name__ == '__main__':
    app.run(debug=True)
