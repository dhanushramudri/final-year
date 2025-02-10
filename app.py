from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# Initialize Flask app
app = Flask(__name__)

# Home route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Predict route to get crop prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON data from the request
    
    # Extract values from the incoming JSON data
    input_data = np.array([[ 
        float(data['N']), float(data['P']), float(data['K']),
        float(data['temperature']), float(data['humidity']),
        float(data['ph']), float(data['rainfall'])
    ]])

    # Scale the input data using the scaler
    input_scaled = scaler.transform(input_data)
    
    # Get the prediction from the model
    prediction = model.predict(input_scaled)[0]
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
