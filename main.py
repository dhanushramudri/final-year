from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load and prepare data minimally
df = pd.read_csv('Crop_recommendation.csv')
X = df.drop('label', axis=1)
y = df['label']

# Train model
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True  # Corrected configuration

@app.route('/')
def index():
    return render_template('index.html')

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