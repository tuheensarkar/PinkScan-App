from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)  # Fixed _name to _name_
CORS(app)

# Load the trained model
model = joblib.load('breast_cancer_model.pkl')

@app.route('/predict', methods=['POST'])
def home():
    return render_template('index.html')
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    result = 'Malignant' if prediction[0] == 0 else 'Benign'
    return jsonify({'prediction': result})

# Fixed name to _name_
if _name_ == '_main_':
    app.run(debug=True, port=5001)