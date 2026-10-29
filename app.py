import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf
import keras

app = Flask(__name__, template_folder='.')
CORS(app)

# Load your trained Keras model
# Make sure 'road_damage_mobilenet.keras' is in the same directory
model = keras.models.load_model('road_damage_mobilenet.keras')

IMG_SIZE = 224

@app.route('/')
def home():
    return render_template('frontend.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    try:
        # Preprocess image exactly like your training pipeline
        img = Image.open(file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
        
        # Run model prediction
        prediction = float(model.predict(img_array)[0][0])
        
        # Interpret binary classification result
        is_broken = prediction > 0.5
        label = "Broken" if is_broken else "Not Broken"
        confidence = prediction if is_broken else (1 - prediction)
        
        return jsonify({
            'success': True,
            'label': label,
            'confidence': round(confidence * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)