import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf
import keras

app = Flask(__name__)
CORS(app)

# Load your trained Keras model safely
MODEL_PATH = 'road_damage_mobilenet.keras'
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH)
    print("Successfully loaded Keras model from disk.")
else:
    model = None
    print(f"Warning: Model file '{MODEL_PATH}' not found in directory.")

IMG_SIZE = 224

@app.route('/')
def home():
    return render_template('frontend.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Keras model not loaded on server.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    try:
        img = Image.open(file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = float(model.predict(img_array, verbose=0)[0][0])
        
        # Adjust threshold based on your model configuration (default > 0.5 = Broken)
        is_broken = prediction > 0.5
        label = "Broken" if is_broken else "Intact"
        confidence = round((prediction if is_broken else (1 - prediction)) * 100, 2)
        
        return jsonify({
            'success': True,
            'label': label,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)