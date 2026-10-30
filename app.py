import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')  # quieter TensorFlow logs

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import keras

app = Flask(__name__)
CORS(app)

# Reject huge uploads early (phone photos can be big and use a lot of RAM)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

# Load your trained Keras model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'road_damage_mobilenet.keras')
IMG_SIZE = 224

model = None
if os.path.exists(MODEL_PATH):
    try:
        model = keras.models.load_model(MODEL_PATH)
        print("Successfully loaded Keras model from disk.")
        # Warm-up so the first real request is not slow
        model(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32), training=False)
        print("Model warm-up done.")
    except Exception as e:
        model = None
        print(f"Error: failed to load model: {e}")
else:
    print(f"Warning: Model file '{MODEL_PATH}' not found in directory.")


@app.route('/')
def home():
    return render_template('frontend.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Image is too large (max 10 MB).'}), 413


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Keras model not loaded on server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        img = Image.open(file.stream)
        img.draft('RGB', (IMG_SIZE * 2, IMG_SIZE * 2))  # decode big JPEGs at lower size (saves RAM)
        img = ImageOps.exif_transpose(img)              # fix phone-photo rotation
        img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Calling the model directly is lighter than model.predict() for a single image
        prediction = float(np.asarray(model(img_array, training=False))[0][0])

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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG') == '1')
