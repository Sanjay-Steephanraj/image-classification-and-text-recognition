from flask import Flask, request, jsonify
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = load_model("receipt_classifier_model.h5")

# Class labels
CATEGORIES = ["grocery", "restaurant", "electronics", "clothing", "other"]

# Configuration
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return "Receipt Processing API is Running!"

@app.route('/upload', methods=['POST'])
def upload_receipt():
    # Check if a file part is in the request
    if 'receipt' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['receipt']
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Validate filename
    if not file.filename or file.filename.strip() == '':
        return jsonify({"error": "File name is empty or invalid"}), 400

    # Save the file
    filename = os.path.basename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess the image for classification
    image = load_img(filepath, target_size=(128, 128))  # Resize to model input size
    image = img_to_array(image)  # Convert image to array
    image = image / 255.0  # Normalize image to [0, 1] range
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    try:
        predictions = model.predict(image)
        predicted_class = CATEGORIES[np.argmax(predictions)]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({"message": "File uploaded successfully", "predicted_class": predicted_class}), 200

if __name__ == '__main__':
    app.run(debug=True)   