from flask import Flask, request, jsonify
from ocr import perform_ocr
from receipt_parser import parse_receipt
from classifier import classify_receipt
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_receipt():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Perform OCR
    extracted_text = perform_ocr(file_path)

    # Parse receipt data
    parsed_data = parse_receipt(extracted_text)

    # Classify receipt
    category = classify_receipt(file_path)

    # Combine results
    result = {
        "text": extracted_text,
        "parsed_data": parsed_data,
        "category": category
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)