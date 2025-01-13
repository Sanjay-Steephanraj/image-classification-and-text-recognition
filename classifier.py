from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load pre-trained model
model = load_model("receipt_classifier_model.h5")

# Class labels
CATEGORIES = ["grocery", "restaurant", "electronics", "clothing", "other"]

def classify_receipt(image_path):
    # Load and preprocess image
    image = load_img(image_path, target_size=(128, 128))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict category
    predictions = model.predict(image)
    category = CATEGORIES[np.argmax(predictions)]
    return category
