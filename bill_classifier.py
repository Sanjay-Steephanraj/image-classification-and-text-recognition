from transformers import pipeline
import torch
import os

# Path to save the model
MODEL_PATH = "bill_classifier"

# Load or save model
if os.path.exists(MODEL_PATH):
    classifier = pipeline("zero-shot-classification", model=MODEL_PATH)
else:
    print("Model source is not found...")

# Bill categories
bill_categories = [
    "Food", "Travel", "Accomodation", "Others"
]

# Function to classify a bill
def classify_bill(text):
    result = classifier(text, bill_categories)
    return result["labels"][0]  # Return highest probability label