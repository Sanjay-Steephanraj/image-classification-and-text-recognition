import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define your model
def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Set up the image data generator
train_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

# Set up directories
train_dir = './path_to_your_dataset/train'
validation_dir = './path_to_your_dataset/validation'

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # Resize images to match the model input shape
    batch_size=32,
    class_mode='categorical',  # Multi-class classification
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',  # Multi-class classification
    subset='validation'
)

# Build and train the model
input_shape = (128, 128, 3)
num_classes = len(train_generator.class_indices)  # The number of categories
model = build_model(input_shape, num_classes)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32,
    epochs=10
)

# Save the trained model
model.save("receipt_classifier_model.h5")