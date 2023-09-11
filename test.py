import json
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from keras.optimizers import Adam
import numpy as np

# Load the class labels
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

# model = load_model('archilens-model.h5') # Model built from scratch
model = load_model('archilens-model-finetuned.h5') # Pre-trained
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Folder containing all test images
test_folder = "test-images"

# Loop through all files in the folder
for filename in os.listdir(test_folder):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):  # Add more conditions if you have different file types
        img_path = os.path.join(test_folder, filename)
        
        # Load and preprocess the image; note the change in target_size
        img = image.load_img(img_path, target_size=(224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        
        # Make the prediction
        prediction = model.predict(img_tensor)
        
        # Interpret the prediction
        predicted_class = np.argmax(prediction[0])
        predicted_label = class_labels[str(predicted_class)]  # JSON keys are always strings
        confidence_score = prediction[0][predicted_class]        
        
        print(f'File: {filename}, Predicted class: {predicted_label}, Confidence: {confidence_score:.2f}')
