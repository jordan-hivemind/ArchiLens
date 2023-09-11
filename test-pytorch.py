import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Custom architecture to match the original TensorFlow model
class CustomModel(nn.Module):
    def __init__(self, num_classes=25):
        super(CustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 54 * 54, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load class labels
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Initialize the model
model = CustomModel()

# Load the saved model
model.load_state_dict(torch.load('archilens-model.pth'))
model.eval()

# Prepare data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Folder containing all test images
test_folder = "test-images"

# Loop through all files in the folder
for filename in os.listdir(test_folder):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(test_folder, filename)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add a batch dimension

        # Make the prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()
            confidence_score = nn.functional.softmax(outputs, dim=1)[0][predicted_class]

        print(f'File: {filename}, Predicted class: {class_labels[str(predicted_class)]}, Confidence: {confidence_score:.2f}')
