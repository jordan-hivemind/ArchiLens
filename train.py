import os
import json
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

number_of_classes = 25

current_directory = os.path.dirname(os.path.realpath(__file__))
data_directory = os.path.join(current_directory, 'architectural-images-training-set/architectural-styles-dataset/')
print(data_directory)

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(data_directory, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Save the class labels to a JSON file
class_indices = train_generator.class_indices
class_labels = {v: k for k, v in class_indices.items()}

with open('class_labels.json', 'w') as f:
    json.dump(class_labels, f)

# Model Architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(number_of_classes, activation='softmax')  # Replace 'number_of_classes' with the actual number of classes
])

# Compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(train_generator, epochs=30)

model.save('archilens-model.h5')
