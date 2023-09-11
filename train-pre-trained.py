import os
import json
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

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

# Initialize the VGG16 model with its pre-trained weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freezing the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Customizing the model
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(number_of_classes, activation='softmax')
])

# Compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(train_generator, epochs=30)

model.save('archilens-model-finetuned.h5')
