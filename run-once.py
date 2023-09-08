import os
import json

# Assuming your training images are in a folder named 'train'
train_folder = '/Users/jordanschwartz/Library/CloudStorage/Dropbox/z - Miscellaneous/python/ArchiLens/architectural-images-training-set/architectural-styles-dataset'

class_labels = {}
for index, class_name in enumerate(os.listdir(train_folder)):
    class_labels[str(index)] = class_name

with open('class_labels.json', 'w') as f:
    json.dump(class_labels, f)
