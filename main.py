import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Constants
img_width, img_height = 128, 128
batch_size = 32
epochs = 10
dataset_path = 'dataset'

# Data preparation without validation split
data_gen = ImageDataGenerator(rescale=1./255)

train_gen = data_gen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

# Model building
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_gen.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model training without validation
model.fit(train_gen, epochs=epochs)

# Save the model
model.save('face_recognition_model.h5')

# Save class labels
class_labels = {v: k for k, v in train_gen.class_indices.items()}
np.save('class_labels.npy', class_labels)

print("Model and class labels saved.")
