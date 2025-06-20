import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

image_x, image_y = 50, 50

# Load Dataset
def load_data():
    images, labels = [], []
    for gesture_id in os.listdir("gestures"):
        folder = f"gestures/{gesture_id}"
        if os.path.isdir(folder):
            for img_file in os.listdir(folder):
                img = cv2.imread(f"{folder}/{img_file}", 0)
                img = img.astype(np.float32) / 255.0  # normalize
                images.append(img)
                labels.append(int(gesture_id))
    images = np.array(images).reshape(-1, image_x, image_y, 1)
    labels = to_categorical(labels)
    return images, labels

images, labels = load_data()

# Split train/validation
from sklearn.model_selection import train_test_split
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.1, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True
)
datagen.fit(train_images)

# Build stronger model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(image_x, image_y, 1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(labels.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(datagen.flow(train_images, train_labels, batch_size=32),
        epochs=30,
        validation_data=(val_images, val_labels))

model.save("cnn_model_keras2.h5")
