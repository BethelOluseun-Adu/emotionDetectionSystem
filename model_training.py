
# OPTIONAL: model_training.py
# This script shows how you'd train a CNN using TensorFlow on a facial expression dataset.
# It's commented out so this file is safe to run even without TensorFlow installed.
#
# To use it:
# 1) Install TensorFlow: pip install tensorflow
# 2) Prepare training data in data/train and data/validation with subfolders per emotion
# 3) Uncomment code below and run: python model_training.py

"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = 'data/train'
val_dir = 'data/validation'

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(48,48), batch_size=32, color_mode='grayscale', class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(48,48), batch_size=32, color_mode='grayscale', class_mode='categorical')

model = Sequential([
    Conv2D(32,(3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, validation_data=val_generator, epochs=15)
model.save('face_emotionModel.h5')
"""
