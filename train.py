import tensorflow as tf
from tensorflow.keras import layers, models
import os

# Dataset path
dataset_path = "dataset"

# Image parameters
img_height = 224
img_width = 224
batch_size = 32

# Load training dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Load validation dataset
val_data = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Get class names
class_names = train_data.class_names

print("Classes:")
print(class_names)

# Number of classes
num_classes = len(class_names)

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.cache().shuffle(1000).prefetch(
    buffer_size=AUTOTUNE
)

val_data = val_data.cache().prefetch(
    buffer_size=AUTOTUNE
)

# CNN Model
model = models.Sequential([

    # Normalize pixel values
    layers.Rescaling(1.0 / 255),

    # Convolution Layer 1
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    # Convolution Layer 2
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    # Convolution Layer 3
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    # Flatten
    layers.Flatten(),

    # Fully Connected Layer
    layers.Dense(128, activation="relu"),

    # Output Layer
    layers.Dense(num_classes, activation="softmax")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Display model
model.summary()

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

# Save model
model.save("plant_disease_model.keras")

print("Model training completed!")
print("Model saved as plant_disease_model.keras")
