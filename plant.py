from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("plant_disease_model.keras")

# IMPORTANT:
# These names must be in the SAME ORDER as the folders
# used during training.
class_names = [
    "Rice_Blast",
    "Rice_Healthy",
    "Wheat_Rust",
    "Wheat_Healthy",
    "Maize_Rust",
    "Maize_Healthy",
    "Cotton_Disease",
    "Cotton_Healthy",
    "Groundnut_Disease",
    "Groundnut_Healthy",
    "Chilli_Disease",
    "Chilli_Healthy",
    "Mango_Disease",
    "Mango_Healthy",
    "Banana_Disease",
    "Banana_Healthy",
    "Sugarcane_Disease",
    "Sugarcane_Healthy",
    "Pigeon_Pea_Disease",
    "Pigeon_Pea_Healthy",
    "Tomato_Disease",
    "Tomato_Healthy"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get uploaded image
    file = request.files["file"]

    # Open image
    image = Image.open(file).convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Convert image to numpy array
    image = np.array(image)

    # Normalize pixels
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Make prediction
    prediction = model.predict(image)

    # Get predicted class
    predicted_index = np.argmax(prediction[0])

    predicted_class = class_names[predicted_index]

    # Get confidence
    confidence = prediction[0][predicted_index] * 100

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=round(confidence, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)
