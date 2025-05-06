import streamlit as st
from PIL import Image
import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np

model = tf.keras.models.load_model("my_model.h5")


def preprocess_image(image):
    # Convert to grayscale if required by your model
    image = image.convert('RGB')
    image = np.array(image)
    image = cv2.resize(image, (256, 256))  # Resize to the required size (256x256)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


st.title("Image Classification with TensorFlow")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(preprocessed_image)
    # predicted_class = np.argmax(prediction, axis=1)[0]

    if prediction > 0.5:
        prediction = 'Sad'
    else:
        prediction='Happy'
    # Display the prediction
    st.header(f"Predicted Expression: {prediction}")
