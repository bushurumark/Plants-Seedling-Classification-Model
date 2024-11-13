#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# Direct download link of the model file from Google Drive
url = 'https://drive.google.com/uc?id=1olb0yiQB1n-QH2QCdPLbkn-JJV_bjxB7'

# Path to save the downloaded model file
model_path = 'model.h5'

# Download the model if it does not exist
try:
    if not os.path.exists(model_path):
        with st.spinner('Downloading model...'):
            gdown.download(url, model_path, quiet=False)
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model(model_path)
    model = load_model()
except Exception as e:
    st.error("Error loading the model. Please try again later.")
    st.stop()

# Define the seedling names
Seedling_Names = [
    "Black-grass", "Charlock", "Cleavers", "Common Chickweed", "Common Wheat", 
    "Fat Hen", "Loose Silky-bent", "Maize", "Scentless Mayweed", 
    "Shepherds Purse", "Small-flowered Cranesbill", "Sugar beet"
]

# Define a function to preprocess the image for the model
def preprocess_image(image):
    image = image.resize((64, 64))  # Adjust size to match model input
    image = image.convert("RGB")  # Ensure image is in RGB mode
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize
    return image

# Define a function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction

# Custom CSS for green background and other elements
st.markdown("""
    <style>
    .main {
        background-color: #a8d5ba;  /* Light green background */
    }
    .title {
        color: #333;
        font-family: 'Arial';
        text-align: center;
    }
    .prediction {
        font-size: 20px;
        color: indigo;
        font-weight: bold;
        text-align: center;
    }
    .uploaded-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        border: 5px solid #ccc;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit interface with custom styles
st.markdown('<h1 class="title">Plants Seedlings Classification Model</h1>', unsafe_allow_html=True)

# Option to choose between uploading an image or taking a photo
option = st.radio("Select image input method:", ("Upload an Image", "Take a Photo"))

# Initialize the image variable
image = None

# Handle image input based on the user's choice
if option == "Upload an Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

elif option == "Take a Photo":
    camera_photo = st.camera_input("Take a photo...")
    if camera_photo is not None:
        image = Image.open(camera_photo)

# If an image is provided, display it and make a prediction
if image is not None:
    st.image(image, caption='Selected Image.', use_column_width=True)
    st.write("")
    st.markdown('<div class="prediction">Classifying...</div>', unsafe_allow_html=True)
    
    # Make prediction
    prediction = predict(image)
    
    # Get the index of the highest probability
    predicted_index = np.argmax(prediction, axis=1)[0]
    confidence_score = np.max(prediction, axis=1)[0]
    
    # Get the corresponding species name
    predicted_seedling = Seedling_Names[predicted_index]
    
    # Display the prediction with confidence score
    confidence_threshold = 0.7  # Threshold to control prediction confidence
    if confidence_score > confidence_threshold:
        st.markdown(f'<div class="prediction">Prediction: {predicted_seedling} (Confidence: {confidence_score:.2f})</div>', unsafe_allow_html=True)
        st.progress(int(confidence_score * 100))
    else:
        st.markdown('<div class="prediction">The uploaded image is not confidently recognized as a seedling species.</div>', unsafe_allow_html=True)

