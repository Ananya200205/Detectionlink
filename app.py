import streamlit as st
import joblib
import numpy as np
from PIL import Image
import os

# Load the trained model and label encoder
@st.cache_resource
def load_model():
    model = joblib.load('random_forest_model.pkl')  # Load the trained model
    label_encoder = joblib.load('label_encoder.pkl')  # Load the saved label encoder
    return model, label_encoder

model, label_encoder = load_model()

# Preprocess image and make predictions
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to match the training size
    image_array = np.array(image).flatten()  # Convert to 1D array
    return image_array

# Streamlit app
st.title("Plant Disease Detection")

st.write("Upload a plant leaf image to predict the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        image_data = preprocess_image(image)

        # Make a prediction
        with st.spinner("Analyzing..."):
            prediction = model.predict([image_data])
            predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Display the result
        st.success(f"Predicted Disease: {predicted_label}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
