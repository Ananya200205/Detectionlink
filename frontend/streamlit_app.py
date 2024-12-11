import streamlit as st
import requests

st.title("Plant Disease Detection")

# File uploader
uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Send the file to Flask API
    with st.spinner("Analyzing..."):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://127.0.0.1:5000/predict", files={"file": uploaded_file})
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Disease: {result['predicted_disease']}")
        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")
