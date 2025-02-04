import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import tempfile
import requests
import os
from dotenv import load_dotenv  

# Load environment variables
load_dotenv()
API_KEY = os.getenv("TOGETHER_AI_API_KEY")
API_URL = "https://api.together.xyz/v1/chat/completions"

# Load Model Once and Cache
@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("Trained_Eye_Disease_Model.keras")

model = load_trained_model()

# Function to preprocess image and make prediction
def model_prediction(test_image_path):
    img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)[0]  # Get probabilities
    return np.argmax(predictions)  # Return top predicted class

# Function to get AI-generated recommendation using Together AI
def get_ai_recommendation(disease_name):
    if not API_KEY:
        return "Error: API key is missing. Please check your .env file."

    prompt = f"""
    You are an AI medical assistant. A user uploaded an OCT retinal scan, and the AI model predicted that the user has '{disease_name}'. 
    Provide a detailed but professional recommendation in points, including:
    - Treatment options
    - Lifestyle modifications
    - Next steps
    """

    data = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": "You are a medical assistant providing eye disease recommendations."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 700,
        "temperature": 0.7
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    response = requests.post(API_URL, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Error: " + response.json().get("error", {}).get("message", "Unknown error")

# Sidebar Navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Identification"])

# Home Page
if app_mode == "Home":
    st.markdown("""
    ## **OCT Retinal Analysis Platform**
    
    **Optical Coherence Tomography (OCT)** is a high-resolution imaging technique used for diagnosing retinal diseases like **CNV, DME, and Drusen**.
    
    #### **Key Features**
    - **AI-powered OCT scan classification** (Normal, CNV, DME, Drusen)
    - **AI-generated medical recommendations**
    - **Fast, accurate, and easy-to-use interface**
    
    **Get Started:** Upload an OCT scan to analyze retinal health.
    """)

# About Page
elif app_mode == "About":
    st.header("About This Project")
    st.markdown("""
    ### **Dataset Overview**
    - **84,495 OCT images** categorized into **Normal, CNV, DME, Drusen**
    - Data collected from leading medical centers worldwide
    - Labeled and validated by experienced ophthalmologists
    
    ### **How It Works**
    - The AI model analyzes uploaded OCT scans.
    - Classifies the image into one of four categories.
    - Provides relevant **medical recommendations** based on the result.
    """)

# Disease Identification Page
elif app_mode == "Disease Identification":
    st.header("Retinal OCT Analysis")

    # Upload Image
    test_image = st.file_uploader("Upload an OCT Image", type=["jpg", "png", "jpeg"])

    temp_file_path = None  # Initialize temporary file path

    # Process the uploaded image
    if test_image:
        # Save to a temporary file and get its path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(test_image.read())  # ✅ Read the image properly
            temp_file_path = tmp_file.name
        
        # Display Uploaded Image
        st.image(test_image, caption="Uploaded Image", use_container_width=True)  # ✅ Fix deprecated parameter

    # Predict Button
    if st.button("Predict") and temp_file_path:
        with st.spinner("Analyzing OCT scan... Please wait..."):
            result_index = model_prediction(temp_file_path)
            class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
            disease_name = class_names[result_index]

        # Display Prediction
        st.success(f"Prediction: **{disease_name}**")

        # AI-Generated Recommendation
        with st.expander("AI-Generated Medical Recommendation"):
            with st.spinner("Fetching recommendation..."):
                recommendation = get_ai_recommendation(disease_name)
            st.markdown(recommendation)
