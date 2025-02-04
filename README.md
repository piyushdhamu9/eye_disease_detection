# OCT Retinal Analysis Platform

## Overview
This project is an AI-powered platform for analyzing Optical Coherence Tomography (OCT) retinal scans. It classifies the scans into one of four categories: CNV, DME, DRUSEN, or NORMAL, and provides AI-generated medical recommendations based on the classification.

## Features
- **AI-powered OCT scan classification**: Classifies OCT scans into Normal, CNV, DME, or Drusen.
- **AI-generated medical recommendations**: Provides detailed recommendations based on the classification.
- **User-friendly interface**: Easy-to-use interface for uploading and analyzing OCT scans.

## Installation
1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up the environment variables:
    - Create a [.env](http://_vscodecontentref_/8) file in the root directory.
    - Add your Together AI API key to the [.env](http://_vscodecontentref_/9) file:
      ```
      TOGETHER_AI_API_KEY=your_api_key_here
      ```

## Usage
1. Run the Streamlit application:
    ```sh
    streamlit run main.py
    ```

2. Navigate to the Streamlit web interface in your browser.

3. Upload an OCT scan image to analyze retinal health.

## Project Structure

## Notebooks
- **Training_Model.ipynb**: Notebook for training the OCT classification model.
- **Model_Prediction.ipynb**: Notebook for making predictions using the trained model.

## Model
- The trained model is saved as [Trained_Eye_Disease_Model.keras](http://_vscodecontentref_/10).

## License
This project is licensed under the MIT License.

## Acknowledgements
- The dataset used for training the model was collected from leading medical centers worldwide and labeled by experienced ophthalmologists.
- The project uses TensorFlow and Keras for building and training the model.