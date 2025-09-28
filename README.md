# Audio Emotion Detection

This is a web application built with Streamlit that analyzes an audio file and predicts the emotion of the speaker. The app uses deep learning models (CNN and CNN-BiLSTM) to classify emotions into one of seven categories: angry, disgust, fear, happy, neutral, sad, or surprise.

## Features

* **Upload Audio:** Supports various audio formats (`.wav`, `.mp3`, `.flac`, `.m4a`), but for now just support `.wav` file.

* **Model Selection:** Allows users to choose between a `CNN` and a `CNN-BiLSTM` model for prediction.

* **Emotion Visualization:** Displays the prediction results with probabilities in an interactive bar chart.

* **Detailed Analysis:** Provides a table with the confidence scores for each emotion.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

* Python (version 3.8 or higher)

* `pip` (Python package installer)

## Setup and Installation

Follow these steps to set up the project locally.

1. **Create a Virtual Environment (Optional)**
It's recommended to use a virtual environment to keep project dependencies isolated. If you don't want to use virtual environtment just skip this step.

    * **On macOS and Linux:**

    ```
    python3 -m venv venv
    source venv/bin/activate
    
    ```

    * **On Windows:**

    ```
    python -m venv venv
    .\venv\Scripts\activate
    
    ```

2. **Install Required Libraries**
Install all the necessary Python packages using the `requirements.txt` file.
    ```
    pip install -r requirements.txt
    ```

## How to Run the Demo

1. **Verify Project Structure**
Ensure your project directory looks like this (assuming your script is named `app.py`):
    ```
    .
    ├── app.py
    ├── requirements.txt
    ├── README.md
    ├── emotion_model_CNN_BiLSTM.h5
    ├── emotion_model_CNN.h5
    ├── scaler.pkl
    └── encoder.pkl
    ```

2. **Start the Application**
Run the following command in your terminal from the project's root directory:
    ```
    streamlit run app.py
    ```

The application will automatically open in a new tab in your default web browser. You can now upload an audio file to begin detecting emotions.