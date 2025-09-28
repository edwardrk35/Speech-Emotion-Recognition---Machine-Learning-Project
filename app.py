import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import pickle
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import tempfile
import os
from tensorflow.keras.models import load_model
import joblib

st.set_page_config(
    page_title="Audio Emotion Detection",
    page_icon="🎵",
    layout="wide"
)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

MODEL_CONFIGS = {
    'CNN-BiLSTM': {
        'model_path': 'emotion_model_CNN_BiLSTM.h5',
        'scaler_path': 'scaler.pkl',
        'encoder_path': 'encoder.pkl',
        'description': 'CNN-BiLSTM model combines Convolutional Neural Networks with Bidirectional LSTM for better temporal feature learning'
    },
    'CNN': {
        'model_path': 'emotion_model_CNN.h5',
        'scaler_path': 'scaler.pkl',
        'encoder_path': 'encoder.pkl',
        'description': 'CNN model uses Convolutional Neural Networks for spatial feature extraction'
    }
}

@st.cache_resource
def load_models(model_type):
    """Load the pre-trained model, scaler, and encoder based on model type"""
    try:
        config = MODEL_CONFIGS[model_type]
        model = load_model(config['model_path'])
    
        scaler = joblib.load(config['scaler_path'])
        encoder = joblib.load(config['encoder_path'])
        
        return model, scaler, encoder
    except Exception as e:
        st.error(f"Error loading {model_type} models: {str(e)}")
        return None, None, None

def extract_features(data, sample_rate=22050):
    """Extract audio features from the input data"""
    res = np.array([])
    
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    res = np.hstack((res, mfcc))
    
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    res = np.hstack((res, zcr))
    
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    res = np.hstack((res, mel))
    
    return res

def predict_emotion(audio_data, sample_rate, model, scaler, encoder, model_type):
    """Predict emotion from audio data"""
    try:
    
        features = extract_features(audio_data, sample_rate)
        features = features.reshape(1, -1)
        print(f"After extracting features : {features.shape}")
        
        features = scaler.transform(features)
        print(f"After scaling : {features.shape}")

        if model_type == 'CNN-BiLSTM':
            features = np.expand_dims(features, axis=2)
            print(f"After expand dimension for CNN-BiLSTM : {features.shape}")
        elif model_type == 'CNN':
            features = features.reshape(features.shape[0], features.shape[1], 1)
            print(f"After reshape for CNN : {features.shape}")
        
    
        prediction = model.predict(features)
        print(f"Prediction: {prediction}")
        print(f"Prediction shape : {prediction.shape}")
        
        emotion_probs = prediction[0]
    
        tmp = encoder.inverse_transform(prediction)
        print(f"Predicted emotion : {tmp}")

        return tmp[0][0], emotion_probs
    
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None

def create_emotion_chart(emotion_probs, model_type):
    """Create a bar chart showing emotion probabilities"""
    df = pd.DataFrame({
        'Emotions': EMOTIONS,
        'Probability': emotion_probs
    })
    
    df = df.sort_values('Probability', ascending=True)
    
    fig = px.bar(df, 
                 x='Probability', 
                 y='Emotions',
                 orientation='h',
                 title=f'Emotion Detection Probabilities ({model_type})',
                 color='Probability',
                 color_continuous_scale='viridis')
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_font_size=16
    )
    
    return fig

def display_model_info(model_type):
    """Display information about the selected model"""
    config = MODEL_CONFIGS[model_type]
    st.info(f"**{model_type} Model:** {config['description']}")

def main():
    st.title("🎵 Audio Emotion Detection")
    st.markdown("Upload an audio file to detect the emotion expressed in the speech using different deep learning models.")
    
    st.sidebar.header("🤖 Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose a model:",
        options=list(MODEL_CONFIGS.keys()),
        help="Select which deep learning model to use for emotion detection"
    )
    
    display_model_info(selected_model)

    with st.spinner(f"Loading {selected_model} model..."):
        model, scaler, encoder = load_models(selected_model)
    
    if model is None or scaler is None or encoder is None:
        st.error(f"Failed to load required {selected_model} models. Please ensure the following files are in the same directory:")
        config = MODEL_CONFIGS[selected_model]
        st.write(f"- {config['model_path']}")
        st.write(f"- {config['scaler_path']}") 
        st.write(f"- {config['encoder_path']}")
        return
    
    st.success(f"✅ {selected_model} model loaded successfully!")

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Audio File")
    
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'flac', 'm4a'],
            help="Supported formats: WAV, MP3, FLAC, M4A"
        )
        
        if uploaded_file is not None:
        
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size / 1024:.2f} KB")
            st.write(f"**Selected Model:** {selected_model}")
            
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("🔍 Analyze Emotion", type="primary"):
                with st.spinner(f"Processing audio with {selected_model}..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_file_path = tmp_file.name

                        audio_data, sample_rate = librosa.load(tmp_file_path)
                    
                        os.unlink(tmp_file_path)
                        
                        predicted_emotion, emotion_probs = predict_emotion(
                            audio_data, sample_rate, model, scaler, encoder, selected_model
                        )
                        
                        if predicted_emotion is not None:
                        
                            st.session_state.predicted_emotion = predicted_emotion
                            st.session_state.emotion_probs = emotion_probs
                            st.session_state.selected_model = selected_model
                            st.session_state.analysis_complete = True
                            
                    except Exception as e:
                        st.error(f"Error processing audio: {str(e)}")
    
    with col2:
        st.subheader("Emotion Analysis Results")
        
        if hasattr(st.session_state, 'analysis_complete') and st.session_state.analysis_complete:
            predicted_emotion = st.session_state.predicted_emotion
            emotion_probs = st.session_state.emotion_probs
            used_model = st.session_state.selected_model
            
            emotion_emojis = {
                'angry': '😠',
                'disgust': '🤢', 
                'fear': '😨',
                'happy': '😊',
                'neutral': '😐',
                'sad': '😢',
                'surprise': '😲'
            }
            
            emoji = emotion_emojis.get(predicted_emotion, '🎭')
            st.success(f"**Detected Emotion:** {emoji} {predicted_emotion.title()}")
            st.info(f"**Model Used:** {used_model}")
            
            max_prob = max(emotion_probs)
            st.info(f"**Confidence:** {max_prob:.2%}")
            
            fig = create_emotion_chart(emotion_probs, used_model)
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Detailed Probabilities")
            prob_df = pd.DataFrame({
                'Emotion': EMOTIONS,
                'Probability': [f"{prob:.4f}" for prob in emotion_probs],
                'Percentage': [f"{prob:.2%}" for prob in emotion_probs]
            })
            
            prob_df['Prob_Val'] = emotion_probs
            prob_df = prob_df.sort_values('Prob_Val', ascending=False).drop('Prob_Val', axis=1)
            
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
        else:
            st.info("👆 Upload an audio file and click 'Analyze Emotion' to see results here.")
    

    # if hasattr(st.session_state, 'analysis_complete') and st.session_state.analysis_complete:
    #     st.markdown("---")
    #     st.subheader("🔄 Compare with Other Model")
        
    #     other_models = [m for m in MODEL_CONFIGS.keys() if m != st.session_state.selected_model]
        
    #     if other_models:
    #         compare_model = st.selectbox(
    #             "Compare with:",
    #             options=other_models,
    #             help="Select another model to compare results"
    #         )
            
    #         if st.button(f"🔍 Analyze with {compare_model}", type="secondary"):
    #             with st.spinner(f"Processing with {compare_model}..."):
    #                 try:
    #                     comp_model, comp_scaler, comp_encoder = load_models(compare_model)
    #                     if comp_model is not None:
    #                         st.info("Please re-upload the audio file to compare with different models.")
                            
    #                 except Exception as e:
    #                     st.error(f"Error loading {compare_model}: {str(e)}")
    

    st.markdown("---")
    st.subheader("ℹ️ About This App")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Supported Emotions:**
        - 😠 Angry
        - 🤢 Disgust
        - 😨 Fear
        """)
    
    with col2:
        st.markdown("""
        **More Emotions:**
        - 😊 Happy
        - 😐 Neutral
        - 😢 Sad
        """)
    
    with col3:
        st.markdown("""
        **Additional:**
        - 😲 Surprise
        
        **Features Used:**
        - MFCC coefficients
        - Zero crossing rate
        - Mel spectrogram
        """)
    

    st.markdown("---")
    st.subheader("🤖 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **CNN-BiLSTM Model:**
        - Combines Convolutional layers for feature extraction
        - Uses Bidirectional LSTM for temporal sequence modeling
        - Better for capturing long-term dependencies in audio
        - Generally higher accuracy but slower processing
        """)
    
    with col2:
        st.markdown("""
        **CNN Model:**
        - Uses only Convolutional Neural Networks
        - Focuses on spatial feature patterns
        - Faster processing and inference
        - Simpler architecture, good baseline performance
        """)

if __name__ == "__main__":
    main()