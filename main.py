import streamlit as st
import tempfile
import os
from preprocessing.audio_preprocessing import preprocess_audio
from sarvam.sarvam_transcriber import transcribe_sarvam
from indic_conformer.indic_conformer_transcriber import transcribe_indic_conformer
from audio_recorder_streamlit import audio_recorder

# Streamlit UI
st.title("🎙️ ASR Pipeline with Multiple Models")

# Select Data Source
data_source = st.radio("Select Data Source:", ["Local File", "Live Audio"])

if data_source == "Local File":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])
elif data_source == "Live Audio":
    audio_bytes = audio_recorder()
    if audio_bytes:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio.write(audio_bytes)
        temp_audio.close()
        uploaded_file = temp_audio.name

# Select ASR Model
asr_model = st.selectbox("Select ASR Model:", ["Sarvam AI", "Indic Conformer"])

# Enable Preprocessing
enable_preprocessing = st.checkbox("Enable Preprocessing")

# Process Button
if st.button("Process Audio"):
    if uploaded_file:
        # Apply Preprocessing
        if enable_preprocessing:
            preprocessed_audio = preprocess_audio(uploaded_file, "processed.wav")
        else:
            preprocessed_audio = uploaded_file

        # Transcription
        if asr_model == "Sarvam AI":
            transcript = transcribe_sarvam(preprocessed_audio)
        elif asr_model == "Indic Conformer":
            transcript = transcribe_indic_conformer(preprocessed_audio)
        else:
            transcript = "Invalid model selected."

        st.text(f"Transcription: {transcript}")
    else:
        st.error("Please upload an audio file!")
