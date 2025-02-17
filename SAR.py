import streamlit as st
import tempfile
import os
from preprocessing.audio_preprocessing import preprocess_audio
from sarvam.sarvam_transcriber import transcribe_sarvam
from audio_recorder_streamlit import audio_recorder

# Streamlit UI
st.title("🎙️ Sarvam AI ASR Pipeline")

# Select Data Source
data_source = st.radio("Select Data Source:", ["Local File", "Live Audio"])
uploaded_file = None

if data_source == "Local File":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])
elif data_source == "Live Audio":
    audio_bytes = audio_recorder()
    if audio_bytes:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio.write(audio_bytes)
        temp_audio.close()
        uploaded_file = temp_audio.name

# Enable Preprocessing
enable_preprocessing = st.checkbox("Enable Preprocessing")

# Process Button
if st.button("Process Audio"):
    if uploaded_file:
        # Handle File Uploader (Convert to Temporary File)
        if isinstance(uploaded_file, str):  # If live audio, it's already a file path
            file_path = uploaded_file
        else:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.write(uploaded_file.read())
            temp_file.close()
            file_path = temp_file.name

        # Apply Preprocessing
        if enable_preprocessing:
            processed_audio = preprocess_audio(file_path, "processed.wav")
        else:
            processed_audio = file_path

        # Transcription
        transcript = transcribe_sarvam(processed_audio)
        st.text(f"Transcription:\n{transcript}")

        # Cleanup Temporary Files
        os.remove(file_path)
    else:
        st.error("Please upload an audio file or record live audio!")
