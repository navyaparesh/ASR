import gradio as gr
import tempfile
import os
from audio_preprocessing import preprocess_audio
from sarvam_transcriber import transcribe_sarvam
from indic_conformer_transcriber import transcribe_indic

def process_audio(audio_file, asr_model, enable_preprocessing):
    """
    Processes the audio file and returns transcription.
    
    Parameters:
    - audio_file: file : Uploaded audio file.
    - asr_model: str : Selected ASR model.
    - enable_preprocessing: bool : Whether to preprocess the audio.

    Returns:
    - str : Transcribed text.
    """
    if audio_file is None:
        return "Please upload an audio file."

    # Save the uploaded file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.write(audio_file)
    temp_file.close()
    file_path = temp_file.name

    # Apply Preprocessing
    if enable_preprocessing:
        processed_audio = preprocess_audio(file_path, "processed.wav")
    else:
        processed_audio = file_path

    # Transcription
    if asr_model == "Sarvam AI":
        transcript = transcribe_sarvam(processed_audio)
    elif asr_model == "Indic Conformer":
        transcript = transcribe_indic(processed_audio)
    else:
        transcript = "Invalid model selected."

    # Cleanup Temporary File
    os.remove(file_path)

    return transcript

# Gradio Interface
interface = gr.Interface(
    fn=process_audio,
    inputs=[
        gr.Audio(source="upload", type="file", label="Upload Audio"),
        gr.Radio(["Sarvam AI", "Indic Conformer"], label="Select ASR Model"),
        gr.Checkbox(label="Enable Preprocessing"),
    ],
    outputs=gr.Textbox(label="Transcription"),
    title="🎙️ ASR Pipeline with Multiple Models",
    description="Upload an audio file and transcribe it using either Sarvam AI or Indic Conformer models.",
)

if __name__ == "__main__":
    interface.launch()
