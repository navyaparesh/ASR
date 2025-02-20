import gradio as gr
import os
import requests

SARVAM_API_URL = "https://api.sarvam.ai/speech-to-text"
HEADERS = { "api-subscription-key": "6f50ea02-8ad6-41fe-b265-58b506e07cd3" }

def transcribe_sarvam(audio_path, language_code="hi-IN", with_diarization=False, with_timestamps=False):
    try:
        with open(audio_path, 'rb') as audio_file:
            files = [('file', (audio_path, audio_file, 'audio/wav'))]
            payload = {
                'model': 'saarika:v2',
                'language_code': language_code,
                'with_diarization': str(with_diarization).lower(),
                'with_timesteps': str(with_timestamps).lower()
            }
            response = requests.post(SARVAM_API_URL, headers=HEADERS, data=payload, files=files)

            if response.status_code == 200:
                return response.json().get('transcript', 'No transcription available')
            else:
                return f"Error: {response.status_code}, {response.text}"
    
    except Exception as e:
        return f"Error in Sarvam AI transcription: {e}"

def process_audio(audio_path):
    if audio_path is None or not os.path.exists(audio_path):
        return "Please upload a valid audio file."

    transcript = transcribe_sarvam(audio_path)

    return transcript

interface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="Upload Audio"),
    outputs=gr.Textbox(label="Transcription"),
    title="🎙️ Sarvam AI Transcriber",
    description="Upload an audio file and transcribe it using Sarvam AI's ASR model.",
)

if __name__ == "__main__":
    interface.launch()
