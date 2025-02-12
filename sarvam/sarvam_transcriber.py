import requests

SARVAM_API_URL = "https://api.sarvam.ai/speech-to-text"
HEADERS = {"api-subscription-key": "YOUR_API_KEY"}  # Replace with your key

def transcribe_sarvam(audio_path, language_code="en", with_diarization=False, with_timestamps=False):
    """
    Transcribe an audio file using Sarvam AI's API.
    
    Parameters:
    - audio_path: str : Path to the audio file.
    - language_code: str : Language code for transcription.
    - with_diarization: bool : Enable speaker diarization.
    - with_timestamps: bool : Include timestamps in transcription.
    
    Returns:
    - str : Transcribed text or error message.
    """
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
