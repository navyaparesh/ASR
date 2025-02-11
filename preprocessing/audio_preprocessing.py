import librosa
import soundfile as sf
import noisereduce as nr
from pydub import AudioSegment, silence

def preprocess_audio(audio_path, output_path):
    """
    Apply noise reduction, silence trimming, and volume normalization.
    
    Parameters:
    - audio_path: str : Path to the input audio file.
    - output_path: str : Path to save the processed audio file.
    
    Returns:
    - str : Path to the processed audio file.
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Noise reduction
        reduced_noise = nr.reduce_noise(y=y, sr=sr)
        
        # Save a temporary cleaned WAV file
        temp_wav = output_path.replace(".wav", "_temp.wav")
        sf.write(temp_wav, reduced_noise, sr)

        # Load with Pydub for silence trimming
        audio_segment = AudioSegment.from_wav(temp_wav)
        trimmed_audio = silence.split_on_silence(
            audio_segment, silence_thresh=-40, min_silence_len=800, keep_silence=300
        )

        # Combine trimmed segments
        processed_audio = sum(trimmed_audio) if trimmed_audio else audio_segment

        # Normalize volume
        processed_audio = processed_audio.apply_gain(-processed_audio.dBFS)

        # Save final processed audio
        processed_audio.export(output_path, format="wav")

        return output_path

    except Exception as e:
        return f"Error in preprocessing: {e}"
