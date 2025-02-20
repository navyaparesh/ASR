import gradio as gr
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff=1000, fs=44100, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def preprocess_audio(audio_path):
    try:
        if not audio_path or not os.path.exists(audio_path):
            return "Please upload a valid audio file."
        
        sr, y = wavfile.read(audio_path)
        y = y.astype(np.float32) / np.max(np.abs(y))  # Normalize
        
        # Noise reduction using low-pass filter
        reduced_noise = butter_lowpass_filter(y, cutoff=1000, fs=sr)
        
        # Trim silence (simple thresholding method)
        threshold = 0.02  # Adjust as needed
        non_silent_indices = np.where(np.abs(reduced_noise) > threshold)[0]
        if len(non_silent_indices) == 0:
            return "Error: Audio is silent."
        trimmed_audio = reduced_noise[non_silent_indices[0]:non_silent_indices[-1]]
        
        # Normalize volume
        max_val = np.max(np.abs(trimmed_audio))
        normalized_audio = trimmed_audio / max_val if max_val > 0 else trimmed_audio
        
        # Save final processed audio
        final_output = audio_path.replace(".wav", "_processed.wav")
        wavfile.write(final_output, sr, (normalized_audio * 32767).astype(np.int16))
        return final_output
    
    except Exception as e:
        return f"Error in processing audio: {e}"

def process_audio(audio_path):
    if not audio_path or not os.path.exists(audio_path):
        return "Please upload a valid audio file."
    
    return preprocess_audio(audio_path)

interface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath", label="Upload Audio"),
    outputs=gr.Audio(type="filepath", label="Processed Audio"),
    title="🎙️ Audio Cleaner",
    description="Upload an audio file to reduce noise, trim silence, and normalize volume.",
)

if __name__ == "__main__":
    interface.launch()
