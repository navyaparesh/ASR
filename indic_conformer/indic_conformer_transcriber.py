import torch
import nemo.collections.asr as nemo_asr
import subprocess
model_path = "hindi.nemo"
# Clone the NeMo repository
subprocess.run(["git", "clone", "https://github.com/AI4Bharat/NeMo.git", "-b", "nemo-v2"], check=True)

# Load Indic Conformer model
# Download model file using wget
model_url = "https://objectstore.e2enetworks.net/indic-asr-public/indicConformer/ai4b_indicConformer_hi.nemo"


subprocess.run(["wget", model_url, "-O", model_path], check=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ASR model
if os.path.exists(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=model_path)
    model.eval().to(device)
else:
    raise FileNotFoundError(f"ASR Model file {model_path} not found!")

def transcribe_indic(audio_path):
    """
    Transcribe an audio file using Indic Conformer ASR.
    
    Parameters:
    - audio_path: str : Path to the input audio file.
    
    Returns:
    - str : Transcribed text or error message.
    """
    try:
        transcription = model.transcribe([audio_path], batch_size=1, logprobs=False, language_id='hi')[0].strip()
        return transcription
    except Exception as e:
        return f"Error in Indic Conformer transcription: {e}"
