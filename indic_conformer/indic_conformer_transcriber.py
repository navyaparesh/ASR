import torch
import nemo.collections.asr as nemo_asr
!git clone https://github.com/AI4Bharat/NeMo.git -b nemo-v2
# Load Indic Conformer model
!wget https://objectstore.e2enetworks.net/indic-asr-public/indicConformer/ai4b_indicConformer_hi.nemo -O hindi.nemo

model_path = "hindi.nemo"  # Ensure this file is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ASR model
model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=model_path)
model.eval().to(device)

def transcribe_indic_conformer(audio_path):
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
