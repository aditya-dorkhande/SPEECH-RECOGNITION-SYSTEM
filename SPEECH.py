# --- Step 1: Install required packages ---
!pip install torch torchvision torchaudio transformers librosa --quiet

# --- Step 2: Upload audio file in Google Colab ---
from google.colab import files
uploaded = files.upload()

# --- (Optional) clear output after upload ---
from IPython.display import clear_output
clear_output()

# --- Step 3: Import libraries ---
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# --- Step 4: Load a publicly available Hindi model ---
model_name = "theainerd/wav2vec2-large-xlsr-hindi"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# --- Step 5: Retrieve the uploaded filename ---
filename = list(uploaded.keys())[0]

# --- Step 6: Load audio at 16kHz ---
audio, sr = librosa.load(filename, sr=16000)

# --- Step 7: Convert audio to model input format ---
input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values

# --- Step 8: Perform inference to get logits ---
with torch.no_grad():
    logits = model(input_values).logits

# --- Step 9: Pick the most likely token IDs ---
predicted_ids = torch.argmax(logits, dim=-1)

# --- Step 10: Decode token IDs to text (Hindi) ---
transcription = processor.decode(predicted_ids[0])

print("Transcription (Hindi):")
print(transcription)
