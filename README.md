# SPEECH-RECOGNITION-SYSTEM

*COMPANY*: CODTECH IT SOLUTION

*NAME*: ADITYA DORKHANDE

*INTERN ID*: CT08TLF

*DOMAIN*: ARRIFICIAL INTELLIGENCE

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH


*DESCRIPTION*
Purpose
The script transcribes Hindi audio files (WAV, MP3, etc.) into text using a pretrained model from Hugging Face.
By default, it uses theainerd/wav2vec2-large-xlsr-hindi, a Wav2Vec2 model fine-tuned for Hindi speech.
How It Works
Dependency Imports

PyTorch (torch, torchaudio) for loading and running the model.
Transformers to load the Wav2Vec2 model/processor from Hugging Face.
librosa to load audio at the correct sample rate (16 kHz).
Audio Loading

The script calls librosa.load(audio_path, sr=16000) to load your audio file and resample it to 16 kHz, which is the typical sampling rate for Wav2Vec2 models.
Model Processing

The script initializes a Wav2Vec2Processor (which handles tokenization) and a Wav2Vec2ForCTC model (which generates the speech logits).
It feeds the audio data (converted into tensors) to the model.
The model produces logits, which are basically probability distributions over the possible tokens.

#output
![Image](https://github.com/user-attachments/assets/9b4df624-0473-4dae-a1bd-71f31bfc5e85)
