# Alex-Ai
A.I. Agent Audio Processor built with user experience in mind - free to use non-API.
This will serve as the Audio Backbone of Somi A.I. Agent framework 

A Python-based voice assistant that uses wake word detection, real-time speech transcription, and text-to-speech synthesis. Powered by OpenAI Whisper, Coqui TTS, and Ollama, it listens for wake words (e.g., "hey assistant"), transcribes speech, processes queries via a local Ollama model, and responds with synthesized speech. Features include customizable chime sounds, audio file cleanup, and robust error handling.

Hardware requirements: Flexible and depends on the models used
Low-end Hardware - recommend switching model from gemma3:4b to gemma3:1b in settings 
High-end Hardware - recommend larger models for better processing
IF nvidia gpu please install cuda driver and cuda for python 
Features:
Wake Word Detection: Activates on phrases like "hey assistant" or "hi" with a pleasant chime sound.

Real-Time Transcription: Uses OpenAI Whisper to transcribe spoken input.

Natural Language Processing: Queries are processed by a local Ollama model (e.g., gemma3:4b).

Text-to-Speech: Generates responses using Coqui TTS with a natural-sounding voice.

Audio File Management: Saves output files in audio_outputs/ and cleans up files older than 1 hour every 5 minutes.

Configurable Settings: Customize wake words, chime sounds, and cleanup intervals in settings.py.

Debugging Support: Logs interactions and errors to speech_pipeline.log.

## Prerequisites
Please download and install the following

Python: Version 3.8–3.11 (3.10 recommended).

FFmpeg: Required for Whisper audio processing.

Ollama: Local server for NLP (model: gemma3:4b).

Microphone and Speakers: For audio input/output.

Optional: NVIDIA GPU for faster TTS with CUDA. Please install CUDA drivers from the Nvidia website for your gpu 

Installation
using command prompt
- git clone https://github.com/Somi-Project/Alex-Ai
- cd Alex-Ai
- pip install -r requirements.txt

In a separate command prompt
-ollama pull gemma3:4b
-ollama run gemma3:4b

Return to Alex-Ai root folder via Command Prompt
IF you have an Nvidia GPU we recommend the following: 
pip install torch==2.4.1+cu121 -f https://download.pytorch.org/whl/torch_stable.html

Return to Alex-Ai root folder 
- python speech.py

The assistant starts with a greeting: "Hi! My name is Somi and I'm here to help you - please say a trigger word to start chatting."

Say a wake word (e.g., "hey assistant") followed by a query.

Hear a "Ding...ding...ding!" chime when activated and a "Dong...dong" when the session ends (after 5 minutes of inactivity).


Troubleshooting
Microphone Errors:
Ensure a microphone is connected and set as the default input device.

Check sounddevice logs in speech_pipeline.log.

Ollama Connection Issues:
Verify Ollama is running (ollama serve) and the gemma3:4b model is installed.

Check OLLAMA_ENDPOINT in settings.py.

TTS Model Download:
On first run, Coqui TTS downloads tts_models/en/ljspeech/vits. Ensure internet access.

Low Audio Quality:
Adjust AUDIO_GAIN or SILENCE_THRESHOLD in settings.py.

File Clutter:
The cleanup thread deletes files in audio_outputs/ older than FILE_RETENTION_SECONDS every CLEANUP_INTERVAL seconds. Check logs for confirmation.

