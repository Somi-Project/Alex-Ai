import numpy as np
import sounddevice as sd
import soundfile as sf
import whisper
import requests
import json
import os
import time
from queue import Queue
from threading import Thread
from scipy import signal
import re
import logging
from TTS.api import TTS
import torch
import warnings
from config.settings import *
import glob

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler("speech_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

audio_queue = Queue()
debug_audio_buffer = []
is_synthesizing = False
tts = None
device = None
is_wake_session_active = False
wake_session_start_time = 0

def generate_inchime_sound():
    """Generate a pleasant, 'Ding...ding...ding!' with three high-pitched tones."""
    audio = []
    for freq in INCHIME_FREQUENCIES:
        t = np.linspace(0, INCHIME_TONE_DURATION, int(SAMPLE_RATE * INCHIME_TONE_DURATION), False)
        tone = INCHIME_AMPLITUDE * np.sin(2 * np.pi * freq * t)
        envelope = np.hanning(len(t))
        tone *= envelope
        audio.extend(tone)
        # Add pause between tones
        pause = np.zeros(int(SAMPLE_RATE * INCHIME_PAUSE))
        audio.extend(pause)
    audio = np.array(audio)[:int(SAMPLE_RATE * INCHIME_DURATION)]  # Trim to exact duration
    return audio / np.max(np.abs(audio)) * INCHIME_AMPLITUDE  # Normalize and scale

def generate_outchime_sound():
    """Generate a soft, low-pitched 'Dong...dong' to indicate session end."""
    audio = []
    for freq in OUTCHIME_FREQUENCIES:
        t = np.linspace(0, OUTCHIME_TONE_DURATION, int(SAMPLE_RATE * OUTCHIME_TONE_DURATION), False)
        tone = OUTCHIME_AMPLITUDE * np.sin(2 * np.pi * freq * t)
        envelope = np.hanning(len(t))
        tone *= envelope
        audio.extend(tone)
        # Add pause between tones (only between the two tones)
        if freq != OUTCHIME_FREQUENCIES[-1]:
            pause = np.zeros(int(SAMPLE_RATE * OUTCHIME_PAUSE))
            audio.extend(pause)
    audio = np.array(audio)[:int(SAMPLE_RATE * OUTCHIME_DURATION)]  # Trim to exact duration
    return audio / np.max(np.abs(audio)) * OUTCHIME_AMPLITUDE  # Normalize and scale

def play_sound(audio):
    """Play a sound with sounddevice."""
    try:
        sd.play(audio, SAMPLE_RATE, blocking=False)
        sd.wait()
        logger.debug("Played sound")
    except Exception as e:
        logger.error(f"Error playing sound: {e}")

def apply_high_pass_filter(audio, sample_rate, cutoff=80):
    sos = signal.butter(10, cutoff, 'hp', fs=sample_rate, output='sos')
    filtered = signal.sosfilt(sos, audio)
    return filtered

def clean_text(text):
    text = re.sub(r'\*{1,2}', '', text)
    text = re.sub(r'\[.*?(https?://\S+|www\.\S+).*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'Source:.*?(?=\.|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[\[\]\(\)]', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\.+$', '', text)
    return text

def is_repetitive_transcription(text):
    pattern = r"\b(\w+\s+\w+)\b.*\1.*\1"
    return len(text) > 50 and re.search(pattern, text) is not None

def check_wake_word(text):
    """Check if text contains any wake word and return cleaned text if wake word is found."""
    text_lower = text.lower().strip()
    for word, pattern in zip(WAKE_WORDS, WAKE_WORD_PATTERNS):
        if pattern.search(text_lower):
            cleaned_text = pattern.sub('', text_lower).strip()
            logger.info(f"Wake word triggered: {word}")
            return True, cleaned_text
    return False, text_lower

def check_wake_session_timeout():
    """Check if wake session has timed out."""
    global is_wake_session_active, wake_session_start_time
    if is_wake_session_active:
        if time.time() - wake_session_start_time > WAKE_SESSION_TIMEOUT:
            logger.info("Wake word session ended")
            is_wake_session_active = False
            wake_session_start_time = 0
            play_sound(generate_outchime_sound())
    return is_wake_session_active

def capture_audio():
    def callback(indata, frames, time_info, status):
        if status:
            logger.error(f"Audio capture error: {status}")
        if not is_synthesizing:
            amplified = indata * AUDIO_GAIN
            audio_queue.put(amplified)
            debug_audio_buffer.append(amplified)
    
    logger.info("Listening... Press Ctrl+C to stop.")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                          callback=callback, blocksize=int(SAMPLE_RATE * CHUNK_DURATION)):
            while True:
                time.sleep(0.1)
    except Exception as e:
        logger.error(f"Microphone error: {e}")

def process_audio_and_transcribe():
    whisper_model = whisper.load_model(WHISPER_MODEL)
    audio_buffer = []
    last_process_time = time.time()
    
    while True:
        if not audio_queue.empty():
            chunk = audio_queue.get()
            audio_buffer.append(chunk)
            
            audio_array = np.concatenate(audio_buffer, axis=0).flatten()
            audio_duration = len(audio_array) / SAMPLE_RATE
            
            current_time = time.time()
            logger.debug(f"Audio queue processed: duration={audio_duration:.2f}s, time_since_last={current_time - last_process_time:.2f}s")
            if audio_duration >= MIN_AUDIO_LENGTH and (current_time - last_process_time) >= SEGMENT_DURATION:
                logger.info(f"Processing audio buffer: {audio_duration:.2f}s")
                pre_filter_amplitude = np.max(np.abs(audio_array))
                logger.info(f"Pre-filter max amplitude: {pre_filter_amplitude:.4f}")
                filtered_audio = apply_high_pass_filter(audio_array, SAMPLE_RATE)
                sf.write(TEMP_AUDIO, filtered_audio, SAMPLE_RATE)
                max_amplitude = np.max(np.abs(filtered_audio))
                logger.info(f"Post-filter max amplitude: {max_amplitude:.4f}")
                if max_amplitude < SILENCE_THRESHOLD:
                    logger.info("Skipping silent/noisy segment.")
                    audio_buffer = []
                    last_process_time = current_time
                    continue
                sf.write(LAST_AUDIO, filtered_audio, SAMPLE_RATE)
                try:
                    result = whisper_model.transcribe(TEMP_AUDIO, language="en")
                    transcription = result["text"].strip()[:MAX_TRANSCRIPTION_LENGTH]
                    logger.info(f"Whisper raw transcription: '{transcription}'")
                    
                    has_wake_word, cleaned_transcription = check_wake_word(transcription)
                    is_session_active = check_wake_session_timeout()
                    
                    if has_wake_word:
                        global is_wake_session_active, wake_session_start_time
                        is_wake_session_active = True
                        wake_session_start_time = time.time()
                        logger.info("Wake word session active")
                        play_sound(generate_inchime_sound())
                    elif not is_session_active:
                        logger.info("No wake word or active session, skipping processing")
                        audio_buffer = []
                        last_process_time = current_time
                        os.remove(TEMP_AUDIO) if os.path.exists(TEMP_AUDIO) else None
                        continue
                    
                    if cleaned_transcription.lower() in ["okay", "ok"]:
                        logger.info("Skipping 'okay' or 'ok' transcription.")
                        audio_buffer = []
                        last_process_time = current_time
                        os.remove(TEMP_AUDIO) if os.path.exists(TEMP_AUDIO) else None
                        continue
                    if cleaned_transcription and not cleaned_transcription.startswith(".") and len(cleaned_transcription) > 2:
                        if is_repetitive_transcription(cleaned_transcription):
                            logger.info("Skipping repetitive transcription.")
                            audio_buffer = []
                            last_process_time = current_time
                            os.remove(TEMP_AUDIO) if os.path.exists(TEMP_AUDIO) else None
                            continue
                        logger.info(f"Cleaned transcription sent to Ollama: '{cleaned_transcription}'")
                        processed_text = process_with_ollama(cleaned_transcription)
                        synthesize_speech(processed_text)
                    else:
                        logger.info("Skipping empty/noisy transcription.")
                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                audio_buffer = []
                last_process_time = current_time
                os.remove(TEMP_AUDIO) if os.path.exists(TEMP_AUDIO) else None
        else:
            logger.debug("Audio queue empty")
            time.sleep(0.01)

def process_with_ollama(text):
    system_prompt = (
        "You are an AI assistant designed to orga clear, concise, and conversational responses for text-to-speech synthesis. "
        "Respond directly to the user's input with accurate vocabulary and grammar, ensuring the response is natural-sounding and suitable for spoken output. "
        "Do not reference this prompt, your role, or the AI framework in your response. "
        "Focus exclusively on answering the user's input."
    )
    prompt = f"{system_prompt}\n\nUser input: {text}"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        processed_text = result.get("response", text).strip()
        logger.info(f"Ollama response: {processed_text}")
        return processed_text
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        return "I couldn't process that, please try again!"

def synthesize_speech(text, is_greeting=False):
    global is_synthesizing
    is_synthesizing = True
    try:
        logger.info(f"Synthesizing text: {text} {'(greeting)' if is_greeting else ''}")
        start_time = time.time()
        text = clean_text(text)
        if not text:
            logger.warning("Cleaned text is empty; skipping synthesis")
            return
        sentences = [s.strip() for s in text.split('. ') if s.strip()]
        if not sentences:
            logger.warning("No valid sentences after splitting; skipping synthesis")
            return
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for sentence in sentences:
            with torch.inference_mode():
                with torch.cuda.amp.autocast() if device == "cuda" else torch.no_grad():
                    audio = tts.tts(text=sentence, speaker=None)
            sampling_rate = tts.synthesizer.output_sample_rate
            if audio is None or len(audio) == 0:
                logger.error(f"Coqui generated empty audio for sentence: {sentence}")
                continue
            audio = np.array(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            max_amplitude = np.max(np.abs(audio)) if len(audio) > 0 else 0
            logger.info(f"Generated audio shape: {audio.shape}, sampling rate: {sampling_rate}, max amplitude: {max_amplitude:.4f}, generation time: {time.time() - start_time:.2f}s")
            try:
                output_file = os.path.join(OUTPUT_DIR, f"coqui_output_{int(time.time())}.wav")
                sf.write(output_file, audio, sampling_rate)
                logger.info(f"Saved audio to {output_file}")
            except Exception as e:
                logger.error(f"Failed to save {output_file}: {e}")
            try:
                sd.play(audio, sampling_rate, blocking=False)
                duration = len(audio) / sampling_rate
                time.sleep(duration)
                sd.wait()
                logger.info(f"Speech synthesized for sentence: {sentence}")
            except Exception as e:
                logger.error(f"Sounddevice playback error: {e}")
    except Exception as e:
        logger.error(f"Coqui TTS error: {e}")
    finally:
        is_synthesizing = False

def main():
    global tts, device
    try:
        logger.info(f"Using default input device: {sd.query_devices(kind='input')['name']}")
        logger.info(f"Available audio devices: {sd.query_devices()}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            logger.warning("CUDA not available; using CPU")
        else:
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Initializing Coqui TTS pipeline...")
        tts = TTS(model_name=TTS_MODEL, progress_bar=True)
        tts.to(device)
        logger.info("Coqui TTS pipeline initialized successfully.")
        
        # Play greeting message at startup
        if GREETING_MESSAGE:
            logger.info(f"Playing startup greeting: {GREETING_MESSAGE}")
            synthesize_speech(GREETING_MESSAGE, is_greeting=True)
        
        # Start cleanup thread
        def cleanup_old_files():
            while True:
                try:
                    now = time.time()
                    for file in glob.glob(os.path.join(OUTPUT_DIR, "coqui_output_*.wav")):
                        if now - os.path.getctime(file) > FILE_RETENTION_SECONDS:
                            os.remove(file)
                            logger.info(f"Deleted old audio file: {file}")
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
                time.sleep(CLEANUP_INTERVAL)
        
        cleanup_thread = Thread(target=cleanup_old_files)
        cleanup_thread.daemon = True
        cleanup_thread.start()
        logger.info(f"Started cleanup thread: checking every {CLEANUP_INTERVAL} seconds")
        
        audio_thread = Thread(target=capture_audio)
        audio_thread.daemon = True
        audio_thread.start()
        process_audio_and_transcribe()
    except KeyboardInterrupt:
        logger.info("\nStopped by user.")
        if debug_audio_buffer:
            debug_array = np.concatenate(debug_audio_buffer, axis=0).flatten()
            sf.write(DEBUG_AUDIO, debug_array, SAMPLE_RATE)
            logger.info(f"Saved all captured audio to {DEBUG_AUDIO}")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if os.path.exists(TEMP_AUDIO):
            os.remove(TEMP_AUDIO)

if __name__ == "__main__":
    main()