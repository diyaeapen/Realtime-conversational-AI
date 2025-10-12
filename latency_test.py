# --------------------------------------------------------------------------
# --- PHONEMIZER/ESPEAK PATH FIX (FROM YOUR STREAMLIT CODE) ---
import os
import sys
# Set the environment variable to point to the correct DLL
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
try:
    # We patch the phonemizer cleanup function to prevent errors when the DLL is in use.
    import phonemizer.backend.espeak.api as _esapi
    def _noop_delete_win32(self): return None
    _esapi.EspeakAPI._delete_win32 = _noop_delete_win32
except ImportError:
    print("WARNING: phonemizer not installed or fix failed. Kokoro may fail.")
except Exception:
    pass
# --------------------------------------------------------------------------

import time
import torch
import soundfile as sf
import numpy as np
from elevenlabs.client import ElevenLabs 
from kokoro import KPipeline

# --- CONFIGURATION ---voice_id="ErXwobaYiN019PkySvjV", # Antoni - a common default voice
# ⚠️ REPLACE WITH YOUR ACTUAL KEY ⚠️
ELEVENLABS_API_KEY = "sk_5fb4d54231a6cefd7b4c07122a225735d39a981365afb94c" 
TEST_TEXT = "Testing the latency of the quick brown fox jumping over the lazy dog." 
OUTPUT_DIR = "tts_latency_results"
# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Initialize ElevenLabs Client
client = None
if ELEVENLABS_API_KEY != "YOUR_ELEVENLABS_API_KEY":
    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        print("ElevenLabs client initialized.")
    except Exception as e:
        print(f"ElevenLabs client initialization failed: {e}. Skipping API test.")

# --- HELPER FUNCTION FOR LATENCY CHECK ---
def check_latency(model_name, generate_func):
    """Measures the execution time of the text-to-speech function."""
    print(f"\nTesting: {model_name}...")
    
    # Run a warm-up generation (crucial for local models like Kokoro/PyTorch)
    print("   Warming up...")
    generate_func() 
    
    # Actual timed run
    start_time = time.time()
    audio_data, sampling_rate = generate_func()
    end_time = time.time()
    latency = end_time - start_time
    
    # Save the output audio
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)

    output_path = os.path.join(OUTPUT_DIR, f"{model_name.lower().replace(' ', '_').replace('-', '_')}.wav")
    sf.write(output_path, audio_data, sampling_rate)
    print(f"   Audio saved to: {output_path}")

    print(f"✅ {model_name} Latency: {latency:.4f} seconds")
    return latency

# --- MODEL-SPECIFIC GENERATION FUNCTIONS ---

# 1. ElevenLabs (Commercial API)
def generate_elevenlabs():
    # *** FIX APPLIED HERE ***
    # Use the correct method: client.text_to_speech.convert()
    audio = client.text_to_speech.convert(
        text=TEST_TEXT,
        voice_id="ErXwobaYiN019PkySvjV", # Antoni - a common default voice
        model_id="eleven_turbo_v2",      # model_id for convert() method
        output_format="pcm_16000"
    )
    # The API returns audio bytes, which we collect
    audio_bytes = b"".join([chunk for chunk in audio])
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np, 16000 

# 2. Kokoro-82M (Local)
kokoro_pipeline = None
try:
    print("Loading Kokoro model...")
    kokoro_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M', device=device)
    
    KOKORO_VOICE = "af_heart"
    KOKORO_SPEED = 1.0 
    
    def generate_kokoro():
        chunks = []
        for _, _, audio in kokoro_pipeline(TEST_TEXT.strip(), voice=KOKORO_VOICE, speed=KOKORO_SPEED):
            if audio is not None and audio.size:
                chunks.append(np.asarray(audio, dtype=np.float32))
        
        if not chunks:
             raise RuntimeError("Kokoro produced no audio.")
             
        wav = np.concatenate(chunks)
        
        # Normalize the audio
        mx = np.max(np.abs(wav))
        if mx > 0:
            wav = wav / mx * 0.98
            
        return wav, 24000
        
except Exception as e:
    print(f"⚠️ Failed to load Kokoro model: {e}. Skipping.")
    kokoro_pipeline = None 
    
    
# --- EXECUTION AND SUMMARY ---
if __name__ == "__main__":
    
    results = {}
    
    print("\n" + "="*50)
    print("🚀 Starting TTS Latency Benchmark")
    print(f"Text: \"{TEST_TEXT}\"")
    print(f"Hardware: {device}")
    print("="*50)
    
    # 1. Run ElevenLabs test
    if client:
        results['ElevenLabs (API)'] = check_latency("ElevenLabs (API)", generate_elevenlabs)
    
    # 2. Run Kokoro test
    if kokoro_pipeline:
        results['Kokoro-82M (Local)'] = check_latency("Kokoro-82M (Local)", generate_kokoro)
    
    # --- FINAL SUMMARY ---
    print("\n" + "="*50)
    print("✨ LATENCY COMPARISON SUMMARY ✨")
    print("-" * 50)
    
    if not results:
        print("No models were successfully run. Check configuration and error messages.")
    else:
        # Sort and print results
        sorted_results = sorted(results.items(), key=lambda item: item[1])
        
        print(f"| {'MODEL':<25} | {'LATENCY (seconds)':^19} |")
        print("-" * 50)
        for model, latency in sorted_results:
            print(f"| {model:<25} | {latency:^19.4f} |")
    
    print("="*50)