Real-Time Conversational AI
Text-to-Speech + Web Interface (Flask/Socket.IO) + Latency Benchmark
====================================================================

REPO STRUCTURE
--------------
.
├─ README.txt                (this file)
├─ latency_test.py           (latency benchmark for Kokoro vs ElevenLabs)
└─ pipeline_Code/
   ├─ testing_final.py       (Flask + Socket.IO real-time voice assistant)
   └─ templates/
      └─ index.html          (web UI: status, logs, STOP, metrics)

WHAT THIS PROJECT DOES
----------------------
- Real-time voice assistant: mic → Whisper STT → LLM → ElevenLabs TTS → playback.
- Full-duplex streaming with a reliable STOP/interrupt that cancels mid-sentence cleanly.
- Echo control (gating + similarity suppression) and short post-TTS mute.
- Clause-level speaking; optional gap-filler to reduce perceived delay.
- Live latency telemetry (STT, first LLM token, first TTS audio, total pipeline).
- Latency benchmark script to compare Kokoro (local) vs ElevenLabs (API).

PREREQUISITES
-------------
- Python 3.9 – 3.11
- Microphone and speaker access
- Create a virtual environment before installing packages

INSTALL
-------
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install --upgrade pip

# Core dependencies
pip install flask flask-socketio python-dotenv openai==1.* sounddevice numpy soundfile
# Whisper (choose one appropriate for your environment)
pip install openai-whisper
# ElevenLabs SDK
pip install elevenlabs==1.*

# Optional for local Kokoro tests (adjust torch wheel for your CUDA/CPU):
# pip install kokoro-tts phonemizer torch --index-url https://download.pytorch.org/whl/cu121

ENVIRONMENT VARIABLES
---------------------
Create pipeline_Code/.env with:
OPENAI_API_KEY=sk-xxxx
ELEVENLABS_API_KEY=eleven-xxxx
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

RUN THE REAL-TIME ASSISTANT (FLASK)
-----------------------------------
cd pipeline_Code
python testing_final.py
# Open the printed URL (default http://127.0.0.1:5000/)

UI CONTROLS
-----------
- Stop AI Response: cancels current TTS/LLM, drains queues, hard-resets audio to prevent tail-end playback
- Stop/Resume Listening: pause/resume mic ingestion
- Shut Down: graceful shutdown
The page shows current state (Listening / Transcribing / Speaking) and latency stats in real time.

RUNTIME DESIGN (3 QUEUES / 3 THREADS)
-------------------------------------
- audio_q: mic frames (16 kHz) → Whisper STT
- tts_synth_q: clause text / gap-fillers → ElevenLabs TTS (24 kHz PCM)
- play_q: PCM chunks / start-end markers → device writer
STOP = cancellation watermark + queue drain + hard output reset

RUN THE LATENCY BENCHMARK
-------------------------
python latency_test.py
- Compares local Kokoro vs ElevenLabs on fixed text.
- Saves WAVs and prints measured times (edit the script to change text/voices).
- Use short Mozilla Common Voice clips for consistent, non-identifying tests if desired.

TROUBLESHOOTING
---------------
- No audio / device errors: set your OS default output device; the playback thread will try to reopen and resample.
- Tail-end audio after STOP: ensure you run testing_final.py from this repo; STOP drains queues and hard-resets output.
- STT hears itself: reduce speaker volume. The app gates mic during/after TTS and drops transcripts similar to the assistant.

CUSTOMIZATION
-------------
- Change TTS voice: set ELEVENLABS_VOICE_ID in .env.
- Adjust VAD/segmentation: edit RMS/VAD constants in testing_final.py.
- Modify UI text/styles: templates/index.html.

PRIVACY
-------
No persistent storage of user audio/transcripts. Only minimal latency metrics are shown in the UI.

LICENSE
-------
Add your preferred license (e.g., MIT) to the repository root.
