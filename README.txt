Real-Time Conversational AI
Text-to-Speech + Web Interface (Flask/Socket.IO) + Latency Benchmark
====================================================================

REPO STRUCTURE
--------------
.
├─ README.txt                (this file)
├─ requirements.txt          (pinned dependencies for quick setup)
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

QUICK START
-----------
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies from requirements.txt
pip install -r requirements.txt

# (Optional) If you need an async backend for Socket.IO:
# pip install eventlet  # or: pip install gevent

# (Optional) For local Kokoro benchmarking, also install:
# pip install kokoro-tts phonemizer torch

# 3) Configure environment variables (create pipeline_Code/.env)
# Example:
# OPENAI_API_KEY=sk-xxxx
# ELEVENLABS_API_KEY=eleven-xxxx
# ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# 4) Run the real-time assistant (Flask)
cd pipeline_Code
python testing_final.py  # then open http://127.0.0.1:5000/

PREREQUISITES
-------------
- Python 3.9 – 3.11
- Microphone and speaker access

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
