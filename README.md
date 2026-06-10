# AI Smart Glasses Blueprint

This prototype maps a laptop into a pair of AI smart glasses. It defaults to local Ollama mode to avoid Gemini quota failures.

- Webcam: forward POV camera, mirrored in an OpenCV HUD and sent as JPEG at 1 FPS.
- Microphone: frame mic, captured as raw 16 kHz PCM mono audio.
- Headphones or speakers: bone-conduction stems, playing raw 24 kHz PCM mono audio.
- Gemini Live: one stateful bidirectional WebSocket session carrying audio, video, tool calls, and streamed model audio.

## Setup

Use Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Set a Gemini API key:

```powershell
$env:GOOGLE_API_KEY = "your-api-key"
```

`GEMINI_API_KEY` is also accepted for Gemini mode. The Gemini model is `gemini-3.1-flash-live-preview`; override it with `GEMINI_LIVE_MODEL` or `--model` if your account is enabled for a different Live model.

## Run

Check local dependencies, devices, API-key presence, and one webcam frame first:

```powershell
python smart_glasses.py --check
```

```powershell
python smart_glasses.py
```

By default this runs local Ollama mode with `gemma4:e4b`. Press `q`, `Esc`, or `Ctrl+C` to stop.

## Local Ollama Mode

If Gemini Live quota is unavailable, run local webcam vision snapshots through Ollama:

```powershell
ollama pull gemma4:e4b
python smart_glasses.py --check
python smart_glasses.py
```

Ollama mode keeps the OpenCV HUD and sends webcam JPEG snapshots to the local model. It is not the same as Gemini Live: it does not provide server-side VAD or streamed 24 kHz model audio. Add `--speak` to read local responses through Windows speech synthesis when available:

```powershell
python smart_glasses.py --backend ollama --speak
```

## Gemini Live Mode

Use Gemini explicitly only when your API quota is available:

```powershell
python smart_glasses.py --backend gemini --check
python smart_glasses.py --backend gemini
```

If your default microphone or speaker is wrong, list devices:

```powershell
python audio_devices.py
```

Then pass the indexes:

```powershell
python smart_glasses.py --mic-device 1 --speaker-device 3
```

Useful options:

```powershell
python smart_glasses.py --no-hud
python smart_glasses.py --camera-index 1
python smart_glasses.py --jpeg-quality 65
python smart_glasses.py --notepad-path .\memory\glasses.jsonl
```

## Architecture

`smart_glasses.py` starts five async tasks inside a single Gemini Live session:

- Eye engine: pulls OpenCV frames, mirrors them locally, and sends one compressed JPEG per second.
- Ear capture engine: captures microphone input with a non-blocking `sounddevice` raw stream.
- Ear send engine: forwards small 16 kHz PCM audio chunks to Gemini Live for server-side VAD.
- Mouth engine: receives Gemini Live messages, clears playback on interruptions, handles function calls, and queues model audio.
- Speaker engine: drains the playback buffer into a 24 kHz raw PCM mono output stream.

The session config requests audio-only model responses, enables automatic activity detection, includes the smart-glasses persona, attaches Google Search, and registers `save_to_notepad`.

## Memory Tool

When the model calls `save_to_notepad`, the note is appended as JSONL to `notepad_memory.jsonl` by default. Example record:

```json
{"saved_at":"2026-06-04T08:00:00+00:00","note":"Keys are on the right side of the desk.","item":"keys","location":null}
```

## Tests

Install the runtime dependency and pytest, then run:

```powershell
python -m pip install -r requirements.txt pytest
python -m pytest
```

The tests cover config validation, playback buffering, local memory writes, and the continuous Gemini Live receive loop. They do not open a camera, microphone, speaker, or WebSocket.
