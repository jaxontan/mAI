# AI Smart Glasses Blueprint

This prototype maps a laptop into a pair of AI smart glasses using a local Ollama vision model.

- Webcam: forward POV camera, mirrored in an OpenCV HUD and sent as JPEG snapshots.
- Ollama vision model: describes the scene in one sentence at configurable intervals.
- Optional Windows TTS: speaks responses aloud with `--speak`.

## Setup

Use Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Pull a vision-capable Ollama model:

```powershell
ollama pull llava:7b
```

## Run

Check local dependencies, devices, and one webcam frame first:

```powershell
python smart_glasses.py --check
```

```powershell
python smart_glasses.py
```

Press `q`, `Esc`, or `Ctrl+C` to stop.

## Options

```powershell
python smart_glasses.py --model moondream:latest
python smart_glasses.py --camera-index 1
python smart_glasses.py --jpeg-quality 65
python smart_glasses.py --ollama-interval 3.0
python smart_glasses.py --no-hud
python smart_glasses.py --speak
```

List audio devices:

```powershell
python audio_devices.py
```

## Architecture

`smart_glasses.py` runs a single async loop:

- Captures webcam frames via OpenCV.
- Mirrors the feed in a local HUD window with the latest caption overlay.
- Sends JPEG snapshots to Ollama at configurable intervals.
- Prints model responses and optionally speaks them via Windows SAPI.

## Tests

```powershell
python -m pip install -r requirements.txt pytest
python -m pytest
```

Tests cover config parsing and local checks. They do not open a camera, microphone, speaker, or network connection.
