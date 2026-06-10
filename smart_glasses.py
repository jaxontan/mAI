"""AI smart-glasses prototype.

Maps commodity laptop hardware into a simple smart-glasses loop:
camera -> JPEG video blobs, mic -> 16 kHz PCM blobs for Gemini Live, model
audio -> 24 kHz PCM playback, and an OpenCV camera mirror as the local HUD.
Defaults to local Ollama snapshots to avoid Gemini quota issues.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from google.genai import errors as genai_errors
except Exception:
    genai_errors = None


SYSTEM_INSTRUCTION = """
You are the companion AI inside a pair of smart glasses. The user is wearing
you. Act as their extra eyes and ears. Keep spoken responses conversational,
tight, and limited to 1 or 2 sentences. Reference objects spatially based on
the webcam feed, such as to your left, to your right, or in front of you.
Use Google Search when the user asks for current facts about something they are
facing. When asked to remember where an item is, call save_to_notepad.
""".strip()


SAVE_TO_NOTEPAD_DECLARATION: dict[str, Any] = {
    "name": "save_to_notepad",
    "description": (
        "Save a short note, object location, or reminder to the local smart "
        "glasses notepad."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "note": {
                "type": "string",
                "description": "The exact short memory to save.",
            },
            "item": {
                "type": "string",
                "description": "Optional item being remembered.",
            },
            "location": {
                "type": "string",
                "description": "Optional location of the item.",
            },
        },
        "required": ["note"],
    },
}


@dataclass(slots=True)
class Settings:
    model: str
    backend: str
    ollama_url: str
    ollama_prompt: str
    ollama_interval: float
    speak: bool
    camera_index: int
    jpeg_quality: int
    video_fps: float
    mic_device: int | None
    speaker_device: int | None
    mic_rate: int
    speaker_rate: int
    chunk_ms: int
    queue_seconds: float
    notepad_path: Path
    enable_search: bool
    enable_notepad: bool
    show_hud: bool
    api_key: str | None
    check: bool

    @property
    def chunk_frames(self) -> int:
        return max(1, int(self.mic_rate * self.chunk_ms / 1000))

    @property
    def playback_block_frames(self) -> int:
        return max(1, int(self.speaker_rate * self.chunk_ms / 1000))


class PlaybackBuffer:
    """Thread-safe byte buffer consumed by the sounddevice output callback."""

    def __init__(self, max_bytes: int) -> None:
        self._chunks: deque[bytearray] = deque()
        self._size = 0
        self._max_bytes = max_bytes
        self._lock = threading.Lock()

    def push(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            self._chunks.append(bytearray(data))
            self._size += len(data)
            while self._size > self._max_bytes and self._chunks:
                dropped = self._chunks.popleft()
                self._size -= len(dropped)

    def read(self, nbytes: int) -> bytes:
        out = bytearray()
        with self._lock:
            while len(out) < nbytes and self._chunks:
                head = self._chunks[0]
                needed = nbytes - len(out)
                out.extend(head[:needed])
                if len(head) <= needed:
                    self._chunks.popleft()
                    self._size -= len(head)
                else:
                    del head[:needed]
                    self._size -= needed
        if len(out) < nbytes:
            out.extend(b"\x00" * (nbytes - len(out)))
        return bytes(out)

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._size = 0


class SmartGlassesRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=80)
        max_playback_bytes = int(settings.speaker_rate * 2 * settings.queue_seconds)
        self.playback = PlaybackBuffer(max_bytes=max_playback_bytes)
        self.session_send_lock = asyncio.Lock()
        self.stop_event = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    async def run(self) -> None:
        if self.settings.backend == "ollama":
            await self.run_ollama()
            return

        self._loop = asyncio.get_running_loop()
        self._install_signal_handlers()

        genai, types = require_genai()
        api_key = self.settings.api_key
        if not api_key:
            raise RuntimeError(
                "Set GOOGLE_API_KEY or GEMINI_API_KEY before running the glasses."
            )

        client = genai.Client(api_key=api_key)
        config = self._build_live_config()

        try:
            connection = client.aio.live.connect(
                model=self.settings.model, config=config
            )
            session_context = connection
            async with session_context as session:
                print(
                    "Connected to Gemini Live. Speak normally; press Ctrl+C "
                    "or close the HUD to stop."
                )
                tasks = [
                    asyncio.create_task(self.eye_engine(session, types)),
                    asyncio.create_task(self.ear_capture_engine()),
                    asyncio.create_task(self.ear_send_engine(session, types)),
                    asyncio.create_task(self.mouth_engine(session, types)),
                    asyncio.create_task(self.speaker_engine()),
                    asyncio.create_task(self._wait_for_stop()),
                ]
                await self._run_until_first_done(tasks)
        except Exception as exc:
            if is_quota_error(exc):
                raise RuntimeError(
                    "Gemini Live connected but your API key is over quota. "
                    "Run local mode with `python smart_glasses.py` or "
                    "`python smart_glasses.py --backend ollama`; use "
                    "`--backend gemini` only after quota/billing is fixed."
                ) from exc
            raise

    async def run_ollama(self) -> None:
        self._install_signal_handlers()
        cv2 = require_cv2()
        cap = cv2.VideoCapture(self.settings.camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam index {self.settings.camera_index}."
            )

        print(
            "Connected to local Ollama mode. Press q, Esc, or Ctrl+C to stop."
        )
        window_name = "AI Smart Glasses HUD - Ollama"
        next_send = 0.0
        last_caption = ""

        try:
            while not self.stop_event.is_set():
                ok, frame = await asyncio.to_thread(cap.read)
                if not ok:
                    await asyncio.sleep(0.05)
                    continue

                display_frame = frame.copy()
                if last_caption:
                    draw_caption(cv2, display_frame, last_caption)

                if self.settings.show_hud:
                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        self.stop_event.set()
                        return

                now = time.monotonic()
                if now >= next_send:
                    next_send = now + self.settings.ollama_interval
                    jpeg = await asyncio.to_thread(self._encode_jpeg, cv2, frame)
                    if jpeg:
                        try:
                            caption = await asyncio.to_thread(
                                call_ollama,
                                self.settings.ollama_url,
                                self.settings.model,
                                self.settings.ollama_prompt,
                                jpeg,
                            )
                            if caption:
                                last_caption = caption
                                print(caption)
                                if self.settings.speak:
                                    await asyncio.to_thread(speak_windows, caption)
                        except Exception as exc:
                            last_caption = f"Ollama error: {exc}"
                            print(last_caption, file=sys.stderr)

                await asyncio.sleep(0.01)
        finally:
            cap.release()
            if self.settings.show_hud:
                cv2.destroyWindow(window_name)

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop_event.set)
            except (NotImplementedError, RuntimeError):
                continue

    def _build_live_config(self) -> dict[str, Any]:
        tools: list[dict[str, Any]] = []
        if self.settings.enable_search:
            tools.append({"google_search": {}})
        if self.settings.enable_notepad:
            tools.append({"function_declarations": [SAVE_TO_NOTEPAD_DECLARATION]})

        config: dict[str, Any] = {
            "response_modalities": ["AUDIO"],
            "system_instruction": SYSTEM_INSTRUCTION,
            "realtime_input_config": {
                "automatic_activity_detection": {
                    "disabled": False,
                    "prefix_padding_ms": 120,
                    "silence_duration_ms": 650,
                }
            },
            "media_resolution": "MEDIA_RESOLUTION_LOW",
        }
        if tools:
            config["tools"] = tools
        return config

    async def _run_until_first_done(self, tasks: list[asyncio.Task[Any]]) -> None:
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                try:
                    exc = task.exception()
                except asyncio.CancelledError:
                    continue
                if exc is not None and not isinstance(exc, asyncio.CancelledError):
                    raise exc
        finally:
            for task in tasks:
                task.cancel()

    async def _wait_for_stop(self) -> None:
        await self.stop_event.wait()

    async def eye_engine(self, session: Any, types: Any) -> None:
        cv2 = require_cv2()
        cap = cv2.VideoCapture(self.settings.camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam index {self.settings.camera_index}."
            )

        window_name = "AI Smart Glasses HUD"
        send_interval = 1.0 / max(self.settings.video_fps, 0.001)
        next_send = 0.0

        try:
            while not self.stop_event.is_set():
                ok, frame = await asyncio.to_thread(cap.read)
                if not ok:
                    await asyncio.sleep(0.05)
                    continue

                if self.settings.show_hud:
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        self.stop_event.set()
                        return

                now = asyncio.get_running_loop().time()
                if now >= next_send:
                    next_send = now + send_interval
                    jpeg = await asyncio.to_thread(self._encode_jpeg, cv2, frame)
                    if jpeg:
                        await self._send_realtime_input(
                            session,
                            video=types.Blob(data=jpeg, mime_type="image/jpeg"),
                        )

                await asyncio.sleep(0.005)
        finally:
            cap.release()
            if self.settings.show_hud:
                cv2.destroyWindow(window_name)

    def _encode_jpeg(self, cv2: Any, frame: Any) -> bytes | None:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.jpeg_quality]
        ok, encoded = cv2.imencode(".jpg", frame, params)
        if not ok:
            return None
        return encoded.tobytes()

    async def ear_capture_engine(self) -> None:
        sd = require_sounddevice()
        loop = asyncio.get_running_loop()

        def enqueue(data: bytes) -> None:
            try:
                self.mic_queue.put_nowait(data)
            except asyncio.QueueFull:
                try:
                    self.mic_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                self.mic_queue.put_nowait(data)

        def callback(indata: bytes, _frames: int, _time: Any, status: Any) -> None:
            if status:
                print(f"Microphone status: {status}", file=sys.stderr)
            loop.call_soon_threadsafe(enqueue, bytes(indata))

        stream = sd.RawInputStream(
            samplerate=self.settings.mic_rate,
            blocksize=self.settings.chunk_frames,
            channels=1,
            dtype="int16",
            device=self.settings.mic_device,
            callback=callback,
        )
        stream.start()
        try:
            await self.stop_event.wait()
        finally:
            stream.stop()
            stream.close()

    async def ear_send_engine(self, session: Any, types: Any) -> None:
        mime_type = f"audio/pcm;rate={self.settings.mic_rate}"
        while not self.stop_event.is_set():
            data = await self.mic_queue.get()
            await self._send_realtime_input(
                session, audio=types.Blob(data=data, mime_type=mime_type)
            )

    async def _send_realtime_input(self, session: Any, **kwargs: Any) -> None:
        async with self.session_send_lock:
            await session.send_realtime_input(**kwargs)

    async def mouth_engine(self, session: Any, types: Any) -> None:
        while not self.stop_event.is_set():
            saw_message = False
            async for response in session.receive():
                saw_message = True
                if self.stop_event.is_set():
                    return

                server_content = getattr(response, "server_content", None)
                if server_content and getattr(server_content, "interrupted", False):
                    self.playback.clear()

                wrote_inline_part = self._push_inline_audio(server_content)
                if not wrote_inline_part:
                    data = getattr(response, "data", None)
                    if isinstance(data, bytes):
                        self.playback.push(data)

                tool_call = getattr(response, "tool_call", None)
                if tool_call:
                    await self._handle_tool_call(session, types, tool_call)
            if not saw_message:
                await asyncio.sleep(0.05)

    def _push_inline_audio(self, server_content: Any) -> bool:
        if not server_content:
            return False
        model_turn = getattr(server_content, "model_turn", None)
        parts = getattr(model_turn, "parts", None)
        if not parts:
            return False

        wrote_audio = False
        for part in parts:
            inline_data = getattr(part, "inline_data", None)
            data = getattr(inline_data, "data", None)
            if isinstance(data, bytes):
                self.playback.push(data)
                wrote_audio = True
        return wrote_audio

    async def _handle_tool_call(self, session: Any, types: Any, tool_call: Any) -> None:
        responses = []
        for call in getattr(tool_call, "function_calls", []) or []:
            name = getattr(call, "name", "")
            call_id = getattr(call, "id", None)
            args = self._extract_function_args(call)
            result = await asyncio.to_thread(self._execute_function, name, args)
            responses.append(
                types.FunctionResponse(id=call_id, name=name, response=result)
            )
        if responses:
            async with self.session_send_lock:
                await session.send_tool_response(function_responses=responses)

    def _extract_function_args(self, call: Any) -> dict[str, Any]:
        raw_args = getattr(call, "args", None)
        if raw_args is None:
            raw_args = getattr(call, "arguments", None)
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _execute_function(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "save_to_notepad":
            return self._save_to_notepad(args)
        return {"result": "error", "message": f"Unknown function: {name}"}

    def _save_to_notepad(self, args: dict[str, Any]) -> dict[str, Any]:
        note = str(args.get("note") or "").strip()
        item = str(args.get("item") or "").strip()
        location = str(args.get("location") or "").strip()
        if not note and item and location:
            note = f"{item}: {location}"
        if not note:
            return {"result": "error", "message": "No note was provided."}

        self.settings.notepad_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "note": note,
            "item": item or None,
            "location": location or None,
        }
        with self.settings.notepad_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return {
            "result": "saved",
            "path": str(self.settings.notepad_path),
            "note": note,
        }

    async def speaker_engine(self) -> None:
        sd = require_sounddevice()

        def callback(outdata: bytearray, _frames: int, _time: Any, status: Any) -> None:
            if status:
                print(f"Speaker status: {status}", file=sys.stderr)
            outdata[:] = self.playback.read(len(outdata))

        stream = sd.RawOutputStream(
            samplerate=self.settings.speaker_rate,
            blocksize=self.settings.playback_block_frames,
            channels=1,
            dtype="int16",
            device=self.settings.speaker_device,
            callback=callback,
        )
        stream.start()
        try:
            await self.stop_event.wait()
        finally:
            stream.stop()
            stream.close()


def require_genai() -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Missing google-genai. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return genai, types


def is_quota_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if "quota" in message or "billing" in message:
        return True
    if genai_errors is not None and isinstance(exc, getattr(genai_errors, "APIError")):
        return "quota" in message or "billing" in message
    return False


def require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Missing opencv-python. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return cv2


def require_sounddevice() -> Any:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Missing sounddevice. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc
    return sd


def call_ollama(
    base_url: str,
    model: str,
    prompt: str,
    jpeg: bytes | None = None,
    timeout: float = 120.0,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 80,
        },
    }
    if jpeg:
        payload["images"] = [base64.b64encode(jpeg).decode("ascii")]

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach Ollama at {base_url}: {exc}") from exc

    text = str(data.get("response") or "").strip()
    return " ".join(text.split())


def get_ollama_models(base_url: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
    with urllib.request.urlopen(request, timeout=10.0) as response:
        data = json.loads(response.read().decode("utf-8"))
    models = data.get("models", [])
    return models if isinstance(models, list) else []


def speak_windows(text: str) -> None:
    if sys.platform != "win32" or not text:
        return
    try:
        import win32com.client  # type: ignore[import-untyped]

        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        speaker.Speak(text)
    except Exception:
        escaped = text.replace("'", "''")
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Speak('{escaped}')"
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
        except Exception:
            return


def draw_caption(cv2: Any, frame: Any, caption: str) -> None:
    text = caption[:180]
    height, width = frame.shape[:2]
    y = max(32, height - 42)
    cv2.rectangle(frame, (10, y - 28), (width - 10, height - 10), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def find_api_key(env_path: Path = Path(".env")) -> str | None:
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    if not env_path.exists():
        return None

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for line in lines:
        stripped = line.strip().lstrip("\ufeff")
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.lower().startswith("export "):
            stripped = stripped[7:].strip()
        name, value = stripped.split("=", 1)
        name = name.strip()
        if name not in {"GOOGLE_API_KEY", "GEMINI_API_KEY"}:
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(name, value)
            return value
    return None


def dotenv_key_status(env_path: Path = Path(".env")) -> str:
    if not env_path.exists():
        return ".env not found"
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return f".env exists but could not be read: {exc}"

    supported = []
    for line in lines:
        stripped = line.strip().lstrip("\ufeff")
        if stripped.lower().startswith("export "):
            stripped = stripped[7:].strip()
        if "=" not in stripped:
            continue
        name = stripped.split("=", 1)[0].strip()
        if name in {"GOOGLE_API_KEY", "GEMINI_API_KEY"}:
            supported.append(name)

    if supported:
        return f".env contains {', '.join(supported)}"
    return ".env exists but has no GOOGLE_API_KEY or GEMINI_API_KEY entry"


def run_local_checks(settings: Settings) -> int:
    failures = 0

    def ok(message: str) -> None:
        print(f"[ok] {message}")

    def warn(message: str) -> None:
        nonlocal failures
        failures += 1
        print(f"[warn] {message}")

    print("Smart glasses local check")
    print(f"backend: {settings.backend}")
    print(f"model: {settings.model}")
    print(
        "streams: "
        f"camera={settings.camera_index}, video_fps={settings.video_fps}, "
        f"mic_rate={settings.mic_rate}, speaker_rate={settings.speaker_rate}"
    )

    if settings.backend == "gemini":
        if settings.api_key:
            ok("Gemini API key found in environment or .env")
        else:
            warn(
                "Gemini API key not found; set GOOGLE_API_KEY or GEMINI_API_KEY "
                f"({dotenv_key_status()})"
            )

        try:
            _genai, types = require_genai()
            runtime = SmartGlassesRuntime(settings)
            types.LiveConnectConfig(**runtime._build_live_config())
            ok("google-genai imports and Live config validates")
        except Exception as exc:
            warn(f"Gemini SDK/config check failed: {exc}")
    else:
        try:
            models = get_ollama_models(settings.ollama_url)
            names = {str(model.get("name") or model.get("model")) for model in models}
            if settings.model in names:
                ok(f"Ollama model available: {settings.model}")
            else:
                warn(f"Ollama model {settings.model!r} not found; installed={names}")
        except Exception as exc:
            warn(f"Ollama API check failed: {exc}")

    try:
        sd = require_sounddevice()
        input_device = sd.query_devices(settings.mic_device, "input")
        output_device = sd.query_devices(settings.speaker_device, "output")
        ok(f"microphone device available: {input_device['name']}")
        ok(f"speaker device available: {output_device['name']}")
    except Exception as exc:
        warn(f"audio device check failed: {exc}")

    try:
        cv2 = require_cv2()
        cap = cv2.VideoCapture(settings.camera_index)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"camera index {settings.camera_index} did not open")
            success, frame = cap.read()
            if not success:
                raise RuntimeError("camera opened but did not return a frame")
            jpeg = SmartGlassesRuntime(settings)._encode_jpeg(cv2, frame)
            if not jpeg:
                raise RuntimeError("camera frame could not be JPEG encoded")
            ok(f"camera frame captured and JPEG encoded ({len(jpeg)} bytes)")
            if settings.backend == "ollama":
                try:
                    response = call_ollama(
                        settings.ollama_url,
                        settings.model,
                        "Reply with exactly: local vision check ok",
                        jpeg,
                    )
                    if response:
                        ok(f"Ollama image request returned: {response[:80]}")
                    else:
                        warn("Ollama image request returned an empty response")
                except Exception as exc:
                    warn(f"Ollama image request failed: {exc}")
        finally:
            cap.release()
    except Exception as exc:
        warn(f"camera check failed: {exc}")

    if failures:
        print(f"Check finished with {failures} warning(s).")
        return 1
    print("Check finished cleanly.")
    return 0


def parse_args(argv: list[str]) -> Settings:
    parser = argparse.ArgumentParser(
        description="Run the Gemini Live AI smart-glasses prototype."
    )
    parser.add_argument(
        "--backend",
        choices=("gemini", "ollama"),
        default=os.environ.get("SMART_GLASSES_BACKEND", "ollama"),
        help="Runtime backend. Gemini gives realtime audio/video; Ollama is local vision snapshots.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Gemini Live model name.",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        help="Base URL for the local Ollama server.",
    )
    parser.add_argument(
        "--ollama-prompt",
        default=(
            "You are the companion AI inside smart glasses. Describe what is "
            "important in this webcam view in one short sentence. Use spatial "
            "language such as left, right, or in front when useful."
        ),
        help="Prompt used for each Ollama webcam snapshot.",
    )
    parser.add_argument(
        "--ollama-interval",
        type=float,
        default=5.0,
        help="Seconds between Ollama webcam snapshots.",
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help="In Ollama mode, speak responses with Windows SAPI when available.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--mic-device", type=int, default=None)
    parser.add_argument("--speaker-device", type=int, default=None)
    parser.add_argument("--mic-rate", type=int, default=16000)
    parser.add_argument("--speaker-rate", type=int, default=24000)
    parser.add_argument("--chunk-ms", type=int, default=40)
    parser.add_argument("--queue-seconds", type=float, default=3.0)
    parser.add_argument(
        "--notepad-path",
        type=Path,
        default=Path("notepad_memory.jsonl"),
        help="JSONL file used by the save_to_notepad tool.",
    )
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Disable the Google Search grounding tool.",
    )
    parser.add_argument(
        "--no-notepad",
        action="store_true",
        help="Disable the local save_to_notepad function tool.",
    )
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="Do not show the OpenCV camera mirror window.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run local dependency/device/config checks and exit.",
    )
    args = parser.parse_args(argv)

    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if args.video_fps <= 0 or args.video_fps > 1:
        parser.error("--video-fps must be > 0 and <= 1 for Gemini Live video input")
    if args.mic_rate <= 0 or args.speaker_rate <= 0:
        parser.error("Audio sample rates must be positive")
    if args.chunk_ms <= 0:
        parser.error("--chunk-ms must be positive")
    if args.queue_seconds <= 0:
        parser.error("--queue-seconds must be positive")
    if args.ollama_interval <= 0:
        parser.error("--ollama-interval must be positive")

    if args.model:
        model = args.model
    elif args.backend == "ollama":
        model = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
    else:
        model = os.environ.get("GEMINI_LIVE_MODEL", "gemini-3.1-flash-live-preview")

    return Settings(
        model=model,
        backend=args.backend,
        ollama_url=args.ollama_url,
        ollama_prompt=args.ollama_prompt,
        ollama_interval=args.ollama_interval,
        speak=args.speak,
        camera_index=args.camera_index,
        jpeg_quality=args.jpeg_quality,
        video_fps=args.video_fps,
        mic_device=args.mic_device,
        speaker_device=args.speaker_device,
        mic_rate=args.mic_rate,
        speaker_rate=args.speaker_rate,
        chunk_ms=args.chunk_ms,
        queue_seconds=args.queue_seconds,
        notepad_path=args.notepad_path,
        enable_search=not args.no_search,
        enable_notepad=not args.no_notepad,
        show_hud=not args.no_hud,
        api_key=find_api_key(),
        check=args.check,
    )


async def amain(argv: list[str]) -> None:
    settings = parse_args(argv)
    if settings.check:
        raise SystemExit(run_local_checks(settings))
    runtime = SmartGlassesRuntime(settings)
    await runtime.run()


def main() -> None:
    try:
        asyncio.run(amain(sys.argv[1:]))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
