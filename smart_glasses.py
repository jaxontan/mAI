"""AI smart-glasses prototype.

Maps commodity laptop hardware into a simple smart-glasses loop:
camera -> JPEG video blobs -> vision model (OpenRouter or local Ollama) -> HUD caption.
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
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "google/gemma-4-26b-a4b-it:free"
OLLAMA_DEFAULT_MODEL = "llava:7b"


@dataclass(slots=True)
class Settings:
    model: str
    backend: str
    ollama_url: str
    openrouter_url: str
    api_key: str
    ollama_prompt: str
    ollama_interval: float
    speak: bool
    camera_index: int
    jpeg_quality: int
    show_hud: bool
    check: bool


class SmartGlassesRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stop_event = asyncio.Event()

    async def run(self) -> None:
        await self.run_session()

    async def run_session(self) -> None:
        self._install_signal_handlers()
        cv2 = require_cv2()
        cap = cv2.VideoCapture(self.settings.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam index {self.settings.camera_index}."
            )

        backend = self.settings.backend
        if backend == "openrouter":
            print(
                "Connected to OpenRouter vision model. Press q, Esc, or Ctrl+C to stop."
            )
            window_name = "AI Smart Glasses HUD - OpenRouter"
        else:
            print(
                "Connected to local Ollama vision model. "
                "Press q, Esc, or Ctrl+C to stop."
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
                            caption = await asyncio.to_thread(self._call_vision, jpeg)
                            if caption:
                                last_caption = caption
                                print(caption)
                                if self.settings.speak:
                                    await asyncio.to_thread(speak_windows, caption)
                        except Exception as exc:
                            last_caption = f"{backend} error: {exc}"
                            print(last_caption, file=sys.stderr)

                await asyncio.sleep(0.01)
        finally:
            cap.release()
            if self.settings.show_hud:
                cv2.destroyWindow(window_name)

    def _call_vision(self, jpeg: bytes) -> str:
        if self.settings.backend == "openrouter":
            if not self.settings.api_key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is not set; cannot call OpenRouter."
                )
            return call_openrouter(
                self.settings.openrouter_url,
                self.settings.model,
                self.settings.api_key,
                self.settings.ollama_prompt,
                jpeg,
            )
        return call_ollama(
            self.settings.ollama_url,
            self.settings.model,
            self.settings.ollama_prompt,
            jpeg,
        )

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.stop_event.set)
            except (NotImplementedError, RuntimeError):
                continue

    def _encode_jpeg(self, cv2: Any, frame: Any) -> bytes | None:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.settings.jpeg_quality]
        ok, encoded = cv2.imencode(".jpg", frame, params)
        if not ok:
            return None
        return encoded.tobytes()


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


def call_openrouter(
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    jpeg: bytes | None = None,
    timeout: float = 120.0,
) -> str:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if jpeg:
        b64 = base64.b64encode(jpeg).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
        "max_tokens": 80,
        "stream": False,
    }

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"could not reach OpenRouter at {base_url}: {exc}") from exc

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenRouter returned no choices: {data}")
    text = str(choices[0].get("message", {}).get("content") or "").strip()
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
    print(f"camera index: {settings.camera_index}")

    if settings.backend == "openrouter":
        if not settings.api_key:
            warn("OPENROUTER_API_KEY is not set; OpenRouter calls will fail.")
        else:
            ok("OPENROUTER_API_KEY is set.")
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
        input_device = sd.query_devices(None, "input")
        output_device = sd.query_devices(None, "output")
        ok(f"microphone device available: {input_device['name']}")
        ok(f"speaker device available: {output_device['name']}")
    except Exception as exc:
        warn(f"audio device check failed: {exc}")

    try:
        cv2 = require_cv2()
        cap = cv2.VideoCapture(settings.camera_index, cv2.CAP_DSHOW)
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
            try:
                if settings.backend == "openrouter":
                    if not settings.api_key:
                        warn(
                            "openrouter backend selected but OPENROUTER_API_KEY is not set."
                        )
                    else:
                        response = call_openrouter(
                            settings.openrouter_url,
                            settings.model,
                            settings.api_key,
                            "Reply with exactly: remote vision check ok",
                            jpeg,
                        )
                        if response:
                            ok(f"OpenRouter image request returned: {response[:80]}")
                        else:
                            warn("OpenRouter image request returned an empty response")
                else:
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
                warn(f"vision image request failed: {exc}")
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
        description="Run the AI smart-glasses prototype (OpenRouter or local Ollama)."
    )
    parser.add_argument(
        "--backend",
        choices=("openrouter", "ollama"),
        default=os.environ.get("SMART_GLASSES_BACKEND", "openrouter"),
        help="Vision backend to use (default: openrouter).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name. Defaults to google/gemma-4-26b-a4b-it:free for "
            "OpenRouter or llava:7b for Ollama."
        ),
    )
    parser.add_argument(
        "--openrouter-url",
        default=os.environ.get("OPENROUTER_URL", OPENROUTER_API_URL),
        help="Base URL for the OpenRouter API.",
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
        help="Speak responses with Windows SAPI when available.",
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--jpeg-quality", type=int, default=75)
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
    if args.ollama_interval <= 0:
        parser.error("--ollama-interval must be positive")

    if args.model:
        model = args.model
    elif args.backend == "openrouter":
        model = os.environ.get("OPENROUTER_MODEL", OPENROUTER_DEFAULT_MODEL)
    else:
        model = os.environ.get("OLLAMA_MODEL", OLLAMA_DEFAULT_MODEL)

    api_key = os.environ.get("OPENROUTER_API_KEY", "")

    return Settings(
        model=model,
        backend=args.backend,
        ollama_url=args.ollama_url,
        openrouter_url=args.openrouter_url,
        api_key=api_key,
        ollama_prompt=args.ollama_prompt,
        ollama_interval=args.ollama_interval,
        speak=args.speak,
        camera_index=args.camera_index,
        jpeg_quality=args.jpeg_quality,
        show_hud=not args.no_hud,
        check=args.check,
    )


async def amain(argv: list[str]) -> None:
    from dotenv import load_dotenv

    load_dotenv()
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
