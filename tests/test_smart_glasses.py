from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from google.genai import types

from smart_glasses import (
    PlaybackBuffer,
    Settings,
    SmartGlassesRuntime,
    call_ollama,
    dotenv_key_status,
    find_api_key,
)


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        model="gemini-3.1-flash-live-preview",
        backend="gemini",
        ollama_url="http://localhost:11434",
        ollama_prompt="describe the view",
        ollama_interval=5.0,
        speak=False,
        camera_index=0,
        jpeg_quality=75,
        video_fps=1.0,
        mic_device=None,
        speaker_device=None,
        mic_rate=16000,
        speaker_rate=24000,
        chunk_ms=40,
        queue_seconds=3.0,
        notepad_path=tmp_path / "memory.jsonl",
        enable_search=True,
        enable_notepad=True,
        show_hud=True,
        api_key="test-key",
        check=False,
    )


def test_live_config_enables_audio_vad_search_and_notepad(tmp_path: Path) -> None:
    runtime = SmartGlassesRuntime(make_settings(tmp_path))

    config = runtime._build_live_config()
    parsed = types.LiveConnectConfig(**config)

    assert parsed.response_modalities == [types.Modality.AUDIO]
    assert parsed.realtime_input_config is not None
    vad = parsed.realtime_input_config.automatic_activity_detection
    assert vad is not None
    assert vad.disabled is False
    assert parsed.tools is not None
    assert parsed.tools[0].google_search is not None
    declarations = parsed.tools[1].function_declarations
    assert declarations is not None
    assert declarations[0].name == "save_to_notepad"


def test_playback_buffer_reads_and_pads_pcm() -> None:
    buffer = PlaybackBuffer(max_bytes=8)
    buffer.push(b"abcd")

    assert buffer.read(6) == b"abcd\x00\x00"


def test_playback_buffer_drops_old_audio_when_full() -> None:
    buffer = PlaybackBuffer(max_bytes=4)
    buffer.push(b"abcd")
    buffer.push(b"efgh")

    assert buffer.read(4) == b"efgh"


def test_save_to_notepad_writes_jsonl_memory(tmp_path: Path) -> None:
    runtime = SmartGlassesRuntime(make_settings(tmp_path))

    result = runtime._save_to_notepad(
        {"note": "Keys are on the right side of the desk.", "item": "keys"}
    )

    assert result["result"] == "saved"
    contents = (tmp_path / "memory.jsonl").read_text(encoding="utf-8")
    assert "Keys are on the right side of the desk." in contents


def test_find_api_key_loads_dotenv_without_printing_value(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text('GOOGLE_API_KEY="abc123"\n', encoding="utf-8")

    assert find_api_key(env_file) == "abc123"
    assert dotenv_key_status(env_file) == ".env contains GOOGLE_API_KEY"


def test_find_api_key_accepts_export_dotenv_syntax(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("export GEMINI_API_KEY='abc123'\n", encoding="utf-8")

    assert find_api_key(env_file) == "abc123"
    assert dotenv_key_status(env_file) == ".env contains GEMINI_API_KEY"


def test_parse_args_defaults_ollama_model(monkeypatch) -> None:
    import smart_glasses

    monkeypatch.delenv("SMART_GLASSES_BACKEND", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    settings = smart_glasses.parse_args([])

    assert settings.backend == "ollama"
    assert settings.model == "gemma4:e4b"


def test_mouth_engine_restarts_receive_for_multiple_turns(tmp_path: Path) -> None:
    class FakeSession:
        def __init__(self) -> None:
            self.calls = 0

        async def receive(self):
            self.calls += 1
            if self.calls == 1:
                yield SimpleNamespace(
                    data=b"first",
                    server_content=SimpleNamespace(turn_complete=True),
                    tool_call=None,
                )
            elif self.calls == 2:
                yield SimpleNamespace(
                    data=b"second",
                    server_content=SimpleNamespace(turn_complete=True),
                    tool_call=None,
                )
            else:
                runtime.stop_event.set()
                return

    runtime = SmartGlassesRuntime(make_settings(tmp_path))
    fake_session = FakeSession()

    asyncio.run(runtime.mouth_engine(fake_session, types))

    assert fake_session.calls == 3
    assert runtime.playback.read(11) == b"firstsecond"
