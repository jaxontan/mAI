from __future__ import annotations

from pathlib import Path

from smart_glasses import Settings


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        model="llava:7b",
        ollama_url="http://localhost:11434",
        ollama_prompt="describe the view",
        ollama_interval=5.0,
        speak=False,
        camera_index=0,
        jpeg_quality=75,
        show_hud=True,
        check=False,
    )


def test_parse_args_defaults_ollama_model(monkeypatch) -> None:
    import smart_glasses

    monkeypatch.delenv("SMART_GLASSES_BACKEND", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    settings = smart_glasses.parse_args([])
    assert settings.model == "llava:7b"
