"""List sounddevice input and output devices for smart_glasses.py."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import sounddevice as sd
    except ImportError:
        print(
            "Missing sounddevice. Install dependencies with "
            "`pip install -r requirements.txt`.",
            file=sys.stderr,
        )
        return 1

    print(sd.query_devices())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
