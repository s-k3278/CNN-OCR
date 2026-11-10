#!/usr/bin/env python3
"""Launcher so the tool can be run as `textrecognizer image.png`.

    python textrecognizer.py image.png        (any OS)
    textrecognizer image.png                  (Windows: textrecognizer.cmd)
    ./textrecognizer image.png                (Linux / macOS: ./textrecognizer)

Works from any folder; the model path is resolved relative to this project.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cli.main import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
