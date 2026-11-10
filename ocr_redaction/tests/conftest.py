"""pytest setup: makes the tests work wherever the project folder lives.

* the project root is put on sys.path so `ocr_engine`, `cli`, `api`, `training` import
* this folder is put on sys.path so tests can `from synthetic_docs import ...`
This works with or without a tests/__init__.py file.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (HERE, HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
