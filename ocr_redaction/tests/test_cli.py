import json

import cv2

from cli import main as cli_main
from ocr_engine import InferenceEngine

from synthetic_docs import make_page


class FakeRecognizer:
    def recognize_batch(self, crops):
        return ["TEXT"] * len(crops), 0.0


def test_cli_writes_both_outputs(tmp_path, monkeypatch, capsys):
    img, _ = make_page("light", 6)
    src = tmp_path / "invoice.png"
    cv2.imwrite(str(src), img)
    monkeypatch.setattr(
        cli_main, "build_engine", lambda args: InferenceEngine(recognizer=FakeRecognizer())
    )
    assert cli_main.main([str(src)]) == 0
    assert json.loads((tmp_path / "invoice_text.json").read_text())
    assert (tmp_path / "invoice_redacted.png").exists()
    assert "Detected words" in capsys.readouterr().out


def test_cli_reports_missing_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli_main, "build_engine", lambda args: InferenceEngine(recognizer=FakeRecognizer())
    )
    assert cli_main.main([str(tmp_path / "missing.png")]) == 2
    assert "error" in capsys.readouterr().err
