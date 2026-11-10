"""End-to-end tests of the engine with a stand-in recogniser (no model file needed)."""
import json

import cv2
import numpy as np
import pytest

from ocr_engine import InferenceEngine
from ocr_engine.detector import TextDetector

from synthetic_docs import make_page


class FakeRecognizer:
    def recognize_batch(self, crops):
        return [f"WORD{i}" for i, _ in enumerate(crops)], 0.001


@pytest.fixture()
def engine():
    return InferenceEngine(recognizer=FakeRecognizer())


@pytest.fixture()
def page(tmp_path):
    img, _ = make_page("light", 4)
    path = tmp_path / "image.png"
    cv2.imwrite(str(path), img)
    return path, img


def test_outputs_are_named_from_input_stem(engine, page, tmp_path):
    path, _ = page
    out = tmp_path / "out"
    result = engine.process(path, out)
    assert (out / "image_text.json").exists()
    assert (out / "image_redacted.png").exists()
    assert result["json_path"].endswith("image_text.json")


def test_json_schema_and_box_order(engine, page, tmp_path):
    path, img = page
    engine.process(path, tmp_path)
    words = json.loads((tmp_path / "image_text.json").read_text())
    h, w = img.shape[:2]
    assert words
    for item in words:
        assert set(item) == {"word", "bounding_box"}
        assert isinstance(item["word"], str)
        top, left, bottom, right = item["bounding_box"]
        assert all(isinstance(v, int) for v in item["bounding_box"])
        assert 0 <= top < bottom <= h and 0 <= left < right <= w


def test_redacted_image_contains_no_text(engine, page, tmp_path):
    path, img = page
    engine.process(path, tmp_path)
    red = cv2.imread(str(tmp_path / "image_redacted.png"))
    assert red.shape == img.shape
    assert red.min() >= 200
    assert TextDetector().detect(red, mode="document") == []


def test_scale_keeps_boxes_in_original_coordinates(engine, page):
    path, img = page
    base = engine.process_array(img, scale=1.0)["words"]
    big = engine.process_array(img, scale=2.0)["words"]
    h, w = img.shape[:2]
    assert base and big
    for item in big:  # boxes must still fit the ORIGINAL image
        top, left, bottom, right = item["bounding_box"]
        assert 0 <= top < bottom <= h and 0 <= left < right <= w


def test_word_crop_is_a_single_word(engine, tmp_path):
    crop = np.full((40, 160, 3), 255, np.uint8)
    cv2.putText(crop, "HOTEL", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    p = tmp_path / "word.png"
    cv2.imwrite(str(p), crop)
    words = engine.process(p, tmp_path)["words"]
    assert len(words) == 1 and words[0]["bounding_box"] == [0, 0, 40, 160]


def test_missing_and_corrupt_files(engine, tmp_path):
    with pytest.raises(FileNotFoundError):
        engine.process(tmp_path / "nope.png")
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError):
        engine.process(bad)


def test_unicode_path_and_transparent_png(engine, tmp_path):
    img, _ = make_page("light", 5)
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    p = tmp_path / "bild_ä.png"
    ok, enc = cv2.imencode(".png", rgba)
    enc.tofile(str(p))
    assert engine.process(p, tmp_path)["words"]
