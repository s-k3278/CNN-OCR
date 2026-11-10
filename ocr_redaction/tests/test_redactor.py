import numpy as np

from ocr_engine.detector import TextDetector
from ocr_engine.redactor import redact

from synthetic_docs import make_page


def test_redaction_removes_text_but_keeps_input():
    img, gt = make_page("light", 2)
    before = img.copy()
    boxes = TextDetector().detect(img)
    out = redact(img, boxes)
    assert (img == before).all()  # input untouched
    assert out.shape == img.shape
    gray = out[..., 0].astype(int)
    assert gray.min() >= 200, "dark text pixels survived redaction"


def test_dark_page_redaction():
    img, _ = make_page("dark", 3)
    out = redact(img, TextDetector().detect(img))
    assert out.max() <= 80, "light text pixels survived redaction"


def test_no_boxes_returns_copy():
    img = np.full((20, 20, 3), 7, np.uint8)
    out = redact(img, [])
    assert (out == img).all() and out is not img
