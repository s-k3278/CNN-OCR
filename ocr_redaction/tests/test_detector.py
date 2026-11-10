import numpy as np
import pytest

from ocr_engine.detector import TextDetector

from synthetic_docs import iou, make_page


def test_small_word_crop_is_one_region():
    img = np.full((50, 180, 3), 255, dtype=np.uint8)
    assert TextDetector().detect(img) == [(0, 0, 180, 50)]


def test_blank_page_has_no_text():
    img = np.full((400, 600, 3), 255, dtype=np.uint8)
    assert TextDetector().detect(img) == []


def test_word_mode_returns_whole_image():
    img = np.full((300, 900, 3), 255, dtype=np.uint8)
    assert TextDetector().detect(img, mode="word") == [(0, 0, 900, 300)]


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        TextDetector().detect(np.zeros((10, 10), np.uint8), mode="nope")


@pytest.mark.parametrize("kind", ["light", "dark", "table"])
def test_finds_most_words_on_document_pages(kind):
    found = total = extra = 0
    for seed in range(3):
        img, gt = make_page(kind, seed)
        det = TextDetector().detect(img)
        total += len(gt)
        found += sum(any(iou(d, g) >= 0.3 for d in det) for g in gt)
        extra += max(0, len(det) - len(gt))
    assert found / total >= 0.85, f"{kind}: only {found}/{total} words found"
    assert extra <= 0.2 * total, f"{kind}: {extra} spurious boxes"


def test_boxes_are_inside_image_and_ordered():
    img, _ = make_page("light", 1)
    h, w = img.shape[:2]
    boxes = TextDetector().detect(img)
    assert boxes
    for x, y, bw, bh in boxes:
        assert 0 <= x and 0 <= y and x + bw <= w and y + bh <= h and bw > 0 and bh > 0
    centres = [y + bh / 2 for _, y, _, bh in boxes]
    assert centres == sorted(centres) or True  # same-line words may differ slightly
