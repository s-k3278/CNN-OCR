import numpy as np

from ocr_engine.preprocess import Preprocessor, normalize_word_crop, preprocess_image


def test_normalize_preserves_shape_and_dtype():
    img = np.zeros((20, 80), dtype=np.uint8)
    x, valid = normalize_word_crop(img)
    assert x.shape == (32, 192)
    assert x.dtype == np.float32
    assert 4 <= valid <= 192


def test_normalize_accepts_color_crop():
    img = np.zeros((20, 80, 3), dtype=np.uint8)
    x, _ = normalize_word_crop(img)
    assert x.shape == (32, 192)


def test_pipeline_scale_grayscale_sharpen():
    img = np.random.RandomState(0).randint(0, 255, (40, 60, 3), dtype=np.uint8)
    out = Preprocessor(scale=2.0, grayscale=True, sharpen=True)(img)
    assert out.shape == (80, 120)  # scaled and gray
    keep = Preprocessor(scale=1.0, grayscale=False, sharpen=False)(img)
    assert keep.shape == img.shape and (keep == img).all()


def test_custom_step_is_applied():
    img = np.full((10, 10), 100, np.uint8)
    pre = Preprocessor(grayscale=False).add_step(lambda im: im + 1)
    assert int(pre(img)[0, 0]) == 101


def test_legacy_helper_still_works():
    img = np.zeros((20, 30, 3), np.uint8)
    assert preprocess_image(img, grayscale=True, scale=1.0, sharpen=True).ndim == 2
