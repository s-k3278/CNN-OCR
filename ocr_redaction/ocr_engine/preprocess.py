"""CPU-friendly image preprocessing shared by the inference components.

The engine runs an ordered list of steps (scale -> grayscale -> sharpen by
default). New steps are plain callables ``fn(image) -> image`` and can be added
without touching the engine::

    pre = Preprocessor(scale=1.5, sharpen=True)
    pre.add_step(lambda img: cv2.fastNlMeansDenoising(img))
    prepared = pre(image)

Every geometric step must be a pure resize so that detected boxes can be mapped
back to the original image with ``Preprocessor.scale`` (see engine.py).
"""
import cv2
import numpy as np


# ------------------------------------------------------------------ steps
def scale_image(image, scale=1.0):
    """Resize by a factor. INTER_AREA shrinks cleanly, INTER_CUBIC enlarges."""
    if scale == 1.0:
        return image
    if scale <= 0:
        raise ValueError("scale must be positive")
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=interp)


def to_grayscale(image):
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def sharpen_image(image, amount=0.25, sigma=1.0):
    """Unsharp mask: image + amount * (image - blurred)."""
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1.0 + amount, blur, -amount, 0)


class Preprocessor:
    """Ordered preprocessing pipeline (scaling, grayscale, sharpening, custom)."""

    def __init__(self, scale=1.0, grayscale=True, sharpen=False):
        self.scale = float(scale)
        self.grayscale = bool(grayscale)
        self.sharpen = bool(sharpen)
        self.extra_steps = []

    def add_step(self, fn):
        """Append a custom ``fn(image) -> image`` step (must not resize)."""
        self.extra_steps.append(fn)
        return self

    def __call__(self, image):
        out = scale_image(image, self.scale)
        if self.grayscale:
            out = to_grayscale(out)
        if self.sharpen:
            out = sharpen_image(out)
        for fn in self.extra_steps:
            out = fn(out)
        return out


def preprocess_image(image, grayscale=True, scale=1.0, sharpen=True):
    """Backward-compatible helper (kept for older callers)."""
    return Preprocessor(scale=scale, grayscale=grayscale, sharpen=sharpen)(image.copy())


# ------------------------------------------------------- recognizer input
def normalize_word_crop(gray, height=32, width=192):
    """Resize a word crop while preserving aspect ratio and pad with white.

    This must stay identical to the loader in training/dataset.py: both scale
    to ``height``, pad to ``width`` with 255 and map pixels to [-1, 1].
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Empty crop")

    scale = height / float(h)
    new_w = max(1, int(round(w * scale)))

    if new_w > width:
        scale = width / float(w)
        new_w = width
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.full((height, width), 255, dtype=np.uint8)
        y0 = max(0, (height - new_h) // 2)
        canvas[y0:y0 + new_h, :new_w] = resized
    else:
        resized = cv2.resize(gray, (new_w, height), interpolation=cv2.INTER_AREA)
        canvas = np.full((height, width), 255, dtype=np.uint8)
        canvas[:, :new_w] = resized

    x = canvas.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return x, max(4, new_w)
