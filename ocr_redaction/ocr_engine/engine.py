"""Reusable OCR + redaction inference engine.

This module is the only place where the pipeline is wired together. The CLI
(``cli/main.py``) and the web service (``api/main.py``) both call
``InferenceEngine`` and contain no OCR logic of their own.

    image -> Preprocessor -> TextDetector -> Recognizer (batched) -> JSON
                                  \\-> redactor (inpainting) -> PNG

Bounding boxes in the JSON / redaction always refer to the ORIGINAL image,
even when ``scale`` resizes the image internally.
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .detector import TextDetector
from .preprocess import Preprocessor
from .redactor import redact


def load_image_bgr(path):
    """Read an image as 8-bit BGR. Handles unicode paths, alpha and 16-bit."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.fromfile(str(path), dtype=np.uint8)  # works with non-ASCII paths
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED) if data.size else None
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return to_bgr8(image)


def to_bgr8(image):
    if image.dtype != np.uint8:
        scale = 255.0 / 65535.0 if image.dtype == np.uint16 else 1.0
        image = np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:  # composite transparency onto white
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        rgb = image[:, :, :3].astype(np.float32)
        return (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return image


def save_png(path, image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode PNG")
    encoded.tofile(str(path))


class InferenceEngine:
    """OCR + redaction. Construct once, call ``process`` / ``process_array``."""

    def __init__(
        self,
        model_path=None,
        threads=4,
        scale=1.0,
        grayscale=True,
        sharpen=False,
        mode="auto",
        quantize=False,
        detector=None,
        recognizer=None,
    ):
        """
        model_path  CRNN checkpoint (default: <project>/models/ocr_crnn.pt)
        threads     CPU threads used by PyTorch
        scale       resize factor applied before detection / recognition
        grayscale   convert to grayscale before detection / recognition
        sharpen     apply an unsharp mask (off by default: the model was
                    trained on un-sharpened images)
        mode        'auto' | 'word' | 'document' (see detector.py)
        quantize    dynamic int8 quantisation of the recognizer (CPU speed-up)
        detector / recognizer   inject custom components (tests, other models)
        """
        self.defaults = dict(scale=scale, grayscale=grayscale, sharpen=sharpen, mode=mode)
        self.detector = detector or TextDetector(mode=mode)
        if recognizer is None:
            from .recognizer import Recognizer  # imported lazily (needs PyTorch)

            recognizer = Recognizer(model_path, threads=threads, quantize=quantize)
        self.recognizer = recognizer

    # ------------------------------------------------------------------
    def process_array(self, image, scale=None, grayscale=None, sharpen=None, mode=None):
        """Run the full pipeline on a BGR/gray numpy image (no files written)."""
        opts = {
            "scale": self.defaults["scale"] if scale is None else scale,
            "grayscale": self.defaults["grayscale"] if grayscale is None else grayscale,
            "sharpen": self.defaults["sharpen"] if sharpen is None else sharpen,
            "mode": self.defaults["mode"] if mode is None else mode,
        }
        original = to_bgr8(image)
        height, width = original.shape[:2]

        t0 = time.perf_counter()
        pre = Preprocessor(opts["scale"], opts["grayscale"], opts["sharpen"])
        prepared = pre(original)
        t_pre = time.perf_counter() - t0

        t0 = time.perf_counter()
        boxes_prepared = self.detector.detect(prepared, mode=opts["mode"])
        t_det = time.perf_counter() - t0

        crops = [prepared[y:y + h, x:x + w] for x, y, w, h in boxes_prepared]
        t0 = time.perf_counter()
        texts, t_model = self.recognizer.recognize_batch(crops)
        t_rec = time.perf_counter() - t0

        # Map boxes back to the original image (undo `scale`).
        s = pre.scale
        boxes = []
        for x, y, w, h in boxes_prepared:
            x1 = int(np.clip(np.floor(x / s), 0, width))
            y1 = int(np.clip(np.floor(y / s), 0, height))
            x2 = int(np.clip(np.ceil((x + w) / s), 0, width))
            y2 = int(np.clip(np.ceil((y + h) / s), 0, height))
            boxes.append((x1, y1, max(0, x2 - x1), max(0, y2 - y1)))

        words = []
        for (x, y, w, h), text in zip(boxes, texts):
            if text.strip() and w > 0 and h > 0:
                # [top, left, bottom, right]
                words.append({"word": text, "bounding_box": [y, x, y + h, x + w]})

        t0 = time.perf_counter()
        # Redact EVERY detected region, even those the recogniser could not read.
        redacted = redact(original, [b for b in boxes if b[2] > 0 and b[3] > 0])
        t_red = time.perf_counter() - t0

        return {
            "words": words,
            "redacted": redacted,
            "image_size": {"width": int(width), "height": int(height)},
            "ocr_inference_seconds": t_model,
            "timings": {
                "preprocess_s": t_pre,
                "detect_s": t_det,
                "recognize_s": t_rec,
                "redact_s": t_red,
            },
        }

    def process(self, image_path, output_dir=None, sharpen=None, scale=None,
                grayscale=None, mode=None):
        """Read ``image_path``; write ``<stem>_text.json`` and ``<stem>_redacted.png``."""
        image_path = Path(image_path)
        image = load_image_bgr(image_path)
        result = self.process_array(
            image, scale=scale, grayscale=grayscale, sharpen=sharpen, mode=mode
        )

        output_dir = Path(output_dir) if output_dir else image_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{image_path.stem}_text.json"
        redacted_path = output_dir / f"{image_path.stem}_redacted.png"

        json_path.write_text(json.dumps(result["words"], indent=2), encoding="utf-8")
        save_png(redacted_path, result.pop("redacted"))

        result["json_path"] = str(json_path)
        result["redacted_path"] = str(redacted_path)
        return result
