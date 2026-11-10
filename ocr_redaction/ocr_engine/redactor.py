"""OpenCV inpainting-based text redaction."""
import cv2
import numpy as np


def build_mask(shape, boxes, pad=1, dilation=1):
    """Binary mask covering every ``(x, y, w, h)`` box plus a small margin.

    The detector already pads its boxes, so the extra margin is kept small: a
    large margin makes inpainting smear neighbouring colours (e.g. the edge of
    a coloured header band) into the surrounding page.
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for x, y, w, h in boxes:
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(mask.shape[1], x + w + pad), min(mask.shape[0], y + h + pad)
        cv2.rectangle(mask, (x1, y1), (x2 - 1, y2 - 1), 255, -1)
    if dilation:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * dilation + 1, 2 * dilation + 1)
        )
        mask = cv2.dilate(mask, kernel)
    return mask


def redact(image, boxes, pad=1, dilation=1, radius=3):
    """Remove the text inside ``boxes`` by inpainting (Telea, CPU only).

    ``boxes`` are ``(x, y, w, h)`` in the coordinates of ``image``. Returns a
    new BGR image; the input is not modified.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    boxes = [b for b in boxes if b[2] > 0 and b[3] > 0]
    if not boxes:
        return image.copy()
    mask = build_mask(image.shape, boxes, pad, dilation)
    # Telea is fast and well suited to CPU document images.
    return cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)
