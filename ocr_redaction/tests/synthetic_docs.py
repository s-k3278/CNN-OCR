"""Tiny synthetic 'document' generator for tests (OpenCV only, no fonts needed).

Every page comes with the ground-truth word boxes ``(x, y, w, h)``.
"""
import random

import cv2
import numpy as np

WORDS = ("INVOICE TOTAL AMOUNT DATE ACCOUNT NUMBER BALANCE Payment Customer "
         "Address Tax Subtotal Quantity 1,250.00 4521-8890 USD 2026 Support").split()
FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX]


def make_page(kind="light", seed=0, size=(900, 600), lines=9):
    """kind: 'light' (dark text on white), 'dark' (light text on dark), 'table'."""
    rng = random.Random(seed)
    w, h = size
    bg, fg = ((235, 235, 230), (25, 25, 30)) if kind != "dark" else ((30, 34, 40), (235, 235, 230))
    img = np.full((h, w, 3), 255 if kind != "dark" else bg[0], np.uint8)
    if kind == "dark":
        img[:] = bg
    boxes = []
    y = 55
    if kind == "table":
        for r in range(lines + 1):
            cv2.line(img, (30, 30 + r * 52), (w - 30, 30 + r * 52), (40, 40, 40), 2)
        for c in (30, 320, 600, w - 30):
            cv2.line(img, (c, 30), (c, 30 + lines * 52), (40, 40, 40), 2)
    for line in range(lines):
        font = rng.choice(FONTS)
        scale = rng.choice([0.6, 0.7, 0.8, 1.0])
        thick = 1 if scale < 0.9 else 2
        x = 50 if kind != "table" else 42 + rng.choice([0, 290, 570]) * (kind == "table")
        for _ in range(rng.choice([2, 3, 4]) if kind != "table" else 1):
            word = rng.choice(WORDS)
            (tw, th), base = cv2.getTextSize(word, font, scale, thick)
            if x + tw > (w - 50 if kind != "table" else x + 240):
                break
            cv2.putText(img, word, (x, y + th), font, scale, fg, thick, cv2.LINE_AA)
            boxes.append((x, y, tw, th + base))
            (sw, _), _ = cv2.getTextSize(" ", font, scale, thick)
            x += tw + int(sw * 1.4) + 6
        y += int(50 if kind == "table" else 56)
        if y > h - 50:
            break
    return img, boxes


def iou(a, b):
    iw = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    ih = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / float(a[2] * a[3] + b[2] * b[3] - inter)
