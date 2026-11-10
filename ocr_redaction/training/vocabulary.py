"""Vocabulary utilities for IIIT-5K OCR.

IIIT-5K labels in the supplied CSVs are uppercase letters and digits.
Using only the observed label alphabet keeps the CPU model compact.
"""
from pathlib import Path
import json
import string

CHARSET = string.ascii_uppercase + string.digits


class Vocabulary:
    def __init__(self, chars=CHARSET):
        self.chars = chars
        self.blank_idx = 0
        self.char_to_idx = {ch: i + 1 for i, ch in enumerate(chars)}
        self.idx_to_char = {i + 1: ch for i, ch in enumerate(chars)}

    @property
    def size(self):
        return len(self.chars) + 1

    def encode(self, text):
        text = str(text).upper()
        unknown = [c for c in text if c not in self.char_to_idx]
        if unknown:
            raise ValueError(f"Unsupported characters in label {text!r}: {unknown}")
        return [self.char_to_idx[c] for c in text]

    def decode_greedy(self, indices):
        out, prev = [], None
        for idx in indices:
            idx = int(idx)
            if idx != self.blank_idx and idx != prev:
                out.append(self.idx_to_char.get(idx, ""))
            prev = idx
        return "".join(out)

    def save(self, path):
        Path(path).write_text(json.dumps({"chars": self.chars}, indent=2))

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        return cls(data["chars"])
