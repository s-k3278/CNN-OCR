"""CRNN recognizer wrapper (CPU, batched)."""
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from training.model import CRNNOCR
from training.vocabulary import Vocabulary
from .preprocess import normalize_word_crop

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models" / "ocr_crnn.pt"


class Recognizer:
    """Loads a trained checkpoint and transcribes word crops.

    CPU optimisations: ``model.eval()`` + ``torch.inference_mode()``, a
    configurable thread count, fixed-size tensors that are run through the
    network in batches (one forward pass for many words), and optional dynamic
    int8 quantisation of the LSTM / Linear layers (``quantize=True``).
    """

    def __init__(self, model_path=None, threads=4, quantize=False, batch_size=32):
        model_path = Path(model_path) if model_path else DEFAULT_MODEL
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {model_path}. "
                "Train one first (see models/MODEL_REQUIRED.txt)."
            )
        torch.set_num_threads(max(1, min(int(threads), 32)))
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        self.height = int(ckpt["height"])
        self.width = int(ckpt["width"])
        self.vocab = Vocabulary(ckpt["chars"])
        self.batch_size = int(batch_size)
        self.quantized = False

        model = CRNNOCR(self.vocab.size, hidden=int(ckpt["hidden"]))
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        if quantize:
            try:
                model = torch.ao.quantization.quantize_dynamic(
                    model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
                )
                self.quantized = True
            except Exception as exc:  # fall back to the float model
                print(f"WARNING: dynamic quantization unavailable ({exc}); using float32.")
        self.model = model

    def recognize_batch(self, crops):
        """Transcribe a list of crops. Returns ``(texts, seconds_in_model)``."""
        texts = [""] * len(crops)
        if not crops:
            return texts, 0.0

        arrays, valid = [], []
        for crop in crops:
            if crop.ndim == 3:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            arr, valid_width = normalize_word_crop(crop, self.height, self.width)
            arrays.append(arr)
            valid.append(valid_width)

        elapsed = 0.0
        for start in range(0, len(arrays), self.batch_size):
            chunk = np.stack(arrays[start:start + self.batch_size])
            tensor = torch.from_numpy(chunk).unsqueeze(1)  # B,1,H,W
            t0 = time.perf_counter()
            with torch.inference_mode():
                ids = self.model(tensor).argmax(-1)  # T,B
            elapsed += time.perf_counter() - t0
            for b in range(chunk.shape[0]):
                steps = max(1, valid[start + b] // 4)  # CNN reduces width 4x
                texts[start + b] = self.vocab.decode_greedy(ids[:steps, b].tolist())
        return texts, elapsed

    def recognize(self, crop):
        """Single-crop convenience wrapper: ``(text, seconds)``."""
        texts, elapsed = self.recognize_batch([crop])
        return texts[0], elapsed
