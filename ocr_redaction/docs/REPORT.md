# OCR & Text Redaction Utility - Project Report

## 1. Objective

Build a modular, CPU-only system that recognises text in images and produces a redacted image, with the
processing logic (`ocr_engine`) strictly separated from the command-line and web interfaces.

## 2. Dataset

IIIT-5K Words: 2,000 training and 3,000 test word images with ground-truth labels (plus per-image small /
medium lexicons). Labels contain only `A-Z` and `0-9` after upper-casing, so the recogniser has 36 classes plus
the CTC blank. The 2,000 training images are split 1,800 / 200 (train / validation, seed 42); the 3,000
test images are used only for the final evaluation.

## 3. Architecture

```text
                      INPUT IMAGE
                           |
                +----------v-----------+
                |  Preprocessor        |  scale -> grayscale -> sharpen -> (custom steps)
                +----------+-----------+
                           |
                +----------v-----------+
                |  TextDetector        |  edge mask -> remove table rules -> glyphs
                |  (OpenCV, no model)  |  -> text lines -> words -> word boxes
                +----------+-----------+
                           |  word crops (batched)
                +----------v-----------+
                |  Recognizer (CRNN)   |  32x192 crop -> CNN -> BiLSTM -> CTC greedy decode
                +----------+-----------+
                           |
              +------------+-------------+
              |                          |
     word + bounding box           redactor (Telea inpainting)
              |                          |
     <name>_text.json            <name>_redacted.png

   CLI (cli/main.py) ----\
                          >---- ocr_engine.InferenceEngine
   FastAPI (api/main.py) -/
```

The CLI and the FastAPI service contain no OCR code: they parse arguments / HTTP requests and call
`ocr_engine.InferenceEngine`. Boxes always refer to the original image, even when `--scale` resizes it, and
every detected region is redacted even if the recogniser could not read it.

### Recognizer (`training/model.py`)

| Stage | Detail |
|---|---|
| Input | grayscale word crop, aspect ratio preserved, 32 px high, padded with white to 192 px, pixels mapped to [-1, 1] |
| CNN | 4 x (Conv 3x3 + BatchNorm + ReLU), channels 32-64-128-256, pooling reduces height to 1 and width by 4 -> up to 48 time steps |
| Sequence | 2-layer bidirectional LSTM, hidden size 128 |
| Output | linear layer to 37 classes, CTC loss, greedy CTC decoding |
| Size | about 1.19 million parameters, 4.8 MB checkpoint |

The CTC input length of each training word is the width actually occupied by the word (padding is ignored),
and inference decodes only the corresponding time steps.

### Detector (`ocr_engine/detector.py`)

IIIT-5K only contains cropped words, so detection is a separate classical OpenCV component:

1. **Edge mask** - morphological gradient with an Otsu threshold. An edge map has no text colour, so
   dark-on-light, light-on-dark, coloured bands and uneven lighting are all handled the same way.
2. **Rule removal** - long thin horizontal / vertical runs (table lines, underlines, borders) are removed
   so they cannot glue text to the grid.
3. **Glyphs** - connected components of what remains.
4. **Lines and words** - glyphs are linked into text lines; every line is split into words using a gap
   threshold derived from that line's own gap pattern (scaled by glyph height), with a width-based fallback
   for lines whose letters are fused. Tiny marks (i-dots) are absorbed into their word.
5. **Faint text** - a second, lower threshold is used only to adopt faint letters directly next to a
   detected word (so low-contrast text on a textured background is not left behind) without letting noise
   create new detections.
6. **Clean-up** - noise, frames and surviving rules are dropped, overlapping boxes merged, boxes padded and
   put in reading order.

Small images (<= 160 x 600 px) that look like one word are returned as a single box (`--mode` can force
`word` or `document`).

### Redaction (`ocr_engine/redactor.py`)

All detected boxes (padded by the detector, plus a 1 px margin) form a mask that is filled with OpenCV
Telea inpainting. The margin is deliberately small: a large one smears neighbouring colours (for example the
edge of a header band) across the page.

## 4. Training

| Parameter | Value |
|---|---|
| Python / device | 3.14.6, CPU only (8 threads) |
| Epochs | 80 maximum; early stopping ended training after epoch 73 |
| Batch size | 64 |
| Optimiser | AdamW, weight decay 0.01 |
| Learning rate | 1e-3, 3-epoch linear warm-up, cosine decay to 1e-5 |
| Loss | CTC (blank = 0, `zero_infinity`) |
| Augmentation (train only) | rotation +-4 deg, shear, scale, perspective, contrast / brightness, gamma, blur, low-resolution round-trip, Gaussian noise |
| Dropout (train only) | 0.10 on CNN features, 0.25 between LSTM layers, 0.30 before the classifier |
| Model selection | best validation edit distance (ties broken by validation accuracy) |
| Early stopping | patience 10 (best checkpoint at epoch 63, no improvement for the next 10 epochs) |
| Time | about 38 s per epoch on average, 46 minutes in total |

Selected checkpoint (epoch 63): validation accuracy 69.5 %, character precision 97.4 %, recall 97.1 %, edit distance
0.52 (200 images - a small, noisy sample; the test accuracy in section 5 is the figure to rely on). The per-epoch
history is in `models/ocr_crnn.history.json`. Validation accuracy is 0 for the first 9 epochs (the usual CTC "all blanks"
plateau), then rises steadily until about epoch 60 and flattens at 66-70 % as the learning rate decays.

## 5. Results (3,000 IIIT-5K test images)

Produced by `python -m training.evaluate ... --lexicon medium` (4 CPU threads).

| Metric | Raw CRNN | Medium-lexicon assisted |
|---|---:|---:|
| Exact word accuracy | **49.6 %** (1,489 / 3,000) | 79.2 % (2,375 / 3,000) |
| Character precision | 96.81 % | 97.80 % |
| Character recall | 91.63 % | 91.89 % |
| Average edit distance | **1.230** | 0.841 |
| Recognition time | **4.74 ms / image** | n/a |

The lexicon column uses the candidate word list supplied with each IIIT-5K image; it is not available for
arbitrary documents, so the raw column is the honest measure of the model.

### Progression

| Version | Accuracy | Precision | Recall | Edit distance | ms / image |
|---|---:|---:|---:|---:|---:|
| Baseline (first model) | 11.2 % | 94.9 % | 72.0 % | 2.77 | 2.42 |
| V5 (CTC input-length fix, A-Z/0-9 vocabulary) | 15.0 % | 92.1 % | 76.4 % | 2.61 | 3.91 |
| V5 retrained, longer, early-stopped at epoch 56 | 27.5 % | 93.9 % | 84.8 % | 1.95 | 4.56 |
| Strong augmentation, dropout, warm-up, weight decay - 60 epochs | 42.4 % | 95.9 % | 90.1 % | 1.46 | 4.49 |
| **Final** - same recipe, up to 80 epochs (stopped at 73) | **49.6 %** | **96.8 %** | **91.6 %** | **1.23** | 4.74 |

Training and inference preprocessing are now identical (no contrast stretching during training, no sharpening
at inference by default). Earlier versions normalised contrast during training only.

### Detector and redaction on synthetic pages

No annotated financial documents were available, so the detector was measured on generated pages with known
word positions (10 styles x 3 seeds: plain, table with rules, coloured header bands, dark background, uneven
lighting, added noise and JPEG compression; 2,144 words). These pages are synthetic and easier than real
scans.

| | Old detector | New detector |
|---|---:|---:|
| Words found (IoU >= 0.5) | 20.4 % | **95.8 %** |
| Boxes that are real words | 45.0 % | **97.8 %** |
| Real text pixels inside a box | 96.7 % | **99.94 %** |

* Redaction: the average grey-level difference between the text area and a clean page falls from about
  70-90 to 0.2-9 (out of 255) after redaction; on the five-word demo page no dark pixel remains.
* Weak spot: text with very wide letter spacing across several words per line - new detector 83 % words
  found / 56 % precision (old: 90 % / 84 %). Coverage for redaction stays 99.98 %.
* Detection time about 17 ms for a 1000 x 700 page.

## 6. CPU optimisation

Batched recognition (one forward pass per page), `inference_mode`, configurable thread count, fixed-size
tensors, optional dynamic int8 quantisation, lazy one-time model loading, OpenCV/NumPy-only detection and
redaction.

## 7. Interfaces

* **CLI** - `textrecognizer image.png` -> `image_text.json`, `image_redacted.png`. JSON format
  `[{"word": "...", "bounding_box": [top, left, bottom, right]}]`.
* **FastAPI** - `GET /health`, `POST /ocr`, `GET /ocr/{id}/redacted`, `GET /ocr/{id}/text`; upload size
  limit, automatic clean-up of results, non-blocking inference.

## 8. Limitations

* The model was trained on 1,800 word crops: uppercase letters and digits only (no lowercase or punctuation),
  English only, no handwriting; it has not been evaluated on real invoices or statements.
* The detector cannot handle rotated or curved text, handwriting or heavily textured photographs. Whatever it
  misses is not redacted.
* The validation set (200 images) is small, so model selection is noisy.

## 9. Future improvements

* More training data: synthetic word images and document-style word crops, covering lowercase,
  punctuation and currency symbols, with a larger character set.
* A learned text detector (DBNet / CRAFT) for rotated and low-contrast text.
* CTC beam search with a language model or domain lexicon; longer training schedule.
* Layout analysis for financial documents (tables, key-value fields).
