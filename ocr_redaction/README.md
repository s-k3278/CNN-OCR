# CPU OCR & Text Redaction Utility

End-to-end OCR and text-redaction system trained on the **IIIT-5K Words** dataset.
A compact **CRNN (CNN + BiLSTM + CTC)** model reads words, an OpenCV detector finds
them in full images, and OpenCV inpainting removes them. **CPU only - no GPU needed.**

| Assignment requirement | Where |
|---|---|
| Train + evaluate (accuracy, precision, recall, edit distance) | `training/train.py`, `training/evaluate.py`, `training/metrics.py` |
| CPU-specific inference optimisation | batched inference, `inference_mode`, thread control, optional int8 quantisation (see below) |
| Inference engine library (scaling, grayscale, sharpening, OCR, inpainting) | `ocr_engine/` |
| CLI -> `<name>_text.json` + `<name>_redacted.png` | `textrecognizer` / `cli/main.py` |
| FastAPI web service | `api/main.py` |
| Engine separate from CLI and API | both import `ocr_engine.InferenceEngine`, nothing else |
| Documentation + `requirements.txt` | this file, `docs/REPORT.md`, `requirements.txt` |

## Architecture

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

* **Boxes always refer to the original image**, even when `--scale` resizes it internally.
* **Every detected region is redacted**, even if the recogniser could not read it, so the
  redacted image does not depend on recognition accuracy.
* The CLI and the API contain no OCR code - they parse arguments / HTTP and call the engine.

### Project layout

```text
ocr_engine/   preprocess.py  detector.py  recognizer.py  redactor.py  engine.py   <- reusable library
cli/          main.py                     command line wrapper
api/          main.py                     FastAPI wrapper
training/     dataset.py model.py train.py evaluate.py metrics.py vocabulary.py inspect_predictions.py
models/       ocr_crnn.pt  ocr_crnn.vocab.json  ocr_crnn.history.json   <- trained model (epoch 63)
outputs/      evaluation.json (final test metrics)  baseline_evaluation.json (first model, for comparison)
examples/     demo_document.png (+ its JSON / redacted PNG), create_demo.py
tests/        unit + end-to-end tests (no dataset and no trained model needed)
docs/         REPORT.md                   full project report
data/         README.md                   where to put the IIIT-5K dataset (not included)
run_pipeline.py                           train + evaluate in one command
COMMANDS.md                               every command, from training to testing on a PNG
textrecognizer(.py/.cmd)                  launcher: `textrecognizer image.png`
```

## Setup

Python 3.14 (Windows) was used; Python 3.10+ should work.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\activate            # if scripts are blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build of PyTorch
pip install -r requirements.txt
```

(A virtual environment is optional. Without one, run `py -3.14 -m pip install -r requirements.txt`
and use `py -3.14 -m ...` instead of `python -m ...`.)

The trained model `models/ocr_crnn.pt` is included, so inference works immediately.

## Command line

```powershell
textrecognizer image.png                 # Windows (textrecognizer.cmd);  ./textrecognizer on Linux/macOS
py -3.14 textrecognizer.py image.png     # same thing, any OS
py -3.14 -m cli.main image.png           # same thing, as a module
```

Creates `image_text.json` and `image_redacted.png` next to the input (or in `--output-dir`).

```json
[{"word": "TOTAL", "bounding_box": [120, 45, 151, 190]}]
```

`bounding_box` is `[top, left, bottom, right]` in pixels of the original image.

| Option | Meaning |
|---|---|
| `--scale 1.5` | resize before OCR (helps small text); boxes are mapped back |
| `--no-grayscale` | skip grayscale conversion |
| `--sharpen` | unsharp mask (off by default - the model was trained without it) |
| `--mode auto/word/document` | `auto`: small crops are one word, larger images are searched for words; `word`: the whole image is one word; `document`: always detect words |
| `--quantize` | dynamic int8 quantisation of the recogniser |
| `--threads N` | CPU threads for PyTorch (default 4) |
| `--model PATH`, `--output-dir DIR` | model checkpoint / output folder |

## Web service

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

* `GET /health`
* `POST /ocr` - multipart field `file` (PNG/JPG/BMP/WEBP/TIFF, max 20 MB); optional query
  parameters `scale`, `grayscale`, `sharpen`, `mode`. Returns the words, image size, timings and
  URLs of the results.
* `GET /ocr/{job_id}/redacted` - the redacted PNG
* `GET /ocr/{job_id}/text` - the JSON word list

```powershell
curl -F "file=@image.png" http://127.0.0.1:8000/ocr
```

Results are deleted after 15 minutes, uploads are deleted right after processing, and the OCR runs in
a worker thread so the server stays responsive. Interactive docs: `http://127.0.0.1:8000/docs`.

## Library use

```python
from ocr_engine import InferenceEngine

engine = InferenceEngine(scale=1.0, sharpen=False, mode="auto", threads=4)
result = engine.process("invoice.png", output_dir="out")     # writes the two files
result["words"]                                               # [{"word", "bounding_box"}, ...]

# or without touching the disk:
result = engine.process_array(bgr_image)                      # result["redacted"] is a numpy image
```

New preprocessing steps are plain functions: `Preprocessor(...).add_step(fn)`.

## Training and evaluation

Dataset folder (`--data-root`) must contain `traindata.csv`/`testdata.csv` and the
`IIIT5K-Word_V3.0/IIIT5K/{train,test}` images (the loader finds both layouts automatically).

```powershell
python -m training.train --data-root "<path>\IIIT-5K" --epochs 80 --threads 8 --output models\ocr_crnn.pt
python -m training.evaluate --data-root "<path>\IIIT-5K" --model models\ocr_crnn.pt `
        --output outputs\evaluation.json --lexicon medium
```

Training uses a 10 % validation split of the training images, strong on-the-fly augmentation
(rotation, shear, perspective, contrast/gamma, blur, noise), dropout, a 3-epoch learning-rate
warm-up with cosine decay, AdamW with weight decay and early stopping; the checkpoint with the best
validation edit distance is kept. See `docs/REPORT.md` for settings and results.

### Final results (IIIT-5K test split, 3,000 images, CPU, 4 threads)

| Metric | Raw CRNN | With medium lexicon* |
|---|---:|---:|
| Exact word accuracy | **49.6 %** | 79.2 % |
| Character precision | 96.8 % | 97.8 % |
| Character recall | 91.6 % | 91.9 % |
| Average edit distance | **1.23** | 0.84 |
| Recognition time | **4.7 ms / word crop** | - |

\*Snaps the prediction to the nearest word of the per-image lexicon supplied with IIIT-5K. It is
reported separately and is *not* available for arbitrary documents - the raw column is the real model.

## CPU optimisation

* compact model (~1.2 M parameters) and fixed 32 x 192 inputs
* all words of an image are recognised in **one batched forward pass**
* `model.eval()` + `torch.inference_mode()`, configurable thread count
* optional dynamic int8 quantisation (`--quantize`; compare accuracy before using it)
* detector and redactor are pure OpenCV/NumPy (about 17 ms to detect a 1000 x 700 page)
* the model is loaded once and reused (API loads it lazily on the first request)

## Tests

```powershell
python -m pytest -q
```

Covers preprocessing, the detector on synthetic pages (light, dark, table), redaction, the full pipeline
(JSON schema, file names, box order, scaling, unicode paths, transparent PNGs), the CLI and the API
(the API tests are skipped if `httpx` is not installed). They use a stand-in recogniser, so neither the
dataset nor the trained model is needed.

## Known limitations

* The recogniser was trained only on **cropped English words** (1,800 images) and reads
  **uppercase letters and digits only** - lowercase is read as uppercase, and punctuation such as `,` `.` `-`
  cannot be produced. It has not been evaluated on real financial documents or handwriting.
* The detector is classical OpenCV. It handles dark-on-light, light-on-dark, coloured bands, uneven
  lighting and table rules, but not rotated or curved text, heavy textures or handwriting. Very widely
  letter-spaced text can be split into several boxes (redaction is unaffected, recognition suffers).
* Text the detector misses is not redacted. A learned detector (DBNet/CRAFT) would be the next step.
* Small images (<= 160 x 600) that look like a single word are treated as one word (`--mode word|document`
  overrides this).
