# All commands: from training to testing on a PNG

Everything below is for **Windows PowerShell** with Python 3.14 (`py -3.14`).
Run every command from the project folder, the one that contains `ocr_engine`, `training`, `cli`, `api`:

```powershell
cd "path to your redaction project"
```

Set the dataset path ONCE. Edit it to the real folder that contains `traindata.csv`,
`testdata.csv` and `IIIT5K-Word_V3.0` (use the real path, not the placeholder):

```powershell
$DATA = "C:\path\to\IIIT-5K"
```

---

## 1. Install the packages (one time)

```powershell
py -3.14 -m pip install -r requirements.txt
```

## 2. Run the tests (about 28 should pass)

```powershell
py -3.14 -m pytest -q
```

If you get strange import errors, delete the stale caches first:

```powershell
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
Remove-Item .pytest_cache -Recurse -Force -ErrorAction SilentlyContinue
```

## 3. Train the model

The final model was trained with 80 maximum epochs. Early stopping (patience 10) ended it at epoch 73,
and the best checkpoint (epoch 63) is the one that is saved. It took about 46 minutes on CPU.

```powershell
py -3.14 -m training.train --data-root $DATA --epochs 80 --threads 8 --output models\ocr_crnn_v3.pt
```

Output files (written next to `--output`):
`ocr_crnn.pt`, `ocr_crnn.vocab.json`, `ocr_crnn.history.json`

Quick smoke test of the training code (about 1 minute, do not keep this model):

```powershell
py -3.14 -m training.train --data-root $DATA --epochs 1 --threads 8 --output models\smoke_test.pt
```

## 4. Evaluate on the 3,000 test images

```powershell
py -3.14 -m training.evaluate --data-root $DATA --model models\ocr_crnn.pt --output outputs\evaluation.json --lexicon medium
```

It prints accuracy, precision, recall, average edit distance and ms per image, and writes
`outputs\evaluation.json`. Expected for the final model: about 49.6 % accuracy, precision 96.81 %,
recall 91.63 %, edit distance 1.23, about 4.7 ms per image.

## 5. Make it the default model used by the CLI and the API

```powershell
Copy-Item models\ocr_crnn_v3.pt           models\ocr_crnn.pt           -Force
Copy-Item models\ocr_crnn_v3.vocab.json   models\ocr_crnn.vocab.json   -Force
Copy-Item models\ocr_crnn_v3.history.json models\ocr_crnn.history.json -Force
```

## 6. Run OCR + redaction on a PNG (command line)

```powershell
py -3.14 textrecognizer.py examples\demo_document.png
```

This creates, next to the image:
- `demo_document_text.json`: list of `{"word", "bounding_box": [top, left, bottom, right]}`
- `demo_document_redacted.png`: the image with all detected text removed

On your own image, and with a separate output folder:

```powershell
py -3.14 textrecognizer.py "C:\path\to\my_image.png" --output-dir results
```

Useful options:

```powershell
# small or low-resolution text: enlarge the image first (often helps detection)
py -3.14 textrecognizer.py my_image.png --scale 2

# the whole image is ONE word (a cropped word image): skip detection
py -3.14 textrecognizer.py my_word.png --mode word

# force document mode (several words / lines)
py -3.14 textrecognizer.py my_page.png --mode document

# unsharp mask, and faster CPU inference (int8; test it before relying on it)
py -3.14 textrecognizer.py my_page.png --sharpen
py -3.14 textrecognizer.py my_page.png --quantize

# use another model file / more threads
py -3.14 textrecognizer.py my_page.png --model models\ocr_crnn_v3.pt --threads 8
```

(If you installed the launcher on PATH, `textrecognizer image.png` works the same way.)

## 7. Run the web service (FastAPI)

Start the server (leave this window open):

```powershell
py -3.14 -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In a SECOND PowerShell window:

```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -F "file=@examples\demo_document.png" http://127.0.0.1:8000/ocr
```

The reply contains `job_id`, the words, and two URLs. Download the results (replace `<job_id>`):

```powershell
curl.exe http://127.0.0.1:8000/ocr/<job_id>/text     -o image_text.json
curl.exe http://127.0.0.1:8000/ocr/<job_id>/redacted -o image_redacted.png
```

Interactive docs in the browser: http://127.0.0.1:8000/docs

Optional settings (environment variables, set before starting the server):

```powershell
$env:OCR_THREADS = "8"
$env:OCR_QUANTIZE = "1"
$env:OCR_MODEL_PATH = "models\ocr_crnn.pt"
```


```powershell
Expand-Archive ocr_redaction_submission.zip -DestinationPath $env:TEMP\check_submit -Force
cd $env:TEMP\check_submit\ocr_redaction
py -3.14 -m pytest -q
py -3.14 textrecognizer.py examples\demo_document.png
```
