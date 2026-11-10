"""FastAPI wrapper around the reusable InferenceEngine.

    uvicorn api.main:app --host 0.0.0.0 --port 8000

The service contains no OCR logic: it validates the upload, calls the engine and
serves the results. Results are kept in a temporary folder for ``JOB_TTL_SECONDS``
and then deleted automatically.

Environment variables (all optional):
    OCR_MODEL_PATH   path to the checkpoint (default: <project>/models/ocr_crnn.pt)
    OCR_THREADS      CPU threads for PyTorch (default 4)
    OCR_QUANTIZE     "1" to enable dynamic int8 quantization
"""
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `uvicorn api.main:app` from any folder
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, HTTPException, Query, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

from ocr_engine import InferenceEngine  # noqa: E402

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
JOB_TTL_SECONDS = 15 * 60
MAX_JOBS = 200
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
WORK_ROOT = Path(tempfile.gettempdir()) / "ocr_redaction_jobs"

app = FastAPI(
    title="CPU OCR & Text Redaction API",
    version="2.0.0",
    description="CRNN (trained on IIIT-5K) OCR with OpenCV text detection and "
                "inpainting-based redaction. CPU only.",
)

_engine = None
_engine_lock = threading.Lock()   # guards lazy model loading
_infer_lock = threading.Lock()    # one inference at a time on the CPU model
_jobs = {}                        # job_id -> {"dir": Path, "created": float, "redacted": str, "json": str}
_jobs_lock = threading.Lock()


def get_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = InferenceEngine(
                    model_path=os.environ.get("OCR_MODEL_PATH") or None,
                    threads=int(os.environ.get("OCR_THREADS", "4")),
                    quantize=os.environ.get("OCR_QUANTIZE") == "1",
                )
    return _engine


def _purge_jobs():
    """Delete expired jobs, and the oldest ones if there are too many."""
    now = time.time()
    with _jobs_lock:
        expired = [j for j, v in _jobs.items() if now - v["created"] > JOB_TTL_SECONDS]
        overflow = max(0, len(_jobs) - len(expired) - MAX_JOBS)
        oldest = sorted((v["created"], j) for j, v in _jobs.items() if j not in expired)
        expired += [j for _, j in oldest[:overflow]]
        for job_id in expired:
            shutil.rmtree(_jobs.pop(job_id)["dir"], ignore_errors=True)


@app.get("/health")
def health():
    return {"status": "ok", "device": "cpu", "model_loaded": _engine is not None}


# A plain `def` endpoint runs in FastAPI's thread pool, so the CPU-heavy OCR
# does not block the event loop (an `async def` one would).
@app.post("/ocr")
def ocr(
    file: UploadFile = File(...),
    scale: float = Query(1.0, gt=0.1, le=4.0, description="Resize factor before OCR"),
    grayscale: bool = Query(True),
    sharpen: bool = Query(False, description="Unsharp mask (model was trained without it)"),
    mode: Literal["auto", "word", "document"] = Query("auto"),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, f"Unsupported image format; use one of {sorted(ALLOWED_SUFFIXES)}")

    _purge_jobs()
    job_id = uuid.uuid4().hex
    job_dir = WORK_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    source = job_dir / f"input{suffix}"

    try:
        size = 0
        with source.open("wb") as out:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"File larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
                out.write(chunk)

        t0 = time.perf_counter()
        engine = get_engine()
        with _infer_lock:
            result = engine.process(
                source, job_dir, scale=scale, grayscale=grayscale, sharpen=sharpen, mode=mode
            )
        total_ms = (time.perf_counter() - t0) * 1000
        source.unlink(missing_ok=True)  # do not keep the uploaded document

        with _jobs_lock:
            _jobs[job_id] = {
                "dir": job_dir,
                "created": time.time(),
                "redacted": result["redacted_path"],
                "json": result["json_path"],
            }
        return {
            "job_id": job_id,
            "words": result["words"],
            "image_size": result["image_size"],
            "redacted_image_url": f"/ocr/{job_id}/redacted",
            "text_json_url": f"/ocr/{job_id}/text",
            "inference_ms": round(result["ocr_inference_seconds"] * 1000, 3),
            "total_ms": round(total_ms, 3),
        }
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except ValueError as exc:  # unreadable / corrupt image
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(400, str(exc))
    except FileNotFoundError as exc:  # model checkpoint missing
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(503, str(exc))
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(500, "OCR processing failed")


def _job_file(job_id, key):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job or not Path(job[key]).exists():
        raise HTTPException(404, "Result not found or expired")
    return job[key]


@app.get("/ocr/{job_id}/redacted")
def get_redacted(job_id: str):
    return FileResponse(_job_file(job_id, "redacted"), media_type="image/png",
                        filename="image_redacted.png")


@app.get("/ocr/{job_id}/text")
def get_text(job_id: str):
    return FileResponse(_job_file(job_id, "json"), media_type="application/json",
                        filename="image_text.json")
