"""Command-line wrapper around the reusable inference engine.

    python textrecognizer.py image.png
    python -m cli.main image.png --scale 1.5 --sharpen

Writes ``<stem>_text.json`` and ``<stem>_redacted.png`` (next to the input, or
into ``--output-dir``). All OCR logic lives in ``ocr_engine``.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `python cli/main.py` from any folder
    sys.path.insert(0, str(ROOT))

from ocr_engine import InferenceEngine  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(
        prog="textrecognizer",
        description="CPU OCR + text redaction: reads a PNG/JPG and writes "
                    "<stem>_text.json and <stem>_redacted.png.",
    )
    p.add_argument("image", help="Input PNG/JPG image")
    p.add_argument("--model", default=None,
                   help="Model checkpoint (default: models/ocr_crnn.pt next to this project)")
    p.add_argument("--output-dir", default=None,
                   help="Where to write the outputs (default: next to the input image)")
    p.add_argument("--threads", type=int, default=4, help="CPU threads for PyTorch")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Resize factor applied before OCR (e.g. 1.5 to enlarge small text)")
    p.add_argument("--no-grayscale", action="store_true", help="Skip grayscale conversion")
    p.add_argument("--sharpen", action="store_true",
                   help="Apply an unsharp mask (off by default: the model was trained without it)")
    p.add_argument("--no-sharpen", action="store_true", help=argparse.SUPPRESS)  # old flag, now a no-op
    p.add_argument("--mode", choices=["auto", "word", "document"], default="auto",
                   help="auto: small crops = one word, larger images = full page; "
                        "word: whole image is one word; document: always detect words")
    p.add_argument("--quantize", action="store_true",
                   help="Dynamic int8 quantization of the recognizer (faster on some CPUs)")
    return p


def build_engine(args):
    return InferenceEngine(
        model_path=args.model,
        threads=args.threads,
        scale=args.scale,
        grayscale=not args.no_grayscale,
        sharpen=args.sharpen,
        mode=args.mode,
        quantize=args.quantize,
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        engine = build_engine(args)
        result = engine.process(args.image, args.output_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("Detected words:")
    for item in result["words"]:
        print(f"  {item['word']}  {item['bounding_box']}")
    print(f"\nJSON: {result['json_path']}")
    print(f"Redacted image: {result['redacted_path']}")
    t = result["timings"]
    print(
        f"OCR model time: {result['ocr_inference_seconds'] * 1000:.2f} ms "
        f"(detect {t['detect_s'] * 1000:.0f} ms, redact {t['redact_s'] * 1000:.0f} ms)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
