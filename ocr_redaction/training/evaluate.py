"""Evaluate a trained CRNN on the IIIT-5K test split."""
import argparse
import ast
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import IIIT5KDataset
from .metrics import aggregate_metrics, edit_distance
from .model import CRNNOCR
from .train import collate, greedy_decode
from .vocabulary import Vocabulary


def parse_lexicon(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def correct_with_lexicon(pred, candidates):
    """Use the nearest lexicon candidate as a lightweight post-decoder.

    This is intentionally separate from raw CRNN metrics so the report can
    distinguish model recognition from lexicon-assisted OCR.
    """
    candidates = [c for c in candidates if c]
    if not candidates:
        return pred
    return min(candidates, key=lambda c: (edit_distance(pred, c), abs(len(c) - len(pred))))


def main(args):
    threads = max(1, min(args.threads, os.cpu_count() or 1))
    torch.set_num_threads(threads)

    ckpt = torch.load(args.model, map_location="cpu", weights_only=True)
    vocab = Vocabulary(ckpt["chars"])
    ds = IIIT5KDataset(args.data_root, "test", ckpt["height"], ckpt["width"], augment=False)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate(batch, vocab),
    )

    model = CRNNOCR(vocab.size, hidden=ckpt["hidden"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds, truths = [], []
    total_time = 0.0
    with torch.inference_mode():
        for images, _, _, texts, input_lengths in loader:
            t0 = time.perf_counter()
            logits = model(images)
            total_time += time.perf_counter() - t0
            for b, length in enumerate(input_lengths.tolist()):
                preds.append(vocab.decode_greedy(logits[:length, b].argmax(-1).tolist()))
            truths.extend(texts)

    result = aggregate_metrics(preds, truths)
    result["avg_inference_ms_per_image"] = total_time / len(ds) * 1000
    result["cpu_threads"] = torch.get_num_threads()

    # Optional lexicon-assisted evaluation. The supplied IIIT-5K CSV contains
    # per-image small/medium lexicons, and this reports it separately.
    if args.lexicon != "none":
        csv_dir, _ = ds.csv_dir, ds.image_root
        csv = pd.read_csv(csv_dir / "testdata.csv")
        col = "smallLexi" if args.lexicon == "small" else "mediumLexi"
        lex_preds = [
            correct_with_lexicon(pred, parse_lexicon(row[col]))
            for pred, (_, row) in zip(preds, csv.iterrows())
        ]
        lex_result = aggregate_metrics(lex_preds, truths)
        result["lexicon_mode"] = args.lexicon
        result["lexicon_accuracy"] = lex_result["accuracy"]
        result["lexicon_precision"] = lex_result["precision"]
        result["lexicon_recall"] = lex_result["recall"]
        result["lexicon_avg_edit_distance"] = lex_result["avg_edit_distance"]
        result["lexicon_exact_matches"] = lex_result["exact_matches"]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--model", default="models/ocr_crnn.pt")
    p.add_argument("--output", default="outputs/evaluation.json")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--lexicon", choices=["none", "small", "medium"], default="medium")
    main(p.parse_args())
