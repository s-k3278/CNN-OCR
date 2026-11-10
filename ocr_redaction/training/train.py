"""Train the lightweight CRNN OCR model on IIIT-5K using CPU only.

The training pipeline keeps a held-out validation split separate from the final
IIIT-5K test set, uses aspect-ratio-preserving word images, and cosine
learning-rate decay.

Changes in this version (all contained in this file):
  * stronger on-the-fly data augmentation (training split only)
  * extra dropout during training only (checkpoint format is unchanged)
  * learning-rate warm-up before the cosine decay
  * larger AdamW weight decay
  * history file is written after every epoch
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset

from .dataset import IIIT5KDataset
from .metrics import aggregate_metrics
from .model import CRNNOCR
from .vocabulary import Vocabulary


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# Data augmentation
# --------------------------------------------------------------------------- #
def strong_augment(img, s=1.0):
    """Random, realistic distortions of a grayscale uint8 word image.

    The image size never changes, so the word width / CTC input length computed
    by the dataset stays correct. `s` scales every magnitude (0 = no change).
    Each call draws new random values, so every epoch sees a different version
    of the same picture.
    """
    if s <= 0:
        return img
    rnd = np.random
    h, w = img.shape[:2]
    out = img

    # --- geometric: small rotation + shear + scale ---
    if rnd.rand() < 0.6:
        angle = rnd.uniform(-4.0, 4.0) * s
        shear = rnd.uniform(-0.15, 0.15) * s
        scale = rnd.uniform(1.0 - 0.08 * s, 1.0 + 0.04 * s)
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        M[0, 1] += shear
        M[0, 2] -= shear * (h / 2.0)  # keep the shear centred on the image
        out = cv2.warpAffine(
            out, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

    # --- geometric: mild perspective (photos taken at an angle) ---
    if rnd.rand() < 0.3:
        dx, dy = w * 0.04 * s, h * 0.08 * s
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        jitter = np.stack(
            [rnd.uniform(-dx, dx, 4), rnd.uniform(-dy, dy, 4)], axis=1
        ).astype(np.float32)
        P = cv2.getPerspectiveTransform(src, src + jitter)
        out = cv2.warpPerspective(
            out, P, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

    # --- photometric: contrast / brightness ---
    if rnd.rand() < 0.6:
        alpha = rnd.uniform(1.0 - 0.3 * s, 1.0 + 0.3 * s)
        beta = rnd.uniform(-25.0, 25.0) * s
        out = cv2.convertScaleAbs(out, alpha=alpha, beta=beta)

    # --- photometric: gamma ---
    if rnd.rand() < 0.3:
        gamma = rnd.uniform(1.0 - 0.3 * s, 1.0 + 0.5 * s)
        lut = (np.linspace(0, 1, 256) ** gamma * 255).astype(np.uint8)
        out = cv2.LUT(out, lut)

    # --- quality loss: blur or low-resolution round-trip ---
    r = rnd.rand()
    if r < 0.25:
        k = int(rnd.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    elif r < 0.45:
        f = rnd.uniform(0.45, 0.8)
        small = cv2.resize(
            out, (max(2, int(w * f)), max(2, int(h * f))), interpolation=cv2.INTER_AREA
        )
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

    # --- sensor noise ---
    if rnd.rand() < 0.3:
        sigma = rnd.uniform(3.0, 12.0) * s
        noisy = out.astype(np.float32) + rnd.randn(h, w).astype(np.float32) * sigma
        out = np.clip(noisy, 0, 255).astype(np.uint8)

    return out


class AugmentedIIIT5K(IIIT5KDataset):
    """IIIT5KDataset whose training-time augmentation is `strong_augment`."""

    def __init__(self, *args, aug_strength=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.aug_strength = float(aug_strength)

    def _augment(self, img):
        return strong_augment(img, self.aug_strength)


# --------------------------------------------------------------------------- #
# Dropout (training only, no change to the saved model structure)
# --------------------------------------------------------------------------- #
def add_training_dropout(model, feat_p, cls_p, rnn_p):
    """Attach extra dropout using hooks.

    Hooks respect model.train()/model.eval(): they are active while training
    and switched off automatically at evaluation / inference. No layers are
    added, so the state_dict keys stay identical and models/ocr_crnn.pt still
    loads in ocr_engine/recognizer.py.
    """
    if rnn_p is not None:
        model.rnn.dropout = float(rnn_p)  # dropout between the two LSTM layers
    if feat_p > 0:  # on the CNN feature map, before the LSTM
        model.cnn.register_forward_hook(
            lambda m, inp, out: F.dropout(out, feat_p, m.training)
        )
    if cls_p > 0:  # on the LSTM output, before the character classifier
        model.classifier.register_forward_pre_hook(
            lambda m, inp: (F.dropout(inp[0], cls_p, m.training),)
        )


def collate(batch, vocab):
    images, texts, valid_widths = zip(*batch)
    targets_list = [vocab.encode(t) for t in texts]
    target_lengths = torch.tensor([len(x) for x in targets_list], dtype=torch.long)
    targets = torch.tensor(
        [x for xs in targets_list for x in xs], dtype=torch.long
    )
    # The CNN has two width-2 pooling layers, so W -> floor(W/4).
    input_lengths = torch.tensor(
        [max(1, w // 4) for w in valid_widths], dtype=torch.long
    )
    return torch.stack(images), targets, target_lengths, list(texts), input_lengths


def greedy_decode(logits, vocab):
    ids = logits.argmax(-1).transpose(0, 1)
    return [vocab.decode_greedy(row.tolist()) for row in ids]


def evaluate_loader(model, loader, vocab):
    model.eval()
    preds, truths = [], []
    with torch.inference_mode():
        for images, _, _, texts, input_lengths in loader:
            logits = model(images)
            for b, length in enumerate(input_lengths.tolist()):
                preds.append(vocab.decode_greedy(logits[:length, b].argmax(-1).tolist()))
            truths.extend(texts)
    return aggregate_metrics(preds, truths)


def build_scheduler(optimizer, epochs, warmup_epochs, min_lr):
    """Linear warm-up for `warmup_epochs`, then cosine decay to `min_lr`."""
    warmup_epochs = max(0, min(warmup_epochs, epochs - 1))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=min_lr
    )
    if warmup_epochs == 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_epochs]
    )


def main(args):
    set_seed(args.seed)
    threads = max(1, min(args.threads, os.cpu_count() or 1))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(2, threads)))
    except RuntimeError:
        # PyTorch only permits setting inter-op threads before parallel work.
        pass

    device = torch.device("cpu")
    vocab = Vocabulary()

    # Separate dataset instances so validation is never augmented.
    full_train = AugmentedIIIT5K(
        args.data_root, "train", args.height, args.width,
        augment=True, aug_strength=args.aug_strength,
    )
    full_eval = IIIT5KDataset(
        args.data_root, "train", args.height, args.width, augment=False
    )

    indices = np.arange(len(full_train))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(indices)
    val_size = max(1, int(round(len(indices) * args.val_split)))
    val_indices = indices[:val_size].tolist()
    train_indices = indices[val_size:].tolist()

    train_ds = Subset(full_train, train_indices)
    val_ds = Subset(full_eval, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda batch: collate(batch, vocab),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda batch: collate(batch, vocab),
    )

    model = CRNNOCR(vocab.size, hidden=args.hidden).to(device)
    add_training_dropout(model, args.feat_dropout, args.dropout, args.rnn_dropout)
    criterion = nn.CTCLoss(blank=vocab.blank_idx, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(
        optimizer, args.epochs, args.warmup_epochs, args.min_lr
    )

    print(
        f"train={len(train_indices)} val={len(val_indices)} | "
        f"aug_strength={args.aug_strength} | dropout: feat={args.feat_dropout} "
        f"rnn={args.rnn_dropout} cls={args.dropout} | "
        f"warmup={args.warmup_epochs} wd={args.weight_decay}"
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    vocab_path = out.with_suffix(".vocab.json")
    history_path = out.with_suffix(".history.json")
    vocab.save(vocab_path)

    best_val_edit = float("inf")
    best_val_acc = -1.0
    patience_count = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        start = time.perf_counter()

        for images, targets, target_lengths, _, input_lengths in train_loader:
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            log_probs = logits.log_softmax(2)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.item())

        avg_loss = running / max(1, len(train_loader))
        val_metrics = evaluate_loader(model, val_loader, vocab)
        lr = optimizer.param_groups[0]["lr"]  # lr that was used this epoch
        scheduler.step()
        elapsed = time.perf_counter() - start

        row = {
            "epoch": epoch,
            "loss": avg_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_avg_edit_distance": val_metrics["avg_edit_distance"],
            "lr": lr,
            "seconds": elapsed,
        }
        history.append(row)
        history_path.write_text(json.dumps(history, indent=2))
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"loss={avg_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_edit={val_metrics['avg_edit_distance']:.3f} | "
            f"lr={lr:.6g} | {elapsed:.1f}s"
        )

        improved = (
            val_metrics["avg_edit_distance"] < best_val_edit
            or (
                val_metrics["avg_edit_distance"] == best_val_edit
                and val_metrics["accuracy"] > best_val_acc
            )
        )
        if improved:
            best_val_edit = val_metrics["avg_edit_distance"]
            best_val_acc = val_metrics["accuracy"]
            patience_count = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "height": args.height,
                    "width": args.width,
                    "hidden": args.hidden,
                    "chars": vocab.chars,
                    "epoch": epoch,
                    "val_accuracy": best_val_acc,
                    "val_avg_edit_distance": best_val_edit,
                },
                out,
            )
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    print(f"Saved best model: {out} (val_edit={best_val_edit:.3f}, val_acc={best_val_acc:.4f})")
    print(f"Saved vocabulary: {vocab_path}")
    print(f"Saved history: {history_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", default="models/ocr_crnn.pt")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--height", type=int, default=32)
    p.add_argument("--width", type=int, default=192)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--val-split", type=float, default=0.10)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--aug-strength", type=float, default=1.0,
                   help="Scale of data augmentation; 0 disables it.")
    p.add_argument("--dropout", type=float, default=0.3,
                   help="Dropout before the classifier (training only).")
    p.add_argument("--feat-dropout", type=float, default=0.1,
                   help="Dropout on the CNN features before the LSTM.")
    p.add_argument("--rnn-dropout", type=float, default=0.25,
                   help="Dropout between the two LSTM layers.")
    main(p.parse_args())