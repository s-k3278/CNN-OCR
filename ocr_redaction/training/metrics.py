
"""OCR evaluation metrics."""
from difflib import SequenceMatcher


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                cur[-1] + 1,
                prev[j] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = cur
    return prev[-1]


def character_counts(pred, truth):
    # Character-level overlap based on aligned edit operations.
    # This is intentionally simple and deterministic for a small assignment.
    sm = SequenceMatcher(a=truth, b=pred)
    tp = fp = fn = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            tp += i2 - i1
        elif tag == "replace":
            m = min(i2-i1, j2-j1)
            tp += m
            fp += (j2-j1) - m
            fn += (i2-i1) - m
        elif tag == "insert":
            fp += j2-j1
        elif tag == "delete":
            fn += i2-i1
    return tp, fp, fn


def aggregate_metrics(preds, truths):
    exact = sum(p == t for p, t in zip(preds, truths))
    tp = fp = fn = 0
    distances = []
    for p, t in zip(preds, truths):
        a, b, c = character_counts(p, t)
        tp += a; fp += b; fn += c
        distances.append(edit_distance(p, t))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "samples": len(truths),
        "accuracy": exact / len(truths) if truths else 0.0,
        "precision": precision,
        "recall": recall,
        "avg_edit_distance": sum(distances) / len(distances) if distances else 0.0,
        "exact_matches": exact,
    }
