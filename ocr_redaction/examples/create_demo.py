"""Create a synthetic multi-word document from IIIT-5K test crops."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def resolve_layout(root):
    root = Path(root).expanduser().resolve()
    csv_candidates = [root, root / "train_test"]
    image_candidates = [
        root / "IIIT5K-Word_V3.0" / "IIIT5K",
        root,
    ]
    csv_dir = next(
        (p for p in csv_candidates if (p / "testdata.csv").exists()), None
    )
    image_root = next(
        (p for p in image_candidates if (p / "test").is_dir()), None
    )
    if csv_dir is None or image_root is None:
        raise FileNotFoundError(
            "Could not resolve IIIT-5K demo layout. Expected testdata.csv and test/."
        )
    return csv_dir, image_root


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", default="examples/demo_document.png")
    p.add_argument("--count", type=int, default=5)
    args = p.parse_args()

    csv_dir, image_root = resolve_layout(args.data_root)
    df = pd.read_csv(csv_dir / "testdata.csv").head(args.count)

    canvas = np.full((700, 1200, 3), 255, dtype=np.uint8)
    y = 70
    placed = 0

    for _, row in df.iterrows():
        rel = str(row["ImgName"]).replace("\\", "/")
        img = cv2.imread(str(image_root / rel))
        if img is None:
            print(f"WARNING: Could not read {image_root / rel}")
            continue

        max_width = 420
        scale = min(1.0, max_width / img.shape[1])
        if scale != 1.0:
            img = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )

        h, w = img.shape[:2]
        if y + h >= canvas.shape[0]:
            break

        # Each source crop is a complete word. Put it on a clean line with
        # generous whitespace so the detector can group its characters.
        x = 70
        canvas[y:y + h, x:x + w] = img
        y += h + 65
        placed += 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas)
    print(f"Created demo document: {output}")
    print(f"Words placed: {placed}")


if __name__ == "__main__":
    main()
