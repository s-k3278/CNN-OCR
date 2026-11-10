import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.dataset import IIIT5KDataset
from training.vocabulary import Vocabulary
from training.model import CRNNOCR


def main(args):
    device = torch.device("cpu")

    checkpoint = torch.load(
        args.model,
        map_location=device,
        weights_only=True,
    )

    state_dict = checkpoint["model_state"]

    hidden = int(checkpoint.get("hidden", 128))
    height = int(checkpoint.get("height", 32))
    width = int(checkpoint.get("width", 192))

    vocab_path = Path(args.model).with_suffix(".vocab.json")
    vocab = Vocabulary.load(vocab_path)

    model = CRNNOCR(
        num_classes=vocab.size,
        hidden=hidden,
    )

    model.load_state_dict(state_dict)
    model.eval()

    dataset = IIIT5KDataset(
        args.data_root,
        split="test",
        height=height,
        width=width,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print("=" * 100)
    print(f"MODEL: {args.model}")
    print(f"SAMPLES: {min(args.count, len(dataset))}")
    print("=" * 100)

    print(
        f"{'#':>3}  "
        f"{'GROUND TRUTH':<25} "
        f"{'PREDICTION':<25} "
        f"IMAGE"
    )

    print("-" * 100)

    shown = 0
    exact = 0

    for images, texts, valid_widths in loader:

        with torch.inference_mode():
            logits = model(images)

        for b, valid_width in enumerate(valid_widths.tolist()):

            # CNN reduces width by 4x.
            valid_steps = max(1, valid_width // 4)

            prediction = vocab.decode_greedy(
                logits[:valid_steps, b].argmax(-1).tolist()
            )

            ground_truth = texts[b]

            image_name = dataset.df.iloc[shown]["ImgName"]

            shown += 1

            if prediction == ground_truth:
                exact += 1

            print(
                f"{shown:3d}  "
                f"{ground_truth:<25} "
                f"{prediction:<25} "
                f"{image_name}"
            )

            if shown >= args.count:
                break

        if shown >= args.count:
            break

    print("-" * 100)

    print(
        f"Exact matches: "
        f"{exact}/{shown} "
        f"({100 * exact / shown:.2f}%)"
    )

    print(f"Vocabulary: {vocab.chars}")
    print(f"Vocabulary size including CTC blank: {vocab.size}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        required=True
    )

    parser.add_argument(
        "--model",
        required=True
    )

    parser.add_argument(
        "--count",
        type=int,
        default=50
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    main(parser.parse_args())