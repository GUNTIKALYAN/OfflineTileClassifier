"""Train the tile CNN once on candidate_tiles; write artifacts/model.pt and meta.json."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model import IMAGE_SIZE, TileCNN, image_to_tensor  # noqa: E402

DATA_DIR = ROOT / "be-mlsys-assignment-dataset" / "candidate_tiles"
ARTIFACTS_DIR = ROOT / "artifacts"
BATCH_SIZE = 32
EPOCHS = 15
SEED = 18
LR = 0.001


def load_records() -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    for class_dir in sorted(path for path in DATA_DIR.iterdir() if path.is_dir()):
        label = class_dir.name
        for image_path in sorted(class_dir.glob("*.png")):
            records.append({"path": image_path, "label": label})
    classes = sorted({row["label"] for row in records})
    label_to_index = {label: index for index, label in enumerate(classes)}
    for row in records:
        row["class_index"] = label_to_index[row["label"]]
    return records, classes


def stack_split(split_records: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    images = [image_to_tensor(row["path"]) for row in split_records]
    labels = [row["class_index"] for row in split_records]
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def choose_threshold(val_confidence: torch.Tensor) -> tuple[float, str]:
    bin_counts = []
    print("confidence  count")
    for bin_index in range(10):
        low = bin_index / 10
        high = (bin_index + 1) / 10
        if bin_index == 9:
            count = int(((val_confidence >= low) & (val_confidence <= high)).sum())
        else:
            count = int(((val_confidence >= low) & (val_confidence < high)).sum())
        bin_counts.append(count)
        print(f"{low:.1f}-{high:.1f}     {count:4d}  {'#' * count}")

    tallest = max(bin_counts)
    peak_low = bin_counts.index(tallest) / 10
    flat = tallest < 0.4 * len(val_confidence)
    if flat:
        threshold = 0.55
        reason = "flat histogram, tallest bin is under 40%"
    elif peak_low >= 0.7:
        threshold = 0.70
        reason = "tallest bin is at 0.7 or above"
    else:
        threshold = 0.55
        reason = "the pile is below 0.7, so the cutoff stays 0.55"
    print(f"threshold {threshold:.2f}  ({reason})")
    return threshold, reason


def main() -> None:
    if not DATA_DIR.is_dir():
        raise SystemExit(f"missing dataset folder: {DATA_DIR}")

    torch.manual_seed(SEED)

    records, classes = load_records()
    print(f"loaded {len(records)} images from {DATA_DIR}")
    print("classes", classes)
    print(Counter(row["label"] for row in records))

    train_records, val_records = train_test_split(
        records,
        test_size=0.2,
        random_state=SEED,
        stratify=[row["label"] for row in records],
    )
    print(f"train {len(train_records)}  val {len(val_records)}")

    train_x, train_y = stack_split(train_records)
    val_x, val_y = stack_split(val_records)

    device = torch.device("cpu")
    model = TileCNN(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    best_val_accuracy = -1.0
    best_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            scores = model(images)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            total_seen += images.size(0)
        train_loss = total_loss / total_seen

        model.eval()
        correct = 0
        seen = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                pred = model(images).argmax(dim=1)
                correct += (pred == labels).sum().item()
                seen += labels.size(0)
        val_accuracy = correct / seen
        print(
            f"epoch {epoch:02d}  train_loss {train_loss:.4f}  "
            f"val_accuracy {val_accuracy:.3f}"
        )
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    print(f"best val_accuracy {best_val_accuracy:.3f}")

    with torch.no_grad():
        val_scores = model(val_x.to(device))
        val_confidence = torch.softmax(val_scores, dim=1).max(dim=1).values
    threshold, _ = choose_threshold(val_confidence)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "model.pt"
    torch.save({"state_dict": model.state_dict(), "classes": classes}, model_path)

    meta = {
        "classes": classes,
        "image_size": IMAGE_SIZE,
        "threshold": threshold,
        "preprocess": "rgb_div_255",
    }
    meta_path = ARTIFACTS_DIR / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {model_path}")
    print(f"wrote {meta_path}")
    print(meta)


if __name__ == "__main__":
    main()
