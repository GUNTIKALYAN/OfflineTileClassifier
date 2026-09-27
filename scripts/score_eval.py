"""Score eval_set against eval_labels.csv using the frozen artifact. Does not train."""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.classifier import classifier  # noqa: E402

EVAL_DIR = ROOT / "be-mlsys-assignment-dataset" / "eval_set"
LABELS_CSV = ROOT / "be-mlsys-assignment-dataset" / "eval_labels.csv"


def main() -> None:
    if not EVAL_DIR.is_dir():
        raise SystemExit(f"missing eval_set: {EVAL_DIR}")
    if not LABELS_CSV.is_file():
        raise SystemExit(f"missing labels: {LABELS_CSV}")

    labels: dict[str, str] = {}
    with LABELS_CSV.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels[row["filename"]] = row["true_label"]

    correct = 0
    total = 0
    per_class: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )

    for path in sorted(EVAL_DIR.glob("*.png")):
        true_label = labels.get(path.name)
        if true_label is None:
            print(f"skip {path.name}: no label in CSV")
            continue
        pred, _, _ = classifier.predict(path)
        total += 1
        per_class[true_label]["total"] += 1
        if pred == true_label:
            correct += 1
            per_class[true_label]["correct"] += 1

    accuracy = correct / total if total else float("nan")
    print(f"eval accuracy {accuracy:.3f}  ({correct}/{total})")
    print("per-class correct/total")
    for class_name in classifier.classes:
        stats = per_class[class_name]
        print(f"  {class_name:12s} {stats['correct']}/{stats['total']}")


if __name__ == "__main__":
    main()
