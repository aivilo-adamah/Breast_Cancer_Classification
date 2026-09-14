"""Copy a handful of validation patches into models/demo/sample_patches/ for the app gallery."""

from __future__ import annotations

import csv
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "IDC_regular_ps50_idx5"
VAL_CSV = PROJECT_ROOT / "artifacts" / "splits" / "val.csv"
GALLERY_DIR = PROJECT_ROOT / "models" / "demo" / "sample_patches"


def main(samples_per_class: int = 6, seed: int = 7) -> None:
    with VAL_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = {"0": [], "1": []}
    for row in rows:
        by_label[row["label"]].append(row)

    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    for existing in GALLERY_DIR.glob("*.png"):
        existing.unlink()

    for label, label_rows in by_label.items():
        chosen = rng.sample(label_rows, min(samples_per_class, len(label_rows)))
        for row in chosen:
            source = DATASET_DIR / row["relative_path"]
            destination = GALLERY_DIR / f"class{label}_{source.name}"
            shutil.copy2(source, destination)

    print(f"Copied gallery patches to {GALLERY_DIR}")


if __name__ == "__main__":
    main()
