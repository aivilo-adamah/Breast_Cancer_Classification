from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset


def load_split_rows(split_csv_path: str | Path) -> list[dict[str, str]]:
    path = Path(split_csv_path).expanduser()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def extract_labels(rows: list[dict[str, str]]) -> list[int]:
    return [int(row["label"]) for row in rows]


def compute_positive_class_weight(labels: list[int]) -> float:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count

    if positive_count == 0:
        raise ValueError("Training split contains no positive samples.")
    if negative_count == 0:
        raise ValueError("Training split contains no negative samples.")

    # BCEWithLogitsLoss uses this value to upweight errors on the minority
    # positive class.
    return negative_count / positive_count


def summarize_label_counts(labels: list[int]) -> dict[str, int]:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    return {
        "total": len(labels),
        "positive": positive_count,
        "negative": negative_count,
    }


def apply_train_augmentations(
    image: Image.Image,
    *,
    color_jitter_strength: float = 0.0,
) -> Image.Image:
    # Augmentation stays intentionally light because the pathology patches are
    # small and aggressive transforms can destroy local texture cues.
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    rotation = random.choice((0, 90, 180, 270))
    if rotation:
        image = image.rotate(rotation)

    if color_jitter_strength > 0.0:
        jitter_low = 1.0 - color_jitter_strength
        jitter_high = 1.0 + color_jitter_strength
        image = ImageEnhance.Brightness(image).enhance(random.uniform(jitter_low, jitter_high))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(jitter_low, jitter_high))
        image = ImageEnhance.Color(image).enhance(random.uniform(jitter_low, jitter_high))

    return image


@dataclass(frozen=True)
class DatasetNormalization:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def compute_dataset_normalization(
    split_csv_path: str | Path,
    dataset_dir: str | Path,
    *,
    max_samples: int | None = None,
    seed: int = 42,
) -> DatasetNormalization:
    dataset_root = Path(dataset_dir).expanduser()
    rows = load_split_rows(split_csv_path)
    if not rows:
        raise ValueError(f"No rows found in split CSV: {split_csv_path}")

    if max_samples is not None and max_samples > 0 and len(rows) > max_samples:
        rows = random.Random(seed).sample(rows, max_samples)

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_square_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    # Mean/std are estimated from the training split only to avoid leaking
    # validation or test information into preprocessing.
    for row in rows:
        image_path = dataset_root / row["relative_path"]
        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
            flat = (np.asarray(image, dtype=np.float32) / 255.0).reshape(-1, 3).astype(np.float64)

        channel_sum += flat.sum(axis=0)
        channel_square_sum += np.square(flat).sum(axis=0)
        total_pixels += flat.shape[0]

    mean = channel_sum / total_pixels
    variance = (channel_square_sum / total_pixels) - np.square(mean)
    std = np.sqrt(np.clip(variance, a_min=1e-12, a_max=None))

    return DatasetNormalization(
        mean=tuple(float(value) for value in mean),
        std=tuple(float(value) for value in std),
    )


class IDCPatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        split_csv_path: str | Path,
        dataset_dir: str | Path,
        *,
        augment: bool = False,
        normalization: DatasetNormalization | None = None,
        color_jitter_strength: float = 0.0,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        self.split_csv_path = Path(split_csv_path).expanduser()
        self.dataset_dir = Path(dataset_dir).expanduser()
        self.records = load_split_rows(self.split_csv_path)
        self.augment = augment
        self.normalization = normalization
        self.color_jitter_strength = color_jitter_strength
        self.image_size = image_size

        if not self.records:
            raise ValueError(f"No rows found in split CSV: {self.split_csv_path}")

        if not 0.0 <= self.color_jitter_strength < 1.0:
            raise ValueError("color_jitter_strength must be in [0.0, 1.0).")
        if self.image_size is not None:
            if len(self.image_size) != 2:
                raise ValueError("image_size must be a tuple of (width, height).")
            width, height = self.image_size
            if width < 1 or height < 1:
                raise ValueError("image_size values must be positive integers.")

        self.labels = extract_labels(self.records)
        self._normalization_mean: torch.Tensor | None = None
        self._normalization_std: torch.Tensor | None = None

        if self.normalization is not None:
            self._normalization_mean = torch.tensor(
                self.normalization.mean,
                dtype=torch.float32,
            ).view(3, 1, 1)
            self._normalization_std = torch.tensor(
                self.normalization.std,
                dtype=torch.float32,
            ).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.records[index]
        image_path = self.dataset_dir / row["relative_path"]

        if not image_path.exists():
            raise FileNotFoundError(f"Image path not found: {image_path}")

        with Image.open(image_path) as image_handle:
            image = image_handle.convert("RGB")
            if self.image_size is not None and image.size != self.image_size:
                # Resizing is mainly used by pretrained comparison models.
                image = image.resize(self.image_size, resample=Image.BILINEAR)
            if self.augment:
                image = apply_train_augmentations(
                    image,
                    color_jitter_strength=self.color_jitter_strength,
                )

            array = np.asarray(image, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(array).permute(2, 0, 1)

        if self.normalization is not None:
            tensor = (tensor - self._normalization_mean) / self._normalization_std

        label = torch.tensor(float(int(row["label"])), dtype=torch.float32)
        return tensor, label


def describe_split(split_csv_path: str | Path) -> dict[str, Any]:
    rows = load_split_rows(split_csv_path)
    labels = extract_labels(rows)
    summary = summarize_label_counts(labels)
    summary["path"] = str(Path(split_csv_path).expanduser())
    return summary
