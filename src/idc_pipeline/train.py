from __future__ import annotations

from dataclasses import dataclass, asdict
import csv
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .dataset import (
    DatasetNormalization,
    IDCPatchDataset,
    compute_positive_class_weight,
    describe_split,
    summarize_label_counts,
)
from .evaluate import BinaryClassificationMetrics, compute_binary_metrics, evaluate_model
from .model import build_model


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 128
    epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    early_stopping_patience: int = 4
    threshold: float = 0.5
    dropout: float = 0.30
    # The baseline path uses "baseline_cnn"; the pretrained comparison uses "resnet18".
    architecture: str = "baseline_cnn"
    pretrained: bool = False
    # image_size stays None for 50x50 baseline patches and is only set for comparison models.
    image_size: tuple[int, int] | None = None
    normalization_mean: tuple[float, float, float] | None = None
    normalization_std: tuple[float, float, float] | None = None
    color_jitter_strength: float = 0.0
    scheduler_name: str = "none"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 1
    scheduler_min_lr: float = 1e-6
    selection_metric: str = "roc_auc"
    pin_memory: bool = True
    verbose: bool = True

    def validate(self) -> None:
        if self.batch_size < 1:
            raise ValueError("TrainingConfig.batch_size must be at least 1.")
        if self.epochs < 1:
            raise ValueError("TrainingConfig.epochs must be at least 1.")
        if self.learning_rate <= 0.0:
            raise ValueError("TrainingConfig.learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("TrainingConfig.weight_decay must be non-negative.")
        if self.num_workers < 0:
            raise ValueError("TrainingConfig.num_workers must be non-negative.")
        if self.early_stopping_patience < 1:
            raise ValueError("TrainingConfig.early_stopping_patience must be at least 1.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("TrainingConfig.dropout must be in [0.0, 1.0).")
        if self.architecture not in {"baseline_cnn", "resnet18"}:
            raise ValueError("Unsupported architecture.")
        if not 0.0 <= self.color_jitter_strength < 1.0:
            raise ValueError("TrainingConfig.color_jitter_strength must be in [0.0, 1.0).")
        if self.image_size is not None:
            if len(self.image_size) != 2:
                raise ValueError("TrainingConfig.image_size must be a tuple of (width, height).")
            width, height = self.image_size
            if width < 1 or height < 1:
                raise ValueError("TrainingConfig.image_size values must be positive integers.")
        has_mean = self.normalization_mean is not None
        has_std = self.normalization_std is not None
        if has_mean != has_std:
            raise ValueError(
                "TrainingConfig.normalization_mean and normalization_std must both be set or both be None."
            )
        if self.scheduler_name not in {"none", "reduce_on_plateau"}:
            raise ValueError("Unsupported scheduler_name.")
        if not 0.0 < self.scheduler_factor < 1.0:
            raise ValueError("TrainingConfig.scheduler_factor must be in (0.0, 1.0).")
        if self.scheduler_patience < 0:
            raise ValueError("TrainingConfig.scheduler_patience must be non-negative.")
        if self.scheduler_min_lr < 0.0:
            raise ValueError("TrainingConfig.scheduler_min_lr must be non-negative.")
        if self.selection_metric not in {"roc_auc", "f1", "accuracy", "recall", "precision"}:
            raise ValueError("Unsupported selection metric.")


@dataclass(frozen=True)
class TrainingRunResult:
    output_dir: Path
    checkpoint_path: Path
    history_path: Path
    metrics_path: Path
    config_path: Path
    best_epoch: int
    best_metric_name: str
    best_metric_value: float
    train_summary: dict[str, int]
    val_summary: dict[str, int]
    test_summary: dict[str, int]
    pos_weight: float
    test_metrics: dict[str, Any]
    best_val_metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "checkpoint_path": str(self.checkpoint_path),
            "history_path": str(self.history_path),
            "metrics_path": str(self.metrics_path),
            "config_path": str(self.config_path),
            "best_epoch": self.best_epoch,
            "best_metric_name": self.best_metric_name,
            "best_metric_value": self.best_metric_value,
            "train_summary": self.train_summary,
            "val_summary": self.val_summary,
            "test_summary": self.test_summary,
            "pos_weight": self.pos_weight,
            "best_val_metrics": self.best_val_metrics,
            "test_metrics": self.test_metrics,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Reproducibility matters more than maximum throughput for this coursework.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint_model(
    checkpoint_path: str | Path,
    *,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict[str, Any], torch.device]:
    resolved_device = device or resolve_device()
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location=resolved_device)

    checkpoint_config = checkpoint.get("config", {})
    architecture = str(checkpoint_config.get("architecture", "baseline_cnn"))
    dropout = float(checkpoint_config.get("dropout", 0.30))

    # The checkpoint stores the training config, so evaluation can rebuild the
    # correct architecture without notebook-specific branching.
    model = build_model(
        architecture=architecture,
        dropout=dropout,
        pretrained=False,
    ).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, resolved_device


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _metric_value(metrics: BinaryClassificationMetrics, metric_name: str) -> float:
    value = getattr(metrics, metric_name)
    if value is None:
        return float("-inf")
    return float(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Training history is empty.")

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _build_dataset_normalization(config: TrainingConfig) -> DatasetNormalization | None:
    if config.normalization_mean is None or config.normalization_std is None:
        return None

    return DatasetNormalization(
        mean=config.normalization_mean,
        std=config.normalization_std,
    )


def build_dataloaders(
    *,
    dataset_dir: str | Path,
    train_csv_path: str | Path,
    val_csv_path: str | Path,
    test_csv_path: str | Path,
    config: TrainingConfig,
) -> tuple[
    IDCPatchDataset,
    IDCPatchDataset,
    IDCPatchDataset,
    DataLoader,
    DataLoader,
    DataLoader,
]:
    normalization = _build_dataset_normalization(config)
    train_dataset = IDCPatchDataset(
        train_csv_path,
        dataset_dir,
        augment=True,
        normalization=normalization,
        color_jitter_strength=config.color_jitter_strength,
        image_size=config.image_size,
    )
    val_dataset = IDCPatchDataset(
        val_csv_path,
        dataset_dir,
        augment=False,
        normalization=normalization,
        image_size=config.image_size,
    )
    test_dataset = IDCPatchDataset(
        test_csv_path,
        dataset_dir,
        augment=False,
        normalization=normalization,
        image_size=config.image_size,
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    pin_memory = config.pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    device: torch.device,
    threshold: float,
) -> BinaryClassificationMetrics:
    model.train()

    all_labels: list[int] = []
    all_probabilities: list[float] = []
    accumulated_loss = 0.0
    total_examples = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        probabilities = torch.sigmoid(logits)

        batch_size = inputs.size(0)
        accumulated_loss += float(loss.item()) * batch_size
        total_examples += batch_size
        all_labels.extend(labels.detach().cpu().to(dtype=torch.int64).tolist())
        all_probabilities.extend(probabilities.detach().cpu().tolist())

    average_loss = accumulated_loss / total_examples
    return compute_binary_metrics(
        all_labels,
        all_probabilities,
        average_loss=average_loss,
        threshold=threshold,
    )


def run_training_experiment(
    *,
    dataset_dir: str | Path,
    train_csv_path: str | Path,
    val_csv_path: str | Path,
    test_csv_path: str | Path,
    output_dir: str | Path,
    config: TrainingConfig | None = None,
) -> TrainingRunResult:
    config = config or TrainingConfig()
    config.validate()
    set_seed(config.seed)

    device = resolve_device()
    output_directory = Path(output_dir).expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)

    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = build_dataloaders(
        dataset_dir=dataset_dir,
        train_csv_path=train_csv_path,
        val_csv_path=val_csv_path,
        test_csv_path=test_csv_path,
        config=config,
    )

    # The training split is imbalanced, so positive patches receive a higher loss
    # weight during optimization.
    pos_weight_value = compute_positive_class_weight(train_dataset.labels)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    model = build_model(
        architecture=config.architecture,
        dropout=config.dropout,
        pretrained=config.pretrained,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    if config.scheduler_name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.scheduler_min_lr,
        )

    checkpoint_path = output_directory / "best_model.pt"
    history_path = output_directory / "history.csv"
    metrics_path = output_directory / "metrics.json"
    config_path = output_directory / "config.json"

    history_rows: list[dict[str, Any]] = []
    best_metric_value = float("-inf")
    best_epoch = 0
    best_val_metrics: BinaryClassificationMetrics | None = None
    epochs_without_improvement = 0

    train_summary = summarize_label_counts(train_dataset.labels)
    val_summary = summarize_label_counts(val_dataset.labels)
    test_summary = summarize_label_counts(test_dataset.labels)

    if config.verbose:
        dataset_size_parts = [
            f"train={train_summary['total']:,}",
            f"val={val_summary['total']:,}",
            f"test={test_summary['total']:,}",
        ]

        print(
            f"Starting training on {device} | "
            f"architecture={config.architecture} | "
            f"normalization={'train_stats' if config.normalization_mean is not None else 'none'} | "
            f"color_jitter={config.color_jitter_strength:.2f} | "
            f"scheduler={config.scheduler_name} | "
            f"{', '.join(dataset_size_parts)} | "
            f"pos_weight={pos_weight_value:.4f}"
        )

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            threshold=config.threshold,
        )
        val_metrics = evaluate_model(
            model,
            val_loader,
            device=device,
            criterion=criterion,
            threshold=config.threshold,
        )

        history_rows.append(
            {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "train_precision": train_metrics.precision,
                "train_recall": train_metrics.recall,
                "train_f1": train_metrics.f1,
                "train_roc_auc": train_metrics.roc_auc,
                "val_loss": val_metrics.loss,
                "val_accuracy": val_metrics.accuracy,
                "val_precision": val_metrics.precision,
                "val_recall": val_metrics.recall,
                "val_f1": val_metrics.f1,
                "val_roc_auc": val_metrics.roc_auc,
            }
        )

        # Model selection happens on the validation set only; the test set is kept
        # untouched until the best checkpoint has already been chosen.
        current_metric_value = _metric_value(val_metrics, config.selection_metric)
        improved = current_metric_value > best_metric_value
        if current_metric_value > best_metric_value:
            best_metric_value = current_metric_value
            best_epoch = epoch
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_metric_name": config.selection_metric,
                    "best_metric_value": best_metric_value,
                    "config": asdict(config),
                    "pos_weight": pos_weight_value,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if scheduler is not None and val_metrics.loss is not None:
            scheduler.step(val_metrics.loss)

        epoch_seconds = time.perf_counter() - epoch_start
        if config.verbose:
            print(
                f"Epoch {epoch:02d}/{config.epochs} | "
                f"{epoch_seconds:.1f}s | "
                f"lr={optimizer.param_groups[0]['lr']:.6f} | "
                f"train_loss={_format_metric(train_metrics.loss)} | "
                f"val_loss={_format_metric(val_metrics.loss)} | "
                f"val_acc={_format_metric(val_metrics.accuracy)} | "
                f"val_f1={_format_metric(val_metrics.f1)} | "
                f"val_auc={_format_metric(val_metrics.roc_auc)} | "
                f"best={'yes' if improved else 'no'}"
            )

        if epochs_without_improvement >= config.early_stopping_patience:
            if config.verbose:
                print(
                    f"Early stopping triggered after epoch {epoch}. "
                    f"Best {config.selection_metric} was {best_metric_value:.4f} at epoch {best_epoch}."
                )
            break

    if best_val_metrics is None:
        raise RuntimeError("Training finished without selecting a best validation checkpoint.")

    # Reload the best validation checkpoint before running the single final test pass.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_model(
        model,
        test_loader,
        device=device,
        criterion=criterion,
        threshold=config.threshold,
    )

    _write_history_csv(history_path, history_rows)

    metrics_payload = {
        "device": str(device),
        "best_epoch": best_epoch,
        "best_metric_name": config.selection_metric,
        "best_metric_value": best_metric_value,
        "pos_weight": pos_weight_value,
        "train_summary": train_summary,
        "val_summary": val_summary,
        "test_summary": test_summary,
        "best_val_metrics": best_val_metrics.to_dict(),
        "test_metrics": test_metrics.to_dict(),
    }
    _write_json(metrics_path, metrics_payload)

    config_payload = {
        "training_config": asdict(config),
        "dataset_dir": str(Path(dataset_dir).expanduser()),
        "train_split": describe_split(train_csv_path),
        "val_split": describe_split(val_csv_path),
        "test_split": describe_split(test_csv_path),
    }
    _write_json(config_path, config_payload)

    if config.verbose:
        print(
            f"Training complete. Best epoch={best_epoch}, "
            f"best_{config.selection_metric}={best_metric_value:.4f}, "
            f"test_f1={test_metrics.f1:.4f}, "
            f"test_auc={_format_metric(test_metrics.roc_auc)}"
        )
        print(f"Saved checkpoint: {checkpoint_path}")

    return TrainingRunResult(
        output_dir=output_directory,
        checkpoint_path=checkpoint_path,
        history_path=history_path,
        metrics_path=metrics_path,
        config_path=config_path,
        best_epoch=best_epoch,
        best_metric_name=config.selection_metric,
        best_metric_value=best_metric_value,
        train_summary=train_summary,
        val_summary=val_summary,
        test_summary=test_summary,
        pos_weight=pos_weight_value,
        test_metrics=test_metrics.to_dict(),
        best_val_metrics=best_val_metrics.to_dict(),
    )
