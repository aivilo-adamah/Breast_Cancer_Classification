from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PredictionOutputs:
    labels: list[int]
    probabilities: list[float]
    average_loss: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels": self.labels,
            "probabilities": self.probabilities,
            "average_loss": self.average_loss,
        }


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    loss: float | None
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    tp: int
    tn: int
    fp: int
    fn: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
        }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_binary_roc_auc(labels: list[int], probabilities: list[float]) -> float | None:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count

    if positive_count == 0 or negative_count == 0:
        return None

    # Sorting by score lets us compute ROC quantities directly from the ranked
    # predictions without adding an external metrics dependency.
    pairs = sorted(zip(probabilities, labels), key=lambda pair: pair[0], reverse=True)
    points = [(0.0, 0.0)]
    tp = 0
    fp = 0
    previous_score: float | None = None

    for score, label in pairs:
        if previous_score is not None and score != previous_score:
            points.append((fp / negative_count, tp / positive_count))

        if label == 1:
            tp += 1
        else:
            fp += 1

        previous_score = score

    points.append((fp / negative_count, tp / positive_count))

    area = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        area += (x2 - x1) * (y1 + y2) / 2.0

    return area


def compute_binary_roc_curve(
    labels: list[int],
    probabilities: list[float],
) -> list[dict[str, float | None]]:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count

    if positive_count == 0 or negative_count == 0:
        return []

    pairs = sorted(zip(probabilities, labels), key=lambda pair: pair[0], reverse=True)
    rows: list[dict[str, float | None]] = [
        {
            "threshold": None,
            "fpr": 0.0,
            "tpr": 0.0,
            "fp": 0,
            "tp": 0,
        }
    ]

    tp = 0
    fp = 0
    previous_score: float | None = None

    for score, label in pairs:
        if previous_score is not None and score != previous_score:
            rows.append(
                {
                    "threshold": previous_score,
                    "fpr": fp / negative_count,
                    "tpr": tp / positive_count,
                    "fp": fp,
                    "tp": tp,
                }
            )

        if label == 1:
            tp += 1
        else:
            fp += 1

        previous_score = score

    rows.append(
        {
            "threshold": previous_score,
            "fpr": fp / negative_count,
            "tpr": tp / positive_count,
            "fp": fp,
            "tp": tp,
        }
    )
    return rows


def compute_binary_precision_recall_curve(
    labels: list[int],
    probabilities: list[float],
) -> list[dict[str, float | None]]:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count

    if positive_count == 0 or negative_count == 0:
        return []

    pairs = sorted(zip(probabilities, labels), key=lambda pair: pair[0], reverse=True)
    rows: list[dict[str, float | None]] = [
        {
            "threshold": None,
            "recall": 0.0,
            "precision": 1.0,
            "fp": 0,
            "tp": 0,
        }
    ]

    tp = 0
    fp = 0
    previous_score: float | None = None

    for score, label in pairs:
        if previous_score is not None and score != previous_score:
            precision = _safe_divide(tp, tp + fp)
            recall = tp / positive_count
            rows.append(
                {
                    "threshold": previous_score,
                    "recall": recall,
                    "precision": precision,
                    "fp": fp,
                    "tp": tp,
                }
            )

        if label == 1:
            tp += 1
        else:
            fp += 1

        previous_score = score

    precision = _safe_divide(tp, tp + fp)
    recall = tp / positive_count
    rows.append(
        {
            "threshold": previous_score,
            "recall": recall,
            "precision": precision,
            "fp": fp,
            "tp": tp,
        }
    )
    return rows


def compute_average_precision(labels: list[int], probabilities: list[float]) -> float | None:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count

    if positive_count == 0 or negative_count == 0:
        return None

    pairs = sorted(zip(probabilities, labels), key=lambda pair: pair[0], reverse=True)
    tp = 0
    fp = 0
    precision_sum = 0.0

    for _, label in pairs:
        if label == 1:
            tp += 1
            precision_sum += tp / (tp + fp)
        else:
            fp += 1

    return precision_sum / positive_count


def compute_binary_metrics(
    labels: list[int],
    probabilities: list[float],
    *,
    average_loss: float | None = None,
    threshold: float = 0.5,
) -> BinaryClassificationMetrics:
    # The threshold converts probabilities into class predictions; notebook 6
    # sweeps this value after training to study different operating points.
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]

    tp = sum(1 for prediction, label in zip(predictions, labels) if prediction == 1 and label == 1)
    tn = sum(1 for prediction, label in zip(predictions, labels) if prediction == 0 and label == 0)
    fp = sum(1 for prediction, label in zip(predictions, labels) if prediction == 1 and label == 0)
    fn = sum(1 for prediction, label in zip(predictions, labels) if prediction == 0 and label == 1)

    accuracy = _safe_divide(tp + tn, len(labels))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    roc_auc = compute_binary_roc_auc(labels, probabilities)

    return BinaryClassificationMetrics(
        loss=average_loss,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        tp=tp,
        tn=tn,
        fp=fp,
        fn=fn,
    )


def confusion_matrix_from_metrics(metrics: BinaryClassificationMetrics) -> list[list[int]]:
    return [
        [metrics.tn, metrics.fp],
        [metrics.fn, metrics.tp],
    ]


@torch.no_grad()
def predict_model_outputs(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    criterion: torch.nn.Module | None = None,
) -> PredictionOutputs:
    model.eval()

    all_labels: list[int] = []
    all_probabilities: list[float] = []
    accumulated_loss = 0.0
    total_examples = 0

    for inputs, labels in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(inputs)
        probabilities = torch.sigmoid(logits)

        if criterion is not None:
            loss = criterion(logits, labels)
            accumulated_loss += float(loss.item()) * inputs.size(0)

        all_labels.extend(labels.detach().cpu().to(dtype=torch.int64).tolist())
        all_probabilities.extend(probabilities.detach().cpu().tolist())
        total_examples += inputs.size(0)

    average_loss = None
    if criterion is not None and total_examples > 0:
        average_loss = accumulated_loss / total_examples

    return PredictionOutputs(
        labels=all_labels,
        probabilities=all_probabilities,
        average_loss=average_loss,
    )


def sweep_thresholds(
    labels: list[int],
    probabilities: list[float],
    thresholds: list[float],
    *,
    average_loss: float | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = compute_binary_metrics(
            labels,
            probabilities,
            average_loss=average_loss,
            threshold=threshold,
        )
        rows.append({"threshold": threshold, **metrics.to_dict()})
    return rows


def select_best_threshold(
    labels: list[int],
    probabilities: list[float],
    thresholds: list[float],
    *,
    metric_name: str = "f1",
    average_loss: float | None = None,
) -> dict[str, Any]:
    sweep_rows = sweep_thresholds(
        labels,
        probabilities,
        thresholds,
        average_loss=average_loss,
    )

    if metric_name not in {"accuracy", "precision", "recall", "f1"}:
        raise ValueError("select_best_threshold only supports threshold-dependent metrics.")

    def row_key(row: dict[str, Any]) -> tuple[float, float, float]:
        # If two thresholds tie on the main metric, prefer better recall and then
        # the threshold closest to the usual default of 0.5.
        metric_value = float(row[metric_name])
        recall = float(row["recall"])
        closeness_to_default = -abs(float(row["threshold"]) - 0.5)
        return metric_value, recall, closeness_to_default

    return max(sweep_rows, key=row_key)


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    criterion: torch.nn.Module | None = None,
    threshold: float = 0.5,
) -> BinaryClassificationMetrics:
    prediction_outputs = predict_model_outputs(
        model,
        data_loader,
        device=device,
        criterion=criterion,
    )

    return compute_binary_metrics(
        prediction_outputs.labels,
        prediction_outputs.probabilities,
        average_loss=prediction_outputs.average_loss,
        threshold=threshold,
    )
