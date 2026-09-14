"""Train the small demo model bundled with the Streamlit app.

Uses the patient-level subset extracted by ``extract_demo_subset.py`` and the
manifest/splits produced by ``idc_pipeline.data``. Kept intentionally small
(few epochs, compact CNN) so it trains in a reasonable time on CPU while still
producing a real, evaluated checkpoint -- not a mock.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from idc_pipeline.dataset import compute_dataset_normalization
from idc_pipeline.evaluate import (
    compute_average_precision,
    compute_binary_precision_recall_curve,
    compute_binary_roc_curve,
    predict_model_outputs,
    select_best_threshold,
)
from idc_pipeline.train import (
    TrainingConfig,
    build_dataloaders,
    load_checkpoint_model,
    resolve_device,
    run_training_experiment,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "IDC_regular_ps50_idx5"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TRAIN_CSV = ARTIFACTS_DIR / "splits" / "train.csv"
VAL_CSV = ARTIFACTS_DIR / "splits" / "val.csv"
TEST_CSV = ARTIFACTS_DIR / "splits" / "test.csv"
TRAINING_RUN_DIR = PROJECT_ROOT / "training_runs" / "demo_cnn"
DEMO_MODEL_DIR = PROJECT_ROOT / "models" / "demo"


def write_curve_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    normalization = compute_dataset_normalization(TRAIN_CSV, DATASET_DIR, max_samples=4000)
    print(f"Train-split normalization: mean={normalization.mean}, std={normalization.std}")

    config = TrainingConfig(
        batch_size=64,
        epochs=12,
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_workers=0,
        seed=42,
        early_stopping_patience=4,
        threshold=0.5,
        dropout=0.30,
        architecture="baseline_cnn",
        normalization_mean=normalization.mean,
        normalization_std=normalization.std,
        color_jitter_strength=0.10,
        scheduler_name="reduce_on_plateau",
        selection_metric="f1",
        pin_memory=False,
    )

    result = run_training_experiment(
        dataset_dir=DATASET_DIR,
        train_csv_path=TRAIN_CSV,
        val_csv_path=VAL_CSV,
        test_csv_path=TEST_CSV,
        output_dir=TRAINING_RUN_DIR,
        config=config,
    )
    print(json.dumps(result.to_dict(), indent=2))

    # Sweep thresholds on validation, matching the notebook 6 evaluation protocol.
    _, _, _, _, val_loader, test_loader = build_dataloaders(
        dataset_dir=DATASET_DIR,
        train_csv_path=TRAIN_CSV,
        val_csv_path=VAL_CSV,
        test_csv_path=TEST_CSV,
        config=config,
    )
    device = resolve_device()
    model, _, _ = load_checkpoint_model(result.checkpoint_path, device=device)

    val_outputs = predict_model_outputs(model, val_loader, device=device)
    test_outputs = predict_model_outputs(model, test_loader, device=device)

    thresholds = [round(0.10 + 0.05 * step, 2) for step in range(17)]
    best_threshold_row = select_best_threshold(
        val_outputs.labels, val_outputs.probabilities, thresholds, metric_name="f1"
    )

    evaluation_dir = TRAINING_RUN_DIR / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    write_curve_csv(
        evaluation_dir / "test_roc_curve.csv",
        compute_binary_roc_curve(test_outputs.labels, test_outputs.probabilities),
    )
    write_curve_csv(
        evaluation_dir / "test_precision_recall_curve.csv",
        compute_binary_precision_recall_curve(test_outputs.labels, test_outputs.probabilities),
    )

    evaluation_summary = {
        "best_val_threshold": best_threshold_row,
        "test_average_precision": compute_average_precision(
            test_outputs.labels, test_outputs.probabilities
        ),
    }
    (evaluation_dir / "evaluation_summary.json").write_text(
        json.dumps(evaluation_summary, indent=2) + "\n"
    )
    print(json.dumps(evaluation_summary, indent=2))

    # Bundle the small artifacts the Streamlit app needs into models/demo/.
    DEMO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(result.checkpoint_path, DEMO_MODEL_DIR / "best_model.pt")
    shutil.copy2(result.metrics_path, DEMO_MODEL_DIR / "metrics.json")
    demo_evaluation_dir = DEMO_MODEL_DIR / "evaluation"
    if demo_evaluation_dir.exists():
        shutil.rmtree(demo_evaluation_dir)
    shutil.copytree(evaluation_dir, demo_evaluation_dir)
    print(f"Copied demo artifacts to {DEMO_MODEL_DIR}")


if __name__ == "__main__":
    main()
