from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
from copy import deepcopy
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import struct
from typing import Any

from .config import CANONICAL_DATASET_DIRNAME, EXPECTED_PATCH_SIZE, SplitConfig


FILENAME_PATTERN = re.compile(
    r"^(?P<patient_id>\d+)_idx5_x(?P<x>\d+)_y(?P<y>\d+)_class(?P<label>[01])\.png$"
)

MANIFEST_FIELDS = ["relative_path", "patient_id", "x_coord", "y_coord", "label"]
INVALID_SAMPLE_FIELDS = ["relative_path", "reason", "details"]
SPLIT_FIELDS = [*MANIFEST_FIELDS, "split"]
PATIENT_SUMMARY_FIELDS = [
    "patient_id",
    "total_patches",
    "positive_patches",
    "negative_patches",
    "positive_rate",
]
SPLIT_SUMMARY_FIELDS = [
    "split",
    "patients",
    "patches",
    "positive_patches",
    "negative_patches",
    "positive_rate",
]


@dataclass(frozen=True)
class ManifestCollectionResult:
    patient_dirs: list[Path]
    records: list[dict[str, Any]]
    invalid_samples: list[dict[str, Any]]
    scanned_image_count: int
    dropped_image_count: int
    dropped_reason_counts: dict[str, int]


@dataclass(frozen=True)
class PipelineResult:
    dataset_dir: Path
    output_dir: Path
    manifest_path: Path
    patient_summary_path: Path
    manifest_with_split_path: Path
    split_summary_path: Path
    cleaning_summary_path: Path
    invalid_samples_path: Path
    split_paths: dict[str, Path]
    split_summary_rows: list[dict[str, Any]]
    scanned_patient_count: int
    patient_count: int
    scanned_image_count: int
    image_count: int
    dropped_image_count: int
    dropped_reason_counts: dict[str, int]
    split_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "output_dir": str(self.output_dir),
            "expected_patch_size": {
                "width": EXPECTED_PATCH_SIZE[0],
                "height": EXPECTED_PATCH_SIZE[1],
            },
            "scanned_patient_count": self.scanned_patient_count,
            "patient_count": self.patient_count,
            "scanned_image_count": self.scanned_image_count,
            "image_count": self.image_count,
            "dropped_image_count": self.dropped_image_count,
            "dropped_reason_counts": self.dropped_reason_counts,
            "split_score": round(self.split_score, 6),
            "artifacts": {
                "manifest": str(self.manifest_path),
                "patient_summary": str(self.patient_summary_path),
                "manifest_with_split": str(self.manifest_with_split_path),
                "split_summary": str(self.split_summary_path),
                "cleaning_summary": str(self.cleaning_summary_path),
                "invalid_samples": str(self.invalid_samples_path),
                **{f"{name}_split": str(path) for name, path in self.split_paths.items()},
            },
            "split_summary": self.split_summary_rows,
        }


def resolve_dataset_dir(dataset_dir: str | Path, split_config: SplitConfig) -> Path:
    resolved = Path(dataset_dir).expanduser()
    nested_candidate = resolved / CANONICAL_DATASET_DIRNAME

    if split_config.search_for_nested_canonical_dir and nested_candidate.exists():
        return nested_candidate

    return resolved


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def read_png_size(image_path: Path) -> tuple[int, int]:
    # Reading only the PNG header keeps manifest generation fast while still
    # verifying that each patch is a valid 50x50 image.
    try:
        with image_path.open("rb") as handle:
            signature = handle.read(8)
            if signature != b"\x89PNG\r\n\x1a\n":
                raise ValueError("Invalid PNG signature.")

            chunk_length = handle.read(4)
            chunk_type = handle.read(4)
            if len(chunk_length) != 4 or len(chunk_type) != 4:
                raise ValueError("Truncated PNG header.")

            length = struct.unpack(">I", chunk_length)[0]
            if chunk_type != b"IHDR":
                raise ValueError("IHDR chunk not found at PNG start.")

            ihdr_data = handle.read(length)
            if len(ihdr_data) != length or length < 8:
                raise ValueError("Truncated IHDR chunk.")

            width, height = struct.unpack(">II", ihdr_data[:8])
            return width, height
    except OSError as exc:
        raise ValueError(f"Could not open PNG file: {exc}") from exc


def collect_manifest_records(dataset_dir: Path) -> ManifestCollectionResult:
    patient_dirs = sorted(
        [path for path in dataset_dir.iterdir() if path.is_dir() and path.name.isdigit()],
        key=lambda path: int(path.name),
    )

    if not patient_dirs:
        raise RuntimeError(
            f"No numeric patient folders were found in {dataset_dir}. "
            "Check that the dataset path points to IDC_regular_ps50_idx5."
        )

    records: list[dict[str, Any]] = []
    invalid_samples: list[dict[str, Any]] = []
    dropped_reason_counts: Counter[str] = Counter()
    scanned_image_count = 0

    def drop_sample(image_path: Path, reason: str, details: str) -> None:
        invalid_samples.append(
            {
                "relative_path": str(image_path.relative_to(dataset_dir)),
                "reason": reason,
                "details": details,
            }
        )
        dropped_reason_counts[reason] += 1

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        class_dirs = [
            path for path in patient_dir.iterdir() if path.is_dir() and path.name in {"0", "1"}
        ]

        for class_dir in sorted(class_dirs, key=lambda path: path.name):
            class_dir_label = int(class_dir.name)

            for image_path in sorted(class_dir.glob("*.png")):
                scanned_image_count += 1

                match = FILENAME_PATTERN.match(image_path.name)
                if match is None:
                    drop_sample(
                        image_path=image_path,
                        reason="invalid_filename",
                        details="Filename does not match the expected IDC pattern.",
                    )
                    continue

                parsed = match.groupdict()
                filename_patient_id = parsed["patient_id"]
                filename_label = int(parsed["label"])

                if filename_patient_id != patient_id:
                    drop_sample(
                        image_path=image_path,
                        reason="patient_mismatch",
                        details=f"folder={patient_id}, filename={filename_patient_id}",
                    )
                    continue

                if filename_label != class_dir_label:
                    drop_sample(
                        image_path=image_path,
                        reason="label_mismatch",
                        details=f"folder={class_dir_label}, filename={filename_label}",
                    )
                    continue

                try:
                    width, height = read_png_size(image_path)
                except ValueError as exc:
                    drop_sample(
                        image_path=image_path,
                        reason="unreadable_png",
                        details=str(exc),
                    )
                    continue

                if (width, height) != EXPECTED_PATCH_SIZE:
                    drop_sample(
                        image_path=image_path,
                        reason="wrong_size",
                        details=(
                            f"expected={EXPECTED_PATCH_SIZE[0]}x{EXPECTED_PATCH_SIZE[1]}, "
                            f"actual={width}x{height}"
                        ),
                    )
                    continue

                records.append(
                    {
                        "relative_path": str(image_path.relative_to(dataset_dir)),
                        "patient_id": patient_id,
                        "x_coord": int(parsed["x"]),
                        "y_coord": int(parsed["y"]),
                        "label": filename_label,
                    }
                )

    if not records:
        raise RuntimeError("No valid PNG records were collected from the dataset.")

    return ManifestCollectionResult(
        patient_dirs=patient_dirs,
        records=records,
        invalid_samples=invalid_samples,
        scanned_image_count=scanned_image_count,
        dropped_image_count=len(invalid_samples),
        dropped_reason_counts=dict(sorted(dropped_reason_counts.items())),
    )


def build_patient_summary(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int], float]:
    patient_rollup: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total_patches": 0, "positive_patches": 0, "negative_patches": 0}
    )

    for record in records:
        patient_bucket = patient_rollup[record["patient_id"]]
        patient_bucket["total_patches"] += 1
        if record["label"] == 1:
            patient_bucket["positive_patches"] += 1
        else:
            patient_bucket["negative_patches"] += 1

    patient_summary: list[dict[str, Any]] = []
    for patient_id in sorted(patient_rollup, key=int):
        stats = patient_rollup[patient_id]
        total_patches = stats["total_patches"]
        patient_summary.append(
            {
                "patient_id": patient_id,
                "total_patches": total_patches,
                "positive_patches": stats["positive_patches"],
                "negative_patches": stats["negative_patches"],
                "positive_rate": round(stats["positive_patches"] / total_patches, 6),
            }
        )

    overall_counts = {
        "patients": len(patient_summary),
        "total_patches": sum(row["total_patches"] for row in patient_summary),
        "positive_patches": sum(row["positive_patches"] for row in patient_summary),
        "negative_patches": sum(row["negative_patches"] for row in patient_summary),
    }
    overall_positive_rate = overall_counts["positive_patches"] / overall_counts["total_patches"]

    return patient_summary, overall_counts, overall_positive_rate


def build_patient_split(
    patient_summary: list[dict[str, Any]],
    overall_counts: dict[str, int],
    overall_positive_rate: float,
    split_config: SplitConfig,
) -> tuple[dict[str, str], dict[str, dict[str, int]], float]:
    ratios = split_config.ratios()

    def empty_state() -> dict[str, dict[str, int]]:
        return {
            split_name: {
                "patients": 0,
                "total_patches": 0,
                "positive_patches": 0,
                "negative_patches": 0,
            }
            for split_name in ratios
        }

    def apply_patient(
        state: dict[str, dict[str, int]],
        split_name: str,
        patient_row: dict[str, Any],
        direction: int,
    ) -> None:
        state[split_name]["patients"] += direction
        state[split_name]["total_patches"] += direction * patient_row["total_patches"]
        state[split_name]["positive_patches"] += direction * patient_row["positive_patches"]
        state[split_name]["negative_patches"] += direction * patient_row["negative_patches"]

    def score_state(state: dict[str, dict[str, int]]) -> float:
        weights = {
            "patients": 1.0,
            "total_patches": 3.0,
            "positive_patches": 3.0,
            "negative_patches": 3.0,
        }
        score = 0.0

        for split_name, ratio in ratios.items():
            for metric, weight in weights.items():
                denominator = overall_counts[metric]
                if denominator == 0:
                    continue

                fraction = state[split_name][metric] / denominator
                difference = fraction - ratio
                score += weight * difference * difference

                if difference > 0:
                    score += weight * 0.35 * difference * difference

        return score

    best_assignments: dict[str, str] | None = None
    best_state: dict[str, dict[str, int]] | None = None
    best_score = float("inf")

    # This is a small greedy search rather than an exhaustive optimizer. The goal
    # is a reproducible patient-level split with similar patch and class ratios.
    for trial in range(split_config.trials):
        trial_rows = [dict(row) for row in patient_summary]
        randomizer = random.Random(split_config.seed + trial)
        randomizer.shuffle(trial_rows)
        trial_rows.sort(
            key=lambda row: (
                row["total_patches"],
                abs(row["positive_rate"] - overall_positive_rate),
                row["positive_patches"],
            ),
            reverse=True,
        )

        state = empty_state()
        assignments: dict[str, str] = {}

        for patient_row in trial_rows:
            preferred_split: str | None = None
            preferred_score: float | None = None
            preferred_tiebreak: float | None = None

            for split_name in ratios:
                apply_patient(state, split_name, patient_row, +1)
                candidate_score = score_state(state)
                total_fraction = state[split_name]["total_patches"] / overall_counts["total_patches"]
                apply_patient(state, split_name, patient_row, -1)

                tiebreak = abs(total_fraction - ratios[split_name])
                if (
                    preferred_score is None
                    or candidate_score < preferred_score
                    or (candidate_score == preferred_score and tiebreak < preferred_tiebreak)
                ):
                    preferred_split = split_name
                    preferred_score = candidate_score
                    preferred_tiebreak = tiebreak

            assignments[patient_row["patient_id"]] = preferred_split
            apply_patient(state, preferred_split, patient_row, +1)

        improved = True
        while improved:
            improved = False
            for patient_row in trial_rows:
                current_split = assignments[patient_row["patient_id"]]
                current_score = score_state(state)
                preferred_split = current_split
                preferred_score = current_score

                for split_name in ratios:
                    if split_name == current_split:
                        continue

                    apply_patient(state, current_split, patient_row, -1)
                    apply_patient(state, split_name, patient_row, +1)
                    candidate_score = score_state(state)
                    apply_patient(state, split_name, patient_row, -1)
                    apply_patient(state, current_split, patient_row, +1)

                    if candidate_score < preferred_score:
                        preferred_split = split_name
                        preferred_score = candidate_score

                if preferred_split != current_split:
                    apply_patient(state, current_split, patient_row, -1)
                    apply_patient(state, preferred_split, patient_row, +1)
                    assignments[patient_row["patient_id"]] = preferred_split
                    improved = True

        final_score = score_state(state)
        if final_score < best_score:
            best_assignments = assignments.copy()
            best_state = deepcopy(state)
            best_score = final_score

    return best_assignments, best_state, best_score


def attach_splits_to_manifest(
    records: list[dict[str, Any]], assignments: dict[str, str]
) -> list[dict[str, Any]]:
    manifest_with_split = []
    for record in records:
        manifest_with_split.append({**record, "split": assignments[record["patient_id"]]})
    return manifest_with_split


def build_split_summary(
    manifest_with_split: list[dict[str, Any]], ratios: dict[str, float]
) -> list[dict[str, Any]]:
    split_patients: dict[str, set[str]] = defaultdict(set)
    split_counts = {
        split_name: {
            "patients": 0,
            "patches": 0,
            "positive_patches": 0,
            "negative_patches": 0,
        }
        for split_name in ratios
    }

    for row in manifest_with_split:
        split_name = row["split"]
        split_patients[split_name].add(row["patient_id"])
        split_counts[split_name]["patches"] += 1
        if row["label"] == 1:
            split_counts[split_name]["positive_patches"] += 1
        else:
            split_counts[split_name]["negative_patches"] += 1

    split_names = list(ratios)
    for split_name in split_names:
        split_counts[split_name]["patients"] = len(split_patients[split_name])

    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            if not split_patients[left_name].isdisjoint(split_patients[right_name]):
                raise AssertionError(
                    f"Leakage detected between {left_name} and {right_name} patients."
                )

    split_summary_rows = []
    for split_name in split_names:
        stats = split_counts[split_name]
        split_summary_rows.append(
            {
                "split": split_name,
                "patients": stats["patients"],
                "patches": stats["patches"],
                "positive_patches": stats["positive_patches"],
                "negative_patches": stats["negative_patches"],
                "positive_rate": round(stats["positive_patches"] / stats["patches"], 6),
            }
        )

    return split_summary_rows


def build_cleaning_summary(
    *,
    dataset_dir: Path,
    scanned_patient_count: int,
    kept_patient_count: int,
    scanned_image_count: int,
    kept_image_count: int,
    dropped_image_count: int,
    dropped_reason_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "dataset_dir": str(dataset_dir),
        "expected_patch_size": {
            "width": EXPECTED_PATCH_SIZE[0],
            "height": EXPECTED_PATCH_SIZE[1],
        },
        "scanned_patient_count": scanned_patient_count,
        "kept_patient_count": kept_patient_count,
        "scanned_image_count": scanned_image_count,
        "kept_image_count": kept_image_count,
        "dropped_image_count": dropped_image_count,
        "drop_fraction": round(dropped_image_count / scanned_image_count, 8),
        "dropped_reason_counts": dropped_reason_counts,
    }


def run_manifest_pipeline(
    dataset_dir: str | Path, output_dir: str | Path, split_config: SplitConfig | None = None
) -> PipelineResult:
    split_config = split_config or SplitConfig()
    split_config.validate()

    resolved_dataset_dir = resolve_dataset_dir(dataset_dir, split_config)
    resolved_output_dir = Path(output_dir).expanduser()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    if not resolved_dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {resolved_dataset_dir}\n"
            "Point the dataset path to the canonical IDC_regular_ps50_idx5 folder."
        )

    collection_result = collect_manifest_records(resolved_dataset_dir)
    manifest_path = resolved_output_dir / "manifest.csv"
    write_csv(manifest_path, MANIFEST_FIELDS, collection_result.records)

    patient_summary, overall_counts, overall_positive_rate = build_patient_summary(
        collection_result.records
    )
    patient_summary_path = resolved_output_dir / "patient_summary.csv"
    write_csv(patient_summary_path, PATIENT_SUMMARY_FIELDS, patient_summary)

    invalid_samples_path = resolved_output_dir / "invalid_samples.csv"
    write_csv(invalid_samples_path, INVALID_SAMPLE_FIELDS, collection_result.invalid_samples)

    cleaning_summary = build_cleaning_summary(
        dataset_dir=resolved_dataset_dir,
        scanned_patient_count=len(collection_result.patient_dirs),
        kept_patient_count=len(patient_summary),
        scanned_image_count=collection_result.scanned_image_count,
        kept_image_count=len(collection_result.records),
        dropped_image_count=collection_result.dropped_image_count,
        dropped_reason_counts=collection_result.dropped_reason_counts,
    )
    cleaning_summary_path = resolved_output_dir / "cleaning_summary.json"
    write_json(cleaning_summary_path, cleaning_summary)

    # Splitting only after cleaning guarantees that every downstream artifact
    # references the same validated subset of image patches.
    assignments, _, split_score = build_patient_split(
        patient_summary=patient_summary,
        overall_counts=overall_counts,
        overall_positive_rate=overall_positive_rate,
        split_config=split_config,
    )

    manifest_with_split = attach_splits_to_manifest(collection_result.records, assignments)
    manifest_with_split_path = resolved_output_dir / "manifest_with_split.csv"
    write_csv(manifest_with_split_path, SPLIT_FIELDS, manifest_with_split)

    split_paths: dict[str, Path] = {}
    splits_dir = resolved_output_dir / "splits"
    for split_name in split_config.ratios():
        split_rows = [row for row in manifest_with_split if row["split"] == split_name]
        split_path = splits_dir / f"{split_name}.csv"
        write_csv(split_path, SPLIT_FIELDS, split_rows)
        split_paths[split_name] = split_path

    split_summary_rows = build_split_summary(manifest_with_split, split_config.ratios())
    split_summary_path = resolved_output_dir / "split_summary.csv"
    write_csv(split_summary_path, SPLIT_SUMMARY_FIELDS, split_summary_rows)

    return PipelineResult(
        dataset_dir=resolved_dataset_dir,
        output_dir=resolved_output_dir,
        manifest_path=manifest_path,
        patient_summary_path=patient_summary_path,
        manifest_with_split_path=manifest_with_split_path,
        split_summary_path=split_summary_path,
        cleaning_summary_path=cleaning_summary_path,
        invalid_samples_path=invalid_samples_path,
        split_paths=split_paths,
        split_summary_rows=split_summary_rows,
        scanned_patient_count=len(collection_result.patient_dirs),
        patient_count=len(patient_summary),
        scanned_image_count=collection_result.scanned_image_count,
        image_count=len(collection_result.records),
        dropped_image_count=collection_result.dropped_image_count,
        dropped_reason_counts=collection_result.dropped_reason_counts,
        split_score=split_score,
    )


def build_argument_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build the IDC manifest and patient-level splits.")
    parser.add_argument("--dataset-dir", required=True, help="Path to IDC_regular_ps50_idx5.")
    parser.add_argument("--output-dir", required=True, help="Path for generated CSV artifacts.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split search.")
    parser.add_argument(
        "--trials",
        type=int,
        default=40,
        help="Number of greedy split trials to search before choosing the best split.",
    )
    parser.add_argument(
        "--disable-nested-search",
        action="store_true",
        help="Do not auto-resolve a nested IDC_regular_ps50_idx5 directory.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    result = run_manifest_pipeline(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        split_config=SplitConfig(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            trials=args.trials,
            search_for_nested_canonical_dir=not args.disable_nested_search,
        ),
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
