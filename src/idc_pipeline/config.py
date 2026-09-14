from __future__ import annotations

from dataclasses import dataclass
import math


CANONICAL_DATASET_DIRNAME = "IDC_regular_ps50_idx5"
EXPECTED_PATCH_SIZE = (50, 50)


@dataclass(frozen=True)
class SplitConfig:
    # Splits are defined at patient level to prevent patches from the same case
    # appearing in both training and evaluation.
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    trials: int = 40
    search_for_nested_canonical_dir: bool = True

    def ratios(self) -> dict[str, float]:
        return {
            "train": self.train_ratio,
            "val": self.val_ratio,
            "test": self.test_ratio,
        }

    def validate(self) -> None:
        ratios = self.ratios()
        if not math.isclose(sum(ratios.values()), 1.0):
            raise ValueError("Split ratios must sum to 1.0.")

        if self.trials < 1:
            raise ValueError("SplitConfig.trials must be at least 1.")

        for split_name, ratio in ratios.items():
            if ratio <= 0.0:
                raise ValueError(f"Split ratio for {split_name} must be positive.")
