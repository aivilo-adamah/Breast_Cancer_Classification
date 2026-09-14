"""Extract a small, patient-level subset of the IDC zip archive for local demo training.

The full dataset (~555k patches, 3.3GB) is too slow to train on with a CPU-only
laptop setup. This script samples a subset of patients and, within each patient,
a capped number of patches per class, then extracts only those files while
preserving the original ``patient_id/class/file.png`` layout so the existing
``idc_pipeline`` package can process it unmodified.
"""

from __future__ import annotations

import argparse
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path

FILENAME_PATTERN = re.compile(r"^(?P<patient_id>\d+)/(?P<label>[01])/.+_class[01]\.png$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-patients", type=int, default=40)
    parser.add_argument("--max-patches-per-class-per-patient", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip_path) as archive:
        names = [n for n in archive.namelist() if n.endswith(".png")]

        by_patient_class: dict[tuple[str, str], list[str]] = defaultdict(list)
        for name in names:
            match = FILENAME_PATTERN.match(name)
            if not match:
                continue
            by_patient_class[(match.group("patient_id"), match.group("label"))].append(name)

        all_patients = sorted({patient for patient, _ in by_patient_class})
        rng.shuffle(all_patients)
        chosen_patients = sorted(all_patients[: args.num_patients], key=int)

        to_extract: list[str] = []
        for patient_id in chosen_patients:
            for label in ("0", "1"):
                candidates = by_patient_class.get((patient_id, label), [])
                if not candidates:
                    continue
                sample_size = min(len(candidates), args.max_patches_per_class_per_patient)
                to_extract.extend(rng.sample(candidates, sample_size))

        print(f"Selected {len(chosen_patients)} patients, {len(to_extract)} patches to extract.")
        for member in to_extract:
            archive.extract(member, output_dir)

    print(f"Done. Extracted dataset root: {output_dir}")


if __name__ == "__main__":
    main()
