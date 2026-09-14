"""Build the animated t-SNE GIF shown at the top of the Streamlit app's Context tab.

For a dense sample of benign and malignant patches, this script extracts the
trained CNN's penultimate feature vector, projects those features to 2D with
t-SNE (so visually/semantically similar patches end up close together, and
the two classes tend to separate into distinct clouds), then renders an
animated GIF of the patch thumbnails settling from a random layout into that
final position.

This is a one-off asset-generation script, not something the app runs live:
t-SNE itself is too slow to recompute per page load.
"""

from __future__ import annotations

import csv
import io
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, OffsetImage  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from idc_pipeline.inference import extract_features, load_model_for_inference  # noqa: E402

DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "IDC_regular_ps50_idx5"
MANIFEST_CSV = PROJECT_ROOT / "artifacts" / "manifest_with_split.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "demo" / "best_model.pt"
OUTPUT_PATH = PROJECT_ROOT / "models" / "demo" / "tsne" / "tsne_animation.gif"

SAMPLES_PER_CLASS = 500
SEED = 42

# Streamlit re-encodes (and un-animates) any image wider than 2*730=1460px;
# stay safely under that so the GIF is always served untouched.
MAX_GIF_WIDTH_PX = 1300

BENIGN_COLOR = "#4fb0ff"
MALIGNANT_COLOR = "#ff5c5c"
BACKGROUND_COLOR = "#0e1117"


def load_balanced_sample() -> list[dict]:
    with MANIFEST_CSV.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    rng = random.Random(SEED)
    by_label: dict[str, list[dict]] = {"0": [], "1": []}
    for row in rows:
        by_label[row["label"]].append(row)

    sample: list[dict] = []
    for label, label_rows in by_label.items():
        sample.extend(rng.sample(label_rows, min(SAMPLES_PER_CLASS, len(label_rows))))

    rng.shuffle(sample)
    return sample


def ease_out_cubic(t: float) -> float:
    return 1 - (1 - t) ** 3


def build_animation_gif(
    coordinates: np.ndarray,
    images: list[Image.Image],
    labels: list[int],
    output_path: Path,
    *,
    num_frames: int = 24,
    hold_frames: int = 10,
    figure_size_inches: float = 9.0,
    dpi: int = 105,
    thumbnail_size_px: int = 14,
) -> None:
    rng = np.random.default_rng(SEED)
    spread = max(np.ptp(coordinates[:, 0]), np.ptp(coordinates[:, 1]))
    start_positions = rng.uniform(-spread, spread, size=coordinates.shape) + coordinates.mean(axis=0)

    x_min, x_max = coordinates[:, 0].min() - 5, coordinates[:, 0].max() + 5
    y_min, y_max = coordinates[:, 1].min() - 5, coordinates[:, 1].max() + 5

    frames: list[Image.Image] = []
    for frame_index in range(num_frames + hold_frames):
        progress = ease_out_cubic(min(frame_index / num_frames, 1.0))
        current_positions = start_positions + (coordinates - start_positions) * progress

        figure, axis = plt.subplots(figsize=(figure_size_inches, figure_size_inches), dpi=dpi)
        figure.patch.set_facecolor(BACKGROUND_COLOR)
        axis.set_facecolor(BACKGROUND_COLOR)

        for (x, y), image, label in zip(current_positions, images, labels):
            border_color = MALIGNANT_COLOR if label == 1 else BENIGN_COLOR
            thumbnail = image.resize((thumbnail_size_px, thumbnail_size_px), resample=Image.LANCZOS)
            offset_image = OffsetImage(np.asarray(thumbnail), zoom=1.0, alpha=0.3 + 0.7 * progress)
            annotation_box = AnnotationBbox(
                offset_image,
                (x, y),
                frameon=True,
                pad=0.0,
                bboxprops=dict(edgecolor=border_color, linewidth=0.5),
            )
            axis.add_artist(annotation_box)

        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)
        axis.axis("off")
        figure.tight_layout(pad=0)

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", facecolor=BACKGROUND_COLOR, bbox_inches="tight")
        plt.close(figure)
        buffer.seek(0)
        frames.append(Image.open(buffer).convert("RGB"))

    # Streamlit silently re-encodes (and un-animates!) any image wider than its
    # MAXIMUM_CONTENT_WIDTH (1460px). bbox_inches="tight" can push the rendered
    # width past that depending on the point cloud's shape, so every frame is
    # capped here to guarantee the GIF is always served untouched.
    max_frame_width = max(frame.width for frame in frames)
    if max_frame_width > MAX_GIF_WIDTH_PX:
        scale = MAX_GIF_WIDTH_PX / max_frame_width
        frames = [
            frame.resize(
                (round(frame.width * scale), round(frame.height * scale)),
                resample=Image.LANCZOS,
            )
            for frame in frames
        ]

    # Quantizing to a shared adaptive palette keeps the file size reasonable at
    # this resolution without a visible quality loss on these mostly-pink/white
    # histopathology thumbnails.
    quantized_frames = [frame.quantize(colors=96, method=Image.MEDIANCUT) for frame in frames]

    # loop=0 makes the GIF play on repeat indefinitely in the browser.
    quantized_frames[0].save(
        output_path,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=60,
        loop=0,
        optimize=True,
    )


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading model and sampling patches...")
    loaded_model = load_model_for_inference(CHECKPOINT_PATH)
    sample_rows = load_balanced_sample()
    print(f"Sampled {len(sample_rows)} patches ({SAMPLES_PER_CLASS} per class, target).")

    feature_vectors: list[np.ndarray] = []
    images: list[Image.Image] = []
    labels: list[int] = []

    for row in sample_rows:
        image_path = DATASET_DIR / row["relative_path"]
        image = Image.open(image_path).convert("RGB")
        feature_vectors.append(extract_features(image, loaded_model))
        images.append(image)
        labels.append(int(row["label"]))

    print("Running t-SNE...")
    scaled_features = StandardScaler().fit_transform(np.stack(feature_vectors))
    coordinates = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=SEED,
        max_iter=1000,
    ).fit_transform(scaled_features)
    # Rescale to a fixed, readable coordinate range.
    coordinates = 50.0 * (coordinates - coordinates.mean(axis=0)) / coordinates.std()

    print("Rendering animated GIF (this takes a bit)...")
    build_animation_gif(coordinates, images, labels, OUTPUT_PATH)

    print(f"Done. GIF written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
