"""Single-image inference helpers, shared by the Streamlit app and any future API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .train import load_checkpoint_model


@dataclass(frozen=True)
class PatchPrediction:
    probability: float
    predicted_label: int
    threshold: float

    @property
    def confidence(self) -> float:
        # Distance from the decision boundary, rescaled to a 0-1 "confidence" score.
        return abs(self.probability - 0.5) * 2.0


@dataclass(frozen=True)
class LoadedModel:
    model: torch.nn.Module
    device: torch.device
    architecture: str
    image_size: tuple[int, int] | None
    normalization_mean: tuple[float, float, float] | None
    normalization_std: tuple[float, float, float] | None
    default_threshold: float
    checkpoint_metadata: dict


def load_model_for_inference(checkpoint_path: str | Path) -> LoadedModel:
    model, checkpoint, device = load_checkpoint_model(checkpoint_path)
    config = checkpoint.get("config", {})

    image_size = config.get("image_size")
    if image_size is not None:
        image_size = tuple(image_size)

    normalization_mean = config.get("normalization_mean")
    normalization_std = config.get("normalization_std")
    if normalization_mean is not None:
        normalization_mean = tuple(normalization_mean)
    if normalization_std is not None:
        normalization_std = tuple(normalization_std)

    return LoadedModel(
        model=model,
        device=device,
        architecture=str(config.get("architecture", "baseline_cnn")),
        image_size=image_size,
        normalization_mean=normalization_mean,
        normalization_std=normalization_std,
        default_threshold=float(config.get("threshold", 0.5)),
        checkpoint_metadata={
            "best_epoch": checkpoint.get("epoch"),
            "best_metric_name": checkpoint.get("best_metric_name"),
            "best_metric_value": checkpoint.get("best_metric_value"),
            "pos_weight": checkpoint.get("pos_weight"),
        },
    )


def preprocess_image(image: Image.Image, loaded_model: LoadedModel) -> torch.Tensor:
    rgb_image = image.convert("RGB")

    if loaded_model.image_size is not None and rgb_image.size != loaded_model.image_size:
        rgb_image = rgb_image.resize(loaded_model.image_size, resample=Image.BILINEAR)

    array = np.asarray(rgb_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)

    if loaded_model.normalization_mean is not None and loaded_model.normalization_std is not None:
        mean = torch.tensor(loaded_model.normalization_mean, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(loaded_model.normalization_std, dtype=torch.float32).view(3, 1, 1)
        tensor = (tensor - mean) / std

    return tensor.unsqueeze(0)


@torch.no_grad()
def extract_features(image: Image.Image, loaded_model: LoadedModel) -> np.ndarray:
    """Return the model's penultimate (pre-classifier) feature vector for one patch.

    Used for the t-SNE visualization: these learned features cluster benign vs.
    malignant patches far better than raw pixels.
    """
    input_tensor = preprocess_image(image, loaded_model).to(loaded_model.device)

    backbone = getattr(loaded_model.model, "features", None)
    if backbone is not None:
        feature_map = backbone(input_tensor)
        features = torch.flatten(feature_map, start_dim=1)
    else:
        # Fallback for architectures without a dedicated `.features` submodule
        # (e.g. ResNet18): use the full model's logit as a 1-D feature.
        features = loaded_model.model(input_tensor).unsqueeze(1)

    return features.squeeze(0).cpu().numpy()


@torch.no_grad()
def predict_patch(
    image: Image.Image,
    loaded_model: LoadedModel,
    *,
    threshold: float | None = None,
) -> PatchPrediction:
    effective_threshold = threshold if threshold is not None else loaded_model.default_threshold

    input_tensor = preprocess_image(image, loaded_model).to(loaded_model.device)
    logits = loaded_model.model(input_tensor)
    probability = float(torch.sigmoid(logits).item())
    predicted_label = 1 if probability >= effective_threshold else 0

    return PatchPrediction(
        probability=probability,
        predicted_label=predicted_label,
        threshold=effective_threshold,
    )
