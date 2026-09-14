from __future__ import annotations

import torch
from torch import nn

IMAGENET_NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
IMAGENET_NORMALIZATION_STD = (0.229, 0.224, 0.225)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class BaselineIDCClassifier(nn.Module):
    def __init__(self, dropout: float = 0.30) -> None:
        super().__init__()
        # The baseline stays compact because the original patches are only 50x50.
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        logits = self.classifier(features)
        return logits.squeeze(1)


class ResNet18BinaryClassifier(nn.Module):
    def __init__(self, *, dropout: float = 0.20, pretrained: bool = False) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise ImportError("torchvision is required for ResNet18 support.") from exc

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        in_features = backbone.fc.in_features
        # Replace the ImageNet head with a single-logit binary IDC classifier.
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
        )
        self.backbone = backbone

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(inputs)
        return logits.squeeze(1)


def build_model(
    *,
    architecture: str = "baseline_cnn",
    dropout: float = 0.30,
    pretrained: bool = False,
) -> nn.Module:
    if architecture == "baseline_cnn":
        return BaselineIDCClassifier(dropout=dropout)
    if architecture == "resnet18":
        return ResNet18BinaryClassifier(dropout=dropout, pretrained=pretrained)
    raise ValueError(f"Unsupported architecture: {architecture}")
