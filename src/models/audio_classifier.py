"""Audio classification neural network for edge threat acoustics.

Classifies log-mel spectrograms into:
- 0: gunshot
- 1: chainsaw
- 2: vehicle
- 3: ambient_forest

Supports ONNX export and low-power Raspberry Pi edge deployment.
"""

import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


class AudioConvBlock(nn.Module):
    """Convolution block with BatchNorm, GELU, and MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(x))


class AudioClassifier(nn.Module):
    """Lightweight 2D CNN for acoustic spectrogram classification.

    Input shape: (B, 1, 128, 157)
    Output logits: (B, 4)
    """

    def __init__(self, num_classes: int = 4, in_channels: int = 1, base_channels: int = 24):
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            AudioConvBlock(in_channels, base_channels),             # -> (B, 24, 64, 78)
            AudioConvBlock(base_channels, base_channels * 2),       # -> (B, 48, 32, 39)
            AudioConvBlock(base_channels * 2, base_channels * 4),   # -> (B, 96, 16, 19)
            AudioConvBlock(base_channels * 4, base_channels * 8),   # -> (B, 192, 8, 9)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(base_channels * 8, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        pooled = self.global_pool(feats)
        logits = self.classifier(pooled)
        return logits


def export_audio_model_to_onnx(
    model: nn.Module,
    output_path: Path | str = "data/checkpoints/audio_classifier.onnx",
    input_shape: tuple[int, int, int, int] = (1, 1, 128, 157),
) -> Path:
    """Export trained PyTorch audio model to ONNX for edge runtime on Raspberry Pi."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(*input_shape)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,
            input_names=["spectrogram"],
            output_names=["class_logits"],
            dynamic_axes={"spectrogram": {0: "batch_size"}, "class_logits": {0: "batch_size"}},
        )
    except Exception as e:
        logger.warning("Standard ONNX export encountered (%s), falling back to torchscript.", e)
        script_model = torch.jit.trace(model, dummy_input)
        script_model.save(str(output_path.with_suffix(".pt")))
    return output_path
