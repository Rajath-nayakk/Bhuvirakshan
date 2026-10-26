"""Siamese U-Net change detection architecture, combined loss, and polygon vectorization.

Designed for multi-temporal optical (Sentinel-2) and SAR (Sentinel-1) stacks.
Features shared-weight encoder feature extraction, multiscale difference/concat fusion,
and a seasonal suppression filter for phenology rejection.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from affine import Affine
from rasterio import features
from shapely.geometry import shape
from torch import nn

from src.data.labeling import CLASS_NAMES


class ConvBlock(nn.Module):
    """Double convolution block with BatchNorm and GELU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SiameseUNet(nn.Module):
    """Siamese U-Net with shared encoder and difference/concatenation feature fusion.

    Inputs:
        x_t1: Before-event tensor of shape (B, C, H, W)
        x_t2: After-event tensor of shape (B, C, H, W)
    Outputs:
        logits: Multiclass segmentation logits of shape (B, num_classes, H, W)
    """

    def __init__(
        self,
        in_channels: int = 10,
        num_classes: int = 4,
        base_channels: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Shared Encoder Stages
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck on fused highest-level representations
        self.bottleneck = ConvBlock(base_channels * 8 * 3, base_channels * 16)

        # Decoder stages (incorporating fused features: diff and cat)
        # Fused channels = enc_channels * 3 (diff + t1 + t2)
        self.up4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_channels * 8 + (base_channels * 8 * 3), base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 4 + (base_channels * 4 * 3), base_channels * 4)

        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 2 + (base_channels * 2 * 3), base_channels * 2)

        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels + (base_channels * 3), base_channels)

        # Final classification head
        self.classifier = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def _fuse(self, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        """Fuse dual temporal features via concatenation and absolute difference."""
        diff = torch.abs(f2 - f1)
        return torch.cat([diff, f1, f2], dim=1)

    def forward(self, x_t1: torch.Tensor, x_t2: torch.Tensor) -> torch.Tensor:
        # Encode T1 (Before)
        e1_t1 = self.enc1(x_t1)
        e2_t1 = self.enc2(self.pool1(e1_t1))
        e3_t1 = self.enc3(self.pool2(e2_t1))
        e4_t1 = self.enc4(self.pool3(e3_t1))

        # Encode T2 (After) using shared weights
        e1_t2 = self.enc1(x_t2)
        e2_t2 = self.enc2(self.pool1(e1_t2))
        e3_t2 = self.enc3(self.pool2(e2_t2))
        e4_t2 = self.enc4(self.pool3(e3_t2))

        # Bottleneck on fused highest-level representations
        bn_in = self.pool4(self._fuse(e4_t1, e4_t2))
        b = self.bottleneck(bn_in)

        # Decoder with multi-scale skip connections
        d4 = self.up4(b)
        fused4 = self._fuse(e4_t1, e4_t2)
        d4 = self.dec4(torch.cat([d4, fused4], dim=1))

        d3 = self.up3(d4)
        fused3 = self._fuse(e3_t1, e3_t2)
        d3 = self.dec3(torch.cat([d3, fused3], dim=1))

        d2 = self.up2(d3)
        fused2 = self._fuse(e2_t1, e2_t2)
        d2 = self.dec2(torch.cat([d2, fused2], dim=1))

        d1 = self.up1(d2)
        fused1 = self._fuse(e1_t1, e1_t2)
        d1 = self.dec1(torch.cat([d1, fused1], dim=1))

        logits = self.classifier(d1)
        return logits


class DiceFocalLoss(nn.Module):
    """Combined Multiclass Dice + Focal Loss for handling severe class imbalance."""

    def __init__(
        self,
        num_classes: int = 4,
        focal_weight: float = 0.5,
        dice_weight: float = 0.5,
        gamma: float = 2.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Args:

        logits: Tensor of shape (B, C, H, W)
        targets: Tensor of shape (B, H, W) with class indices (0 to C-1)
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # 1. Focal Loss component
        log_probs = F.log_softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()

        pt = (probs * targets_one_hot).sum(dim=1)
        log_pt = (log_probs * targets_one_hot).sum(dim=1)
        focal_loss = -((1.0 - pt) ** self.gamma) * log_pt
        focal_loss = focal_loss.mean()

        # 2. Multiclass Dice Loss component (excluding background from numerator if desired, or all classes)
        dice_loss = 0.0
        for c in range(num_classes):
            p_c = probs[:, c, :, :]
            t_c = targets_one_hot[:, c, :, :]
            intersection = (p_c * t_c).sum()
            cardinality = p_c.sum() + t_c.sum()
            dice_c = (2.0 * intersection + self.eps) / (cardinality + self.eps)
            dice_loss += (1.0 - dice_c)
        dice_loss = dice_loss / num_classes

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


class SeasonalSuppressionFilter:
    """Filters out recurrent seasonal phenology (dry-season leaf loss, harvest cycles).

    If an observed vegetation drop closely matches the historical same-season drop
    (within tolerance), the alert confidence is suppressed.
    """

    def __init__(self, tolerance: float = 0.08):
        self.tolerance = tolerance

    def filter(
        self,
        predicted_classes: np.ndarray,
        confidence_map: np.ndarray,
        current_ndvi_drop: np.ndarray,
        historical_same_season_drop: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply seasonal filter.

        Returns:
            Tuple of (suppressed_classes, suppressed_confidences, suppression_mask)
        """
        filtered_classes = predicted_classes.copy()
        filtered_conf = confidence_map.copy()
        suppression_mask = np.zeros_like(predicted_classes, dtype=bool)

        if historical_same_season_drop is None:
            # When historical baseline is not available, return unfiltered
            return filtered_classes, filtered_conf, suppression_mask

        # Where current drop closely mirrors recurrent annual dry season drop
        seasonal_match = np.abs(current_ndvi_drop - historical_same_season_drop) < self.tolerance
        # Only apply to vegetation clearing classes (Class 2), never sand mining in riverbed (Class 1)
        is_clearing = (predicted_classes == 2)
        to_suppress = seasonal_match & is_clearing

        suppression_mask[to_suppress] = True
        filtered_conf[to_suppress] *= 0.25  # Heavily dampen confidence
        # Downgrade to background if confidence drops below 0.35
        filtered_classes[to_suppress & (filtered_conf < 0.35)] = 0

        return filtered_classes, filtered_conf, suppression_mask


def vectorize_predictions(
    class_mask: np.ndarray,
    confidence_map: np.ndarray,
    transform: Affine,
    crs: str = "EPSG:4326",
    min_area_ha: float = 0.02,
) -> dict[str, Any]:
    """Convert raster change detection masks into GeoJSON polygons with area and confidence.

    Args:
        class_mask: 2D array of class labels (0 to 3).
        confidence_map: 2D array of model probabilities [0, 1].
        transform: Affine geotransform.
        crs: Coordinate system string.
        min_area_ha: Minimum polygon area threshold in hectares.

    Returns:
        GeoJSON FeatureCollection dict.
    """
    features_list = []
    # 1 pixel at 10m resolution = 100 sq meters = 0.01 hectares
    pixel_area_ha = 0.01

    for class_id in [1, 2, 3]:
        binary_mask = (class_mask == class_id).astype(np.uint8)
        if not np.any(binary_mask):
            continue

        shapes_gen = features.shapes(
            binary_mask,
            mask=binary_mask == 1,
            transform=transform,
        )

        for geom, val in shapes_gen:
            if val != 1:
                continue
            poly = shape(geom)
            # Calculate pixel count inside polygon
            poly_mask = features.rasterize(
                [(poly, 1)],
                out_shape=class_mask.shape,
                transform=transform,
                fill=0,
                dtype=np.uint8,
            ) == 1
            pixel_count = int(np.sum(poly_mask))
            area_ha = round(pixel_count * pixel_area_ha, 3)

            if area_ha < min_area_ha:
                continue

            mean_conf = float(np.mean(confidence_map[poly_mask])) if pixel_count > 0 else 0.0

            feature = {
                "type": "Feature",
                "geometry": geom,
                "properties": {
                    "class_id": int(class_id),
                    "class_name": CLASS_NAMES.get(class_id, "Unknown"),
                    "area_ha": area_ha,
                    "confidence": round(mean_conf, 3),
                    "pixel_count": pixel_count,
                },
            }
            features_list.append(feature)

    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": crs}},
        "features": features_list,
    }
