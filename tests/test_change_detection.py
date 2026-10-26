"""Unit tests for Phase 3: Siamese U-Net Change Detection Model & Vectorization."""

import numpy as np
import pytest
import torch
from affine import Affine

from src.infer.run_change import run_ndvi_baseline_detector
from src.models.change_detect import (
    DiceFocalLoss,
    SeasonalSuppressionFilter,
    SiameseUNet,
    vectorize_predictions,
)


def test_siamese_unet_forward_pass():
    """Verify Siamese U-Net model accepts dual temporal inputs and outputs multiclass logits."""
    model = SiameseUNet(in_channels=10, num_classes=4, base_channels=16)
    model.eval()

    batch_size = 2
    t1 = torch.randn(batch_size, 10, 64, 64)
    t2 = torch.randn(batch_size, 10, 64, 64)

    with torch.no_grad():
        logits = model(t1, t2)

    assert logits.shape == (batch_size, 4, 64, 64)


def test_dice_focal_loss_backward():
    """Verify combined Dice + Focal loss calculates scalar loss and flows gradients back."""
    loss_fn = DiceFocalLoss(num_classes=4, focal_weight=0.5, dice_weight=0.5)

    logits = torch.randn(2, 4, 32, 32, requires_grad=True)
    targets = torch.randint(0, 4, (2, 32, 32))

    loss = loss_fn(logits, targets)

    assert loss.dim() == 0, "Loss must be scalar"
    assert loss.item() >= 0.0, "Loss must be non-negative"

    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


def test_seasonal_suppression_filter():
    """Verify seasonal filter dampens confidence when current vegetation loss matches annual baseline."""
    filter_obj = SeasonalSuppressionFilter(tolerance=0.06)

    pred_classes = np.array([[2, 1], [0, 2]], dtype=np.uint8)
    conf_map = np.array([[0.85, 0.90], [0.95, 0.80]], dtype=np.float32)

    # Pixel (0,0) is Class 2 (clearing) where current drop (0.15) matches recurrent seasonal drop (0.14)
    # Pixel (1,1) is Class 2 where current drop (0.45) is anomalous vs seasonal drop (0.10)
    current_drop = np.array([[0.15, 0.02], [0.00, 0.45]], dtype=np.float32)
    hist_seasonal_drop = np.array([[0.14, 0.01], [0.00, 0.10]], dtype=np.float32)

    filtered_classes, filtered_conf, suppressed = filter_obj.filter(
        pred_classes, conf_map, current_drop, hist_seasonal_drop
    )

    # Pixel (0, 0) should be suppressed
    assert suppressed[0, 0] is True or suppressed[0, 0] == 1
    assert filtered_conf[0, 0] < conf_map[0, 0]

    # Pixel (1, 1) should NOT be suppressed (it is an anomalous clearcut)
    assert suppressed[1, 1] is False or suppressed[1, 1] == 0
    assert filtered_conf[1, 1] == conf_map[1, 1]
    assert filtered_classes[1, 1] == 2


def test_vectorize_predictions():
    """Verify prediction raster is converted into valid GeoJSON FeatureCollection."""
    class_mask = np.zeros((100, 100), dtype=np.uint8)
    confidence_map = np.full((100, 100), 0.88, dtype=np.float32)

    # Create a 20x20 pixel sand mining disturbance (400 pixels = 4 hectares)
    class_mask[10:30, 10:30] = 1
    # Create a 15x15 pixel forest clearing disturbance (225 pixels = 2.25 hectares)
    class_mask[50:65, 50:65] = 2

    transform = Affine.translation(77.20, 12.10) @ Affine.scale(0.0001, -0.0001)

    geojson = vectorize_predictions(class_mask, confidence_map, transform, min_area_ha=0.1)

    assert "type" in geojson
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 2

    features_by_class = {f["properties"]["class_id"]: f["properties"] for f in geojson["features"]}
    assert 1 in features_by_class
    assert 2 in features_by_class
    assert features_by_class[1]["class_name"] == "Sand Mining / Riverbed Extraction"
    assert features_by_class[1]["area_ha"] > 0.5
    assert features_by_class[1]["confidence"] == pytest.approx(0.88, abs=0.01)


def test_ndvi_baseline_detector():
    """Verify simple NDVI baseline detector runs and flags changes."""
    before = np.zeros((10, 32, 32), dtype=np.float32)
    after = np.zeros((10, 32, 32), dtype=np.float32)

    before[6] = 0.70   # High forest NDVI
    before[7] = -0.50  # Forest NDWI is negative
    after[6] = 0.10    # Major drop
    after[2] = 0.25    # Bare soil jump

    pred = run_ndvi_baseline_detector(before, after)
    assert pred.shape == (32, 32)
    assert np.all(pred == 2)  # Should detect forest clearing
