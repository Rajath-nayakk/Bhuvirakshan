"""Inference, baseline comparison, and GeoJSON polygon generation for satellite change detection.

Evaluates Deep Siamese U-Net vs simple NDVI difference threshold baseline,
applies seasonal suppression, and outputs GeoJSON change alerts.
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from sklearn.metrics import confusion_matrix

from src.data.tiling import read_geotiff
from src.models.change_detect import (
    SeasonalSuppressionFilter,
    SiameseUNet,
    vectorize_predictions,
)
from src.train.train_change import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_ndvi_baseline_detector(
    before_stack: np.ndarray,
    after_stack: np.ndarray,
    ndvi_drop_thresh: float = 0.20,
    bare_soil_thresh: float = 0.22,
) -> np.ndarray:
    """Simple baseline detector using direct spectral differencing without deep learning."""
    before_ndvi = before_stack[6]
    after_ndvi = after_stack[6]
    after_b4 = after_stack[2]
    before_ndwi = before_stack[7]

    h, w = before_ndvi.shape
    baseline_pred = np.zeros((h, w), dtype=np.uint8)

    # Simple NDVI difference
    ndvi_drop = before_ndvi - after_ndvi

    # Rule: Clearing if NDVI drops significantly from high forest
    clearing = (before_ndvi > 0.35) & (ndvi_drop > ndvi_drop_thresh)
    baseline_pred[clearing] = 2

    # Rule: Sand mining if in river corridor and reflectance jumps (exclusive of clearing)
    riverbed = ((before_ndwi > -0.1) | (before_ndvi < 0.25)) & ~clearing
    sand_mining = riverbed & (after_b4 > bare_soil_thresh)
    baseline_pred[sand_mining] = 1

    return baseline_pred


def evaluate_test_set_and_export(
    checkpoint_path: str = "data/checkpoints/siamese_unet_best.pt",
    splits_path: str = "data/labels/spatial_splits.json",
    reports_dir: str = "reports",
    output_geojson_dir: str = "data/processed/change_polygons",
) -> dict[str, Any]:
    """Run full test evaluation, baseline comparison, and polygon generation."""
    reports_dir_p = Path(reports_dir)
    metrics_dir = reports_dir_p / "metrics"
    figures_dir = reports_dir_p / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    out_geo_dir = Path(output_geojson_dir)
    out_geo_dir.mkdir(parents=True, exist_ok=True)

    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)

    test_records = splits.get("test", [])
    if not test_records:
        logger.warning("No test records in splits file. Using validation records for evaluation.")
        test_records = splits.get("val", [])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseUNet(in_channels=10, num_classes=4, base_channels=16).to(device)

    chk_p = Path(checkpoint_path)
    if chk_p.exists():
        ckpt = torch.load(chk_p, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded trained checkpoint from %s (Val IoU: %.4f)", chk_p, ckpt.get("val_iou", 0.0))
    else:
        logger.warning("Checkpoint %s not found. Using initialized Siamese model.", chk_p)

    model.eval()
    seasonal_filter = SeasonalSuppressionFilter(tolerance=0.08)

    all_targets = []
    all_deep_preds = []
    all_baseline_preds = []
    all_geojson_polygons = []

    for rec in test_records:
        tile_id = rec["tile_id"]
        before_arr, before_prof = read_geotiff(rec["before_path"])
        after_arr, _ = read_geotiff(rec["after_path"])
        with rasterio.open(rec["mask_path"]) as src:
            target_mask = src.read(1)

        t1 = torch.from_numpy(before_arr.astype(np.float32)).unsqueeze(0).to(device)
        t2 = torch.from_numpy(after_arr.astype(np.float32)).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(t1, t2)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            raw_pred = np.argmax(probs, axis=0).astype(np.uint8)
            conf_map = np.max(probs, axis=0)

        # Baseline prediction
        base_pred = run_ndvi_baseline_detector(before_arr, after_arr)

        # Apply Seasonal Suppression Filter
        curr_ndvi_drop = before_arr[6] - after_arr[6]
        # Simulate dry season phenology reference
        simulated_hist_drop = np.full_like(curr_ndvi_drop, 0.05)
        filtered_pred, filtered_conf, _ = seasonal_filter.filter(
            raw_pred, conf_map, curr_ndvi_drop, simulated_hist_drop
        )

        # Vectorize into GeoJSON
        geojson = vectorize_predictions(
            filtered_pred,
            filtered_conf,
            before_prof["transform"],
            crs=str(before_prof["crs"]),
            min_area_ha=0.02,
        )

        geojson_path = out_geo_dir / f"{tile_id}_detected_changes.geojson"
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        all_geojson_polygons.append({
            "tile_id": tile_id,
            "polygon_count": len(geojson["features"]),
            "geojson_path": str(geojson_path),
        })

        all_targets.append(target_mask.flatten())
        all_deep_preds.append(filtered_pred.flatten())
        all_baseline_preds.append(base_pred.flatten())

    targets_flat = np.concatenate(all_targets)
    deep_preds_flat = np.concatenate(all_deep_preds)
    base_preds_flat = np.concatenate(all_baseline_preds)

    deep_metrics = compute_metrics(deep_preds_flat, targets_flat, num_classes=4)
    baseline_metrics = compute_metrics(base_preds_flat, targets_flat, num_classes=4)

    # Save baseline comparison summary
    comparison_table = {
        "model_comparison": {
            "Deep_Siamese_UNet": {
                "mean_iou": deep_metrics["mean_iou"],
                "change_mean_iou": deep_metrics["change_mean_iou"],
                "mean_f1": deep_metrics["mean_f1"],
                "sand_mining_iou": deep_metrics["iou_class_1"],
                "forest_clearing_iou": deep_metrics["iou_class_2"],
                "new_track_iou": deep_metrics["iou_class_3"],
            },
            "Simple_NDVI_Baseline": {
                "mean_iou": baseline_metrics["mean_iou"],
                "change_mean_iou": baseline_metrics["change_mean_iou"],
                "mean_f1": baseline_metrics["mean_f1"],
                "sand_mining_iou": baseline_metrics["iou_class_1"],
                "forest_clearing_iou": baseline_metrics["iou_class_2"],
                "new_track_iou": baseline_metrics["iou_class_3"],
            },
        },
        "geojson_outputs": all_geojson_polygons,
    }

    metrics_out = metrics_dir / "change_detection_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(comparison_table, f, indent=2)

    # Generate and save Confusion Matrix figure
    cm = confusion_matrix(targets_flat, deep_preds_flat, labels=[0, 1, 2, 3], normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    short_labels = ["BG", "Sand Mining", "Clearing", "Track"]
    ax.set_xticklabels(short_labels, rotation=30, ha="right")
    ax.set_yticklabels(short_labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Normalized Confusion Matrix: Siamese U-Net")

    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    cm_path = figures_dir / "change_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    logger.info("Evaluation complete! Metrics written to %s, figure to %s", metrics_out, cm_path)
    return comparison_table


if __name__ == "__main__":
    evaluate_test_set_and_export()
