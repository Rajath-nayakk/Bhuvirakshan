"""Dataset builder: generates labels, computes weak supervision, and assigns spatial splits.

Prepares annotated datasets for training the Siamese U-Net change detection model.
"""

import json
import logging
from pathlib import Path
from typing import Any

from src.data.earth_engine import EarthEnginePipeline
from src.data.labeling import (
    create_spatial_split,
    generate_weak_supervision_labels,
    save_label_mask,
)
from src.data.tiling import read_geotiff

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def prepare_labeled_dataset(
    samples_dir: Path | str = "data/samples/satellite",
    labels_dir: Path | str = "data/labels",
    manifest_path: Path | str = "data/labels/spatial_splits.json",
) -> dict[str, Any]:
    """Process paired satellite tiles, generate weak ground-truth masks, and establish spatial partitions."""
    samples_dir = Path(samples_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Earth Engine pipeline to ensure sample tiles exist
    pipeline = EarthEnginePipeline(samples_dir=samples_dir)
    pairs = pipeline.load_cached_pairs()

    tile_records = []

    for before_path, after_path in pairs:
        tile_name = before_path.stem.replace("_before", "").replace("_2023", "")
        before_data, before_profile = read_geotiff(before_path)
        after_data, _ = read_geotiff(after_path)

        # Generate rule-based weak ground truth mask
        mask, meta = generate_weak_supervision_labels(before_data, after_data)

        # Save label mask as GeoTIFF
        mask_path = labels_dir / f"{tile_name}_mask.tif"
        save_label_mask(mask_path, mask, before_profile["transform"], crs=str(before_profile["crs"]))

        # Extract spatial bounding box from geotransform
        transform = before_profile["transform"]
        h, w = mask.shape
        min_lon = transform.c
        max_lat = transform.f
        max_lon = min_lon + w * transform.a
        min_lat = max_lat + h * transform.e

        record = {
            "tile_id": tile_name,
            "before_path": str(before_path),
            "after_path": str(after_path),
            "mask_path": str(mask_path),
            "min_lon": float(min(min_lon, max_lon)),
            "max_lon": float(max(min_lon, max_lon)),
            "min_lat": float(min(min_lat, max_lat)),
            "max_lat": float(max(min_lat, max_lat)),
            "metadata": meta,
        }
        tile_records.append(record)

    # Compute strict spatial split
    split_manifest = create_spatial_split(
        tile_records,
        train_ratio=0.60,
        val_ratio=0.20,
        test_ratio=0.20,
        split_axis="lon",
    )

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2)

    logger.info(
        "Dataset preparation complete: %d train, %d val, %d test tiles. Saved to %s",
        len(split_manifest["train"]),
        len(split_manifest["val"]),
        len(split_manifest["test"]),
        manifest_path,
    )
    return split_manifest


if __name__ == "__main__":
    prepare_labeled_dataset()
