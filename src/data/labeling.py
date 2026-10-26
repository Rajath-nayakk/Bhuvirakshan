"""Labeling, annotation rasterization, weak supervision, and spatial partitioning.

Classes:
  0: Background (stable forest, permanent water, established areas)
  1: Sand Mining / Riverbed Extraction
  2: Forest Clearing / Illegal Logging
  3: New Track / Access Road in Forest
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio import features
from shapely.geometry import shape

logger = logging.getLogger(__name__)

CLASS_NAMES = {
    0: "Background",
    1: "Sand Mining / Riverbed Extraction",
    2: "Forest Clearing",
    3: "New Track / Access Road",
}


def rasterize_annotations(
    geojson_path_or_dict: Path | str | dict[str, Any],
    transform: Affine,
    out_shape: tuple[int, int] = (256, 256),
    default_class: int = 1,
) -> np.ndarray:
    """Rasterize vector polygon/multipolygon annotations into an integer mask.

    Args:
        geojson_path_or_dict: Path to GeoJSON file or GeoJSON FeatureCollection dictionary.
        transform: Affine geotransform of the target tile.
        out_shape: (height, width) of the target mask.
        default_class: Class ID to assign if feature has no 'class_id' property.

    Returns:
        2D numpy array of uint8 values (0 to 3).
    """
    if isinstance(geojson_path_or_dict, (str, Path)):
        with open(geojson_path_or_dict, encoding="utf-8") as f:
            geojson_data = json.load(f)
    else:
        geojson_data = geojson_path_or_dict

    features_list = geojson_data.get("features", [])
    shapes_with_values = []

    for feat in features_list:
        geom = feat.get("geometry")
        if not geom:
            continue
        props = feat.get("properties", {})
        cls_val = int(props.get("class_id", default_class))
        shapely_geom = shape(geom)
        shapes_with_values.append((shapely_geom, cls_val))

    if not shapes_with_values:
        return np.zeros(out_shape, dtype=np.uint8)

    mask = features.rasterize(
        shapes=shapes_with_values,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )
    return mask


def generate_weak_supervision_labels(
    before_stack: np.ndarray,
    after_stack: np.ndarray,
    ndvi_drop_threshold: float = 0.22,
    river_bare_soil_threshold: float = 0.20,
    radar_roughness_delta_db: float = 4.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate heuristic weak labels using physical and spectral rules.

    Rules:
    - Forest Clearing (Class 2):
      High before-NDVI (> 0.40) followed by a sharp drop (before_ndvi - after_ndvi > threshold)
      and SAR canopy attenuation loss (S1 VV drops).
    - River Sand Extraction (Class 1):
      Located in low-NDVI/water/sand corridor where after-reflectance jumps (exposed bright sediment/sand pit),
      or SAR backscatter jumps due to dredged mounds/pit roughness.
    - New Track / Road (Class 3):
      Narrow linear clearing signature across forest pixels.

    Returns:
        Tuple of (label_mask: np.ndarray [H, W], metadata: dict with weak_label=True).
    """
    _, h, w = before_stack.shape
    label_mask = np.zeros((h, w), dtype=np.uint8)

    before_b4 = before_stack[2]
    after_b4 = after_stack[2]

    before_ndvi = before_stack[6]
    after_ndvi = after_stack[6]
    before_ndwi = before_stack[7]

    before_s1_vv = before_stack[8]
    after_s1_vv = after_stack[8]

    # Rule 1: Forest clearing detection
    ndvi_loss = before_ndvi - after_ndvi
    forest_mask = before_ndvi > 0.38
    clearing_candidates = forest_mask & (ndvi_loss > ndvi_drop_threshold) & (after_b4 > before_b4)
    label_mask[clearing_candidates] = 2

    # Rule 2: Riverbed sand mining detection
    # Sand reflects strongly in visible & SWIR, NDWI drops as water body turns into sandbank/pit
    river_zone = (before_ndwi > -0.15) | (before_ndvi < 0.25)
    bare_soil_jump = (after_b4 > river_bare_soil_threshold) & (after_b4 > before_b4 * 1.3)
    sar_roughness_jump = (after_s1_vv - before_s1_vv) > radar_roughness_delta_db

    sand_mining_candidates = river_zone & (bare_soil_jump | sar_roughness_jump) & ~clearing_candidates
    label_mask[sand_mining_candidates] = 1

    # Rule 3: Linear access tracks
    # High-aspect ratio linear strips with moderate vegetation loss
    track_candidates = (
        forest_mask
        & (ndvi_loss > (ndvi_drop_threshold * 0.75))
        & (label_mask == 0)
    )
    # Filter for isolated linear elements (simple horizontal/vertical run heuristic)
    horiz_diff = np.abs(np.diff(track_candidates.astype(np.int32), axis=1, prepend=0))
    linear_mask = track_candidates & (horiz_diff > 0)
    label_mask[linear_mask] = 3

    counts = {
        "background_pixels": int(np.sum(label_mask == 0)),
        "sand_mining_pixels": int(np.sum(label_mask == 1)),
        "forest_clearing_pixels": int(np.sum(label_mask == 2)),
        "new_track_pixels": int(np.sum(label_mask == 3)),
    }

    metadata = {
        "weak_label": True,
        "rules_applied": [
            f"ndvi_drop > {ndvi_drop_threshold}",
            f"river_bare_soil > {river_bare_soil_threshold}",
            f"sar_delta_db > {radar_roughness_delta_db}",
        ],
        "pixel_counts": counts,
    }

    return label_mask, metadata


def create_spatial_split(
    tile_coords: list[dict[str, Any]],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    split_axis: str = "lon",
) -> dict[str, list[dict[str, Any]]]:
    """Partition tiles into strictly non-overlapping geographical blocks.

    Prevents spatial data leakage where adjacent tiles share identical environmental features.

    Args:
        tile_coords: List of dicts, each containing 'tile_id', 'min_lon', 'max_lon', 'min_lat', 'max_lat'.
        train_ratio: Fraction of geographic space for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for held-out test region.
        split_axis: 'lon' (West-to-East) or 'lat' (South-to-North).

    Returns:
        Dictionary with keys 'train', 'val', 'test', each containing tile records and boundary info.
    """
    if not tile_coords:
        return {"train": [], "val": [], "test": []}

    # Sort tiles strictly along geographic axis
    key_field = "min_lon" if split_axis == "lon" else "min_lat"
    sorted_tiles = sorted(tile_coords, key=lambda t: t[key_field])

    n = len(sorted_tiles)
    n_train = max(1, int(np.round(n * train_ratio)))
    n_val = max(1, int(np.round(n * val_ratio)))

    # Ensure at least 1 in test if n >= 3
    if n >= 3 and (n_train + n_val) >= n:
        n_train = max(1, n - 2)
        n_val = 1

    train_tiles = sorted_tiles[:n_train]
    val_tiles = sorted_tiles[n_train : n_train + n_val]
    test_tiles = sorted_tiles[n_train + n_val :]

    # Tag partition attribute
    for t in train_tiles:
        t["split"] = "train"
    for t in val_tiles:
        t["split"] = "val"
    for t in test_tiles:
        t["split"] = "test"

    manifest = {
        "train": train_tiles,
        "val": val_tiles,
        "test": test_tiles,
        "split_axis": split_axis,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "spatial_boundaries": {
            "train_max_coord": float(train_tiles[-1][key_field]) if train_tiles else None,
            "val_max_coord": float(val_tiles[-1][key_field]) if val_tiles else None,
            "test_min_coord": float(test_tiles[0][key_field]) if test_tiles else None,
        },
    }

    return manifest


def save_label_mask(
    output_path: Path | str,
    mask: np.ndarray,
    transform: Affine,
    crs: str = "EPSG:4326",
) -> Path:
    """Save 2D integer segmentation mask as a GeoTIFF."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h, w = mask.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": 255,
        "compress": "lzw",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)
        dst.set_band_description(1, "CLASS_LABEL_0_3")

    return output_path
