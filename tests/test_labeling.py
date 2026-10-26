"""Unit tests for Phase 2: Labeling, GeoJSON rasterization, and spatial partitioning."""

from pathlib import Path

import numpy as np
from affine import Affine

from src.data.labeling import (
    CLASS_NAMES,
    create_spatial_split,
    generate_weak_supervision_labels,
    rasterize_annotations,
    save_label_mask,
)
from src.data.tiling import read_geotiff


def test_class_names_defined():
    """Verify standard 4 classes are properly registered."""
    assert len(CLASS_NAMES) == 4
    assert CLASS_NAMES[0] == "Background"
    assert CLASS_NAMES[1] == "Sand Mining / Riverbed Extraction"
    assert CLASS_NAMES[2] == "Forest Clearing"
    assert CLASS_NAMES[3] == "New Track / Access Road"


def test_geojson_rasterization(tmp_path: Path):
    """Verify rasterizing vector polygons produces accurate integer class masks."""
    # Define affine transform spanning 10.0 to 10.25 deg lon, 20.0 to 20.25 deg lat
    transform = Affine.translation(10.0, 20.25) @ Affine.scale(0.001, -0.001)

    geojson_feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [10.02, 20.20],
                            [10.08, 20.20],
                            [10.08, 20.23],
                            [10.02, 20.23],
                            [10.02, 20.20],
                        ]
                    ],
                },
                "properties": {"incident": "Illegal Dredging", "class_id": 1},
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [10.12, 20.10],
                            [10.18, 20.10],
                            [10.18, 20.15],
                            [10.12, 20.15],
                            [10.12, 20.10],
                        ]
                    ],
                },
                "properties": {"incident": "Illegal Tree Felling", "class_id": 2},
            },
        ],
    }

    mask = rasterize_annotations(geojson_feature_collection, transform, out_shape=(256, 256))
    assert mask.shape == (256, 256)
    assert mask.dtype == np.uint8

    # Verify both class 1 and class 2 exist in mask
    assert np.sum(mask == 1) > 0, "Class 1 (Sand Mining) pixels must be rasterized"
    assert np.sum(mask == 2) > 0, "Class 2 (Forest Clearing) pixels must be rasterized"
    assert np.sum(mask == 0) > 0, "Background pixels must remain 0"


def test_weak_supervision_heuristic():
    """Verify weak labeling rules accurately detect vegetation loss and riverbed changes."""
    before_stack = np.zeros((10, 64, 64), dtype=np.float32)
    after_stack = np.zeros((10, 64, 64), dtype=np.float32)

    # Setup baseline: high NDVI forest
    before_stack[2] = 0.05   # Red
    before_stack[3] = 0.50   # NIR
    before_stack[6] = 0.81   # NDVI
    before_stack[7] = -0.60  # NDWI
    before_stack[8] = -12.0  # S1 VV

    # Copy to after stack
    after_stack = before_stack.copy()

    # Inject forest clearing in top-left block: NDVI drops, Red jumps
    after_stack[2, 5:20, 5:20] = 0.25  # Red jump
    after_stack[3, 5:20, 5:20] = 0.15  # NIR drop
    after_stack[6, 5:20, 5:20] = -0.25 # NDVI drop

    # Inject riverbed sand mining in bottom-right: low NDWI, bare soil jump
    before_stack[7, 40:55, 40:55] = 0.10  # River channel
    before_stack[6, 40:55, 40:55] = 0.05
    after_stack[2, 40:55, 40:55] = 0.35   # Bright sand reflectance
    after_stack[8, 40:55, 40:55] = -5.0   # Radar backscatter roughness increase

    mask, metadata = generate_weak_supervision_labels(before_stack, after_stack)

    assert mask.shape == (64, 64)
    assert metadata["weak_label"] is True
    assert metadata["pixel_counts"]["forest_clearing_pixels"] > 0
    assert metadata["pixel_counts"]["sand_mining_pixels"] > 0
    assert np.all(mask[10, 10] == 2)  # Injected clearing location


def test_spatial_split_no_leakage():
    """Verify spatial partitioning ensures strictly disjoint geographic blocks."""
    # Create 6 synthetic tile coordinates across a longitudinal transect
    tiles = [
        {"tile_id": f"t_{i}", "min_lon": 77.10 + i * 0.04, "max_lon": 77.14 + i * 0.04, "min_lat": 12.0, "max_lat": 12.04}
        for i in range(6)
    ]

    split_manifest = create_spatial_split(tiles, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, split_axis="lon")

    train_tiles = split_manifest["train"]
    val_tiles = split_manifest["val"]
    test_tiles = split_manifest["test"]

    assert len(train_tiles) > 0
    assert len(val_tiles) > 0
    assert len(test_tiles) > 0

    max_train_lon = max(t["max_lon"] for t in train_tiles)
    min_val_lon = min(t["min_lon"] for t in val_tiles)
    max_val_lon = max(t["max_lon"] for t in val_tiles)
    min_test_lon = min(t["min_lon"] for t in test_tiles)

    # Strict spatial disjunction: Train < Val < Test along longitude
    assert max_train_lon <= min_val_lon, f"Spatial overlap between train ({max_train_lon}) and val ({min_val_lon})"
    assert max_val_lon <= min_test_lon, f"Spatial overlap between val ({max_val_lon}) and test ({min_test_lon})"


def test_save_label_mask(tmp_path: Path):
    """Verify label mask can be saved as a georeferenced single-band GeoTIFF."""
    mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    transform = Affine.translation(10, 20) @ Affine.scale(1, -1)
    out_file = tmp_path / "mask.tif"

    save_label_mask(out_file, mask, transform)
    data, profile = read_geotiff(out_file)

    assert data.shape == (1, 2, 2)
    assert profile["dtype"] == "uint8"
    assert np.array_equal(data[0], mask)
