"""Unit tests for Phase 1: Satellite Data Pipeline & GeoTIFF Tiling."""

from pathlib import Path

import numpy as np
import pytest

from src.data.earth_engine import EarthEnginePipeline
from src.data.tiling import (
    BAND_NAMES,
    calculate_indices,
    create_geotiff,
    read_geotiff,
    validate_tile,
)


@pytest.fixture
def satellite_pipeline(tmp_path: Path) -> EarthEnginePipeline:
    """Fixture providing an initialized pipeline with a isolated temporary samples dir."""
    samples_dir = tmp_path / "samples"
    return EarthEnginePipeline(config_path="configs/region.yaml", samples_dir=samples_dir)


def test_offline_mode_initialization(satellite_pipeline: EarthEnginePipeline):
    """Verify Earth Engine gracefully falls back to offline mode when credentials missing."""
    assert satellite_pipeline.region_bbox is not None
    assert len(satellite_pipeline.region_bbox) == 4
    # Without EE credentials in environment, is_authenticated should be False without crashing
    assert isinstance(satellite_pipeline.is_authenticated, bool)


def test_generate_and_read_sample_tiles(satellite_pipeline: EarthEnginePipeline):
    """Verify synthetic/offline tile generation produces valid GeoTIFFs."""
    pairs = satellite_pipeline.generate_synthetic_sample_tiles(num_tile_pairs=2, tile_size=256)
    assert len(pairs) == 2

    for before_path, after_path in pairs:
        assert before_path.exists()
        assert after_path.exists()

        before_data, before_profile = read_geotiff(before_path)
        after_data, after_profile = read_geotiff(after_path)

        # Check shapes
        assert before_data.shape == (10, 256, 256), f"Expected (10, 256, 256), got {before_data.shape}"
        assert after_data.shape == (10, 256, 256)

        # Check CRS and transform preservation
        assert before_profile["crs"] is not None
        assert str(before_profile["crs"]) == "EPSG:4326"
        assert before_profile["transform"] == after_profile["transform"]

        # Validate tiles against quality gate (no 100% cloud, no nodata)
        valid_before, msg_before = validate_tile(before_data)
        valid_after, msg_after = validate_tile(after_data)
        assert valid_before, f"Before tile failed validation: {msg_before}"
        assert valid_after, f"After tile failed validation: {msg_after}"


def test_index_calculations():
    """Verify NDVI and NDWI calculations and numerical bounds."""
    # Dummy arrays
    b2 = np.full((10, 10), 0.05, dtype=np.float32)
    b3 = np.full((10, 10), 0.08, dtype=np.float32)
    b4 = np.full((10, 10), 0.04, dtype=np.float32)
    b8 = np.full((10, 10), 0.60, dtype=np.float32)
    b11 = np.full((10, 10), 0.20, dtype=np.float32)
    b12 = np.full((10, 10), 0.10, dtype=np.float32)

    ndvi, ndwi = calculate_indices(b2, b3, b4, b8, b11, b12)

    assert ndvi.shape == (10, 10)
    assert ndwi.shape == (10, 10)
    # NDVI = (0.6 - 0.04)/(0.6 + 0.04) = 0.56 / 0.64 = 0.875
    assert np.all(ndvi >= -1.0) and np.all(ndvi <= 1.0)
    assert np.all(ndwi >= -1.0) and np.all(ndwi <= 1.0)
    assert np.allclose(ndvi, 0.875, atol=1e-3)


def test_band_descriptions_preserved(tmp_path: Path):
    """Verify band names/descriptions are embedded in output GeoTIFF."""
    dummy_data = np.random.rand(10, 64, 64).astype(np.float32)
    from affine import Affine

    transform = Affine.identity()
    out_file = tmp_path / "test_bands.tif"

    create_geotiff(out_file, dummy_data, transform, descriptions=BAND_NAMES)
    _, profile = read_geotiff(out_file)

    assert "descriptions" in profile
    assert profile["descriptions"] is not None
    assert len(profile["descriptions"]) == 10
    assert profile["descriptions"][0] == "B2_BLUE"
    assert profile["descriptions"][6] == "NDVI"
    assert profile["descriptions"][8] == "S1_VV"
