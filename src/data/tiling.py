"""Tiling, raster operations, and GeoTIFF georeferencing utilities.

Provides utilities for slicing large spatial rasters or AOIs into standard
256x256 tiles at 10m spatial resolution with 10 channels:
[B2, B3, B4, B8, B11, B12, NDVI, NDWI, S1_VV, S1_VH].
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

logger = logging.getLogger(__name__)

# Standard 10 bands used across all change detection pipelines
BAND_NAMES = [
    "B2_BLUE",
    "B3_GREEN",
    "B4_RED",
    "B8_NIR",
    "B11_SWIR1",
    "B12_SWIR2",
    "NDVI",
    "NDWI",
    "S1_VV",
    "S1_VH",
]


def calculate_indices(
    b2: np.ndarray,
    b3: np.ndarray,
    b4: np.ndarray,
    b8: np.ndarray,
    b11: np.ndarray,
    b12: np.ndarray,
    eps: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate NDVI and NDWI vegetation and water indices.

    NDVI = (B8 - B4) / (B8 + B4)
    NDWI = (B3 - B8) / (B3 + B8)  (McFeeters water index)
    """
    ndvi = (b8 - b4) / (b8 + b4 + eps)
    ndwi = (b3 - b8) / (b3 + b8 + eps)
    return np.clip(ndvi, -1.0, 1.0), np.clip(ndwi, -1.0, 1.0)


def create_geotiff(
    output_path: Path | str,
    data: np.ndarray,
    transform: Affine,
    crs: CRS | str = "EPSG:4326",
    nodata: float | None = -9999.0,
    descriptions: list[str] | None = None,
) -> Path:
    """Save multi-band array as a georeferenced GeoTIFF.

    Args:
        output_path: Destination file path.
        data: Array of shape (channels, height, width).
        transform: Affine geotransform.
        crs: Coordinate reference system.
        nodata: Nodata sentinel value.
        descriptions: Optional band names.

    Returns:
        Path to written GeoTIFF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)

    count, height, width = data.shape

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": data.dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)
        if descriptions:
            for idx, desc in enumerate(descriptions, start=1):
                dst.set_band_description(idx, desc)

    return output_path


def read_geotiff(file_path: Path | str) -> tuple[np.ndarray, dict[str, Any]]:
    """Read multi-band GeoTIFF and its spatial profile.

    Returns:
        Tuple of (array of shape [C, H, W], profile dict).
    """
    with rasterio.open(file_path) as src:
        data = src.read()
        profile = src.profile.copy()
        profile["descriptions"] = src.descriptions
        return data, profile


def validate_tile(
    tile_data: np.ndarray,
    expected_channels: int = 10,
    expected_size: int = 256,
    max_cloud_fraction: float = 0.50,
) -> tuple[bool, str]:
    """Validate tile geometry, channel count, and cloud/nodata threshold.

    Returns:
        Tuple of (is_valid: bool, reason: str).
    """
    if tile_data.ndim != 3:
        return False, f"Expected 3D array (C, H, W), got {tile_data.shape}"

    c, h, w = tile_data.shape
    if c != expected_channels:
        return False, f"Expected {expected_channels} channels, got {c}"
    if h != expected_size or w != expected_size:
        return False, f"Expected {expected_size}x{expected_size}, got {h}x{w}"

    # Check for complete nodata or zero tiles
    if np.all(tile_data == 0) or np.all(np.isnan(tile_data)):
        return False, "Tile contains 100% nodata or zeroes"

    # Cloud proxy check on optical reflectance (bright saturated pixels)
    # B2, B3, B4 reflectance > 0.40 typically indicates heavy cloud cover
    b2_refl = tile_data[0]
    cloud_mask = b2_refl > 0.45
    cloud_fraction = float(np.mean(cloud_mask))
    if cloud_fraction > max_cloud_fraction:
        return False, f"Tile cloud cover ({cloud_fraction:.1%}) exceeds maximum ({max_cloud_fraction:.1%})"

    return True, "Tile valid"
