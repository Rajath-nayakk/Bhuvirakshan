"""Phase 0 Setup and configuration verification tests."""

from pathlib import Path

import pytest
import yaml


def test_configs_exist_and_valid():
    """Verify configs exist and parse as valid YAML dictionaries."""
    config_dir = Path("configs")
    assert config_dir.exists(), "configs directory must exist"

    required_configs = ["region.yaml", "training.yaml", "risk.yaml"]
    for cfg in required_configs:
        cfg_path = config_dir / cfg
        assert cfg_path.exists(), f"Configuration file {cfg} must exist"
        with open(cfg_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
            assert isinstance(data, dict), f"{cfg} must contain a dictionary root"


def test_region_config_parameters():
    """Verify region configuration contains required spatial bbox and river specs."""
    with open("configs/region.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "region" in data
    region = data["region"]
    assert "bbox" in region
    assert len(region["bbox"]) == 4
    min_lon, min_lat, max_lon, max_lat = region["bbox"]
    assert min_lon < max_lon
    assert min_lat < max_lat
    assert "river" in region
    assert "forest_block" in region


def test_risk_config_weights_sum():
    """Verify fusion weights are valid and normalized to 1.0."""
    with open("configs/risk.yaml", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    weights = data["fusion"]["weights"]
    total_weight = sum(weights.values())
    assert pytest.approx(total_weight, rel=1e-3) == 1.0


def test_critical_dependencies_importable():
    """Verify critical ML and geospatial libraries import without failure."""
    import fastapi
    import geopandas
    import rasterio
    import shapely
    import sqlalchemy
    import torch

    assert torch.__version__ is not None
    assert rasterio.__version__ is not None
    assert shapely.__version__ is not None
    assert geopandas.__version__ is not None
    assert fastapi.__version__ is not None
    assert sqlalchemy.__version__ is not None
