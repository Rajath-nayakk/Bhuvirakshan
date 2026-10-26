"""Geospatial 500m grid generation, UTM projection, and proximity feature calculation.

Partitions the study region into a regular 500m grid and calculates distance
to river channels, roads, and protected area boundaries.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import shapely
import yaml
from shapely.geometry import LineString, Polygon, mapping

logger = logging.getLogger(__name__)


class StudyRegionGrid:
    """Manages the 500m spatial monitoring grid over the study AOI."""

    def __init__(
        self,
        config_path: Path | str = "configs/region.yaml",
        cell_size_meters: float = 500.0,
    ):
        self.config_path = Path(config_path)
        with open(self.config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.region_bbox = self.config["region"]["bbox"]  # [min_lon, min_lat, max_lon, max_lat]
        self.utm_epsg = self.config["region"].get("utm_epsg", 32643)
        self.cell_size_m = cell_size_meters

        # Setup coordinate transformers
        self.wgs84 = pyproj.CRS("EPSG:4326")
        self.utm = pyproj.CRS(f"EPSG:{self.utm_epsg}")
        self.to_utm = pyproj.Transformer.from_crs(self.wgs84, self.utm, always_xy=True).transform
        self.to_wgs84 = pyproj.Transformer.from_crs(self.utm, self.wgs84, always_xy=True).transform

        # Define simulated vector reference features in UTM (River stretch, Road, Boundary)
        self._init_reference_features()

    def _init_reference_features(self):
        """Initialize reference spatial lines for proximity calculations."""
        min_lon, min_lat, max_lon, max_lat = self.region_bbox
        # Transform bbox corners to UTM meters
        x_min, y_min = self.to_utm(min_lon, min_lat)
        x_max, y_max = self.to_utm(max_lon, max_lat)

        # River channel: Meandering diagonal river corridor
        river_pts = [
            (x_min, y_min + 0.3 * (y_max - y_min)),
            (x_min + 0.3 * (x_max - x_min), y_min + 0.45 * (y_max - y_min)),
            (x_min + 0.7 * (x_max - x_min), y_min + 0.55 * (y_max - y_min)),
            (x_max, y_min + 0.8 * (y_max - y_min)),
        ]
        self.river_line = LineString(river_pts)

        # Main Access Road: Running roughly parallel to the south
        road_pts = [
            (x_min, y_min + 0.15 * (y_max - y_min)),
            (x_min + 0.5 * (x_max - x_min), y_min + 0.25 * (y_max - y_min)),
            (x_max, y_min + 0.4 * (y_max - y_min)),
        ]
        self.road_line = LineString(road_pts)

        # Core Reserve Boundary: Northern half
        boundary_pts = [
            (x_min, y_min + 0.6 * (y_max - y_min)),
            (x_max, y_min + 0.6 * (y_max - y_min)),
        ]
        self.boundary_line = LineString(boundary_pts)

    def generate_grid(self) -> list[dict[str, Any]]:
        """Generate 500m metric grid cells over the study region bounding box.

        Returns list of cell records with UTM and WGS84 geometries, centroids,
        and proximity metrics to critical environmental landmarks.
        """
        min_lon, min_lat, max_lon, max_lat = self.region_bbox
        x_min, y_min = self.to_utm(min_lon, min_lat)
        x_max, y_max = self.to_utm(max_lon, max_lat)

        x_coords = np.arange(x_min, x_max, self.cell_size_m)
        y_coords = np.arange(y_min, y_max, self.cell_size_m)

        grid_cells = []
        cell_idx = 1

        for x in x_coords:
            for y in y_coords:
                # Square polygon in UTM
                utm_poly = Polygon([
                    (x, y),
                    (x + self.cell_size_m, y),
                    (x + self.cell_size_m, y + self.cell_size_m),
                    (x, y + self.cell_size_m),
                    (x, y),
                ])

                # Centroid
                utm_centroid = utm_poly.centroid
                c_lon, c_lat = self.to_wgs84(utm_centroid.x, utm_centroid.y)

                # Distance to river, road, and boundary in meters
                dist_river_m = float(round(utm_centroid.distance(self.river_line), 1))
                dist_road_m = float(round(utm_centroid.distance(self.road_line), 1))
                dist_boundary_m = float(round(utm_centroid.distance(self.boundary_line), 1))

                # Project polygon to WGS84 for web map visualization
                wgs84_poly = shapely.transform(utm_poly, self.to_wgs84, interleaved=False)

                cell_id = f"CELL_{cell_idx:04d}"
                cell_record = {
                    "cell_id": cell_id,
                    "centroid": [round(c_lon, 5), round(c_lat, 5)],
                    "utm_bounds": [round(x, 1), round(y, 1), round(x + self.cell_size_m, 1), round(y + self.cell_size_m, 1)],
                    "dist_to_river_m": dist_river_m,
                    "dist_to_road_m": dist_road_m,
                    "dist_to_boundary_m": dist_boundary_m,
                    "geometry": mapping(wgs84_poly),
                }
                grid_cells.append(cell_record)
                cell_idx += 1

        logger.info("Generated %d regular 500m grid cells across study region", len(grid_cells))
        return grid_cells

    def export_geojson(self, output_path: Path | str = "data/processed/risk_grid_500m.geojson") -> Path:
        """Export grid cells to GeoJSON FeatureCollection."""
        output_p = Path(output_path)
        output_p.parent.mkdir(parents=True, exist_ok=True)

        cells = self.generate_grid()
        features = []
        for c in cells:
            feat = {
                "type": "Feature",
                "geometry": c["geometry"],
                "properties": {
                    "cell_id": c["cell_id"],
                    "centroid_lon": c["centroid"][0],
                    "centroid_lat": c["centroid"][1],
                    "dist_to_river_m": c["dist_to_river_m"],
                    "dist_to_road_m": c["dist_to_road_m"],
                    "dist_to_boundary_m": c["dist_to_boundary_m"],
                },
            }
            features.append(feat)

        geojson = {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
            "features": features,
        }

        with open(output_p, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        logger.info("Exported grid GeoJSON to %s", output_p)
        return output_p
