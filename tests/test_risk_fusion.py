"""Unit tests for Phase 6: Geospatial Risk Fusion, 500m Grid, and Top-N Ranking."""

from pathlib import Path

from src.risk.fusion import MultiSensorRiskFusion
from src.risk.grid import StudyRegionGrid
from src.risk.ranking import DailyPatrolRanker, generate_cell_explanation


def test_grid_generation_and_distances():
    """Verify 500m grid creates valid cells and calculates metric distances."""
    grid_gen = StudyRegionGrid(config_path="configs/region.yaml", cell_size_meters=500.0)
    cells = grid_gen.generate_grid()

    assert len(cells) > 0
    first_cell = cells[0]

    assert "cell_id" in first_cell
    assert "centroid" in first_cell
    assert len(first_cell["centroid"]) == 2
    assert "geometry" in first_cell
    assert first_cell["geometry"]["type"] == "Polygon"

    # Distances must be non-negative floats
    assert first_cell["dist_to_river_m"] >= 0.0
    assert first_cell["dist_to_road_m"] >= 0.0
    assert first_cell["dist_to_boundary_m"] >= 0.0


def test_fusion_score_bounds_and_weights():
    """Verify fusion score is bounded [0, 1] and reflects sensor evidence."""
    fusion = MultiSensorRiskFusion(risk_config_path="configs/risk.yaml")

    dummy_cell = {
        "cell_id": "CELL_0001",
        "centroid": [77.20, 12.05],
        "dist_to_river_m": 120.0,
        "dist_to_road_m": 250.0,
        "dist_to_boundary_m": 800.0,
    }

    # Case A: Low risk (no sensor alerts)
    res_low = fusion.fuse_cell(dummy_cell, [], [], [])
    assert 0.0 <= res_low["risk_score"] <= 1.0
    assert res_low["threat_level"] in ["LOW", "MEDIUM"]

    # Case B: Critical threat (large clearing, gunshots, chainsaws, vehicle)
    sat_changes = [{"area_ha": 1.8, "confidence": 0.95, "class_name": "Forest Clearing"}]
    ac_events = [{"class": "gunshot", "confidence": 0.92}, {"class": "chainsaw", "confidence": 0.90}]
    cam_events = [{"class": "vehicle", "confidence": 0.88}]

    res_critical = fusion.fuse_cell(dummy_cell, sat_changes, ac_events, cam_events, historical_incident_density=2)
    assert 0.0 <= res_critical["risk_score"] <= 1.0
    assert res_critical["risk_score"] > res_low["risk_score"]
    assert res_critical["threat_level"] in ["HIGH", "CRITICAL"]


def test_explanation_generation():
    """Verify plain-language explanation generation for enforcement teams."""
    components = {
        "satellite_change_area_ha": 1.4,
        "gunshot_event_count": 2,
        "chainsaw_event_count": 1,
        "camera_intrusion_count": 1,
        "has_new_track": True,
        "dist_to_river_m": 180.0,
        "dist_to_road_m": 250.0,
    }
    explanation = generate_cell_explanation(components, risk_score=0.85)

    assert "1.4 ha" in explanation
    assert "gunshot" in explanation
    assert "chainsaw" in explanation
    assert "river" in explanation


def test_daily_top_n_ranking(tmp_path: Path):
    """Verify DailyPatrolRanker sorts descending by risk and assigns 1-indexed ranks."""
    ranker = DailyPatrolRanker(top_n=5, output_dir=str(tmp_path))
    top_alerts = ranker.generate_daily_ranking()

    assert len(top_alerts) <= 5
    assert len(top_alerts) > 0

    # Verify descending sort order
    scores = [c["risk_score"] for c in top_alerts]
    assert scores == sorted(scores, reverse=True)

    # Verify ranking metadata
    for rank, cell in enumerate(top_alerts, start=1):
        assert cell["daily_rank"] == rank
        assert "explanation" in cell
        assert len(cell["explanation"]) > 0

    assert (tmp_path / "daily_top_alerts.json").exists()
    assert (tmp_path / "fused_risk_grid.geojson").exists()


def test_ranger_feedback_ml_model():
    """Verify training a decision tree model on confirmed ranger feedback."""
    fusion = MultiSensorRiskFusion(risk_config_path="configs/risk.yaml")

    training_records = [
        {
            "components": {
                "satellite_change_area_ha": 1.5,
                "satellite_confidence": 0.9,
                "gunshot_event_count": 2,
                "chainsaw_event_count": 1,
                "camera_intrusion_count": 2,
                "has_new_track": True,
                "proximity_risk": 0.8,
                "dist_to_river_m": 100.0,
                "historical_density": 3,
            },
            "ranger_confirmed_true": 1,
        },
        {
            "components": {
                "satellite_change_area_ha": 0.0,
                "satellite_confidence": 0.1,
                "gunshot_event_count": 0,
                "chainsaw_event_count": 0,
                "camera_intrusion_count": 0,
                "has_new_track": False,
                "proximity_risk": 0.2,
                "dist_to_river_m": 2500.0,
                "historical_density": 0,
            },
            "ranger_confirmed_true": 0,
        },
    ]

    metrics = fusion.train_ml_feedback_model(training_records)
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] >= 0.5
    assert fusion.ml_model is not None
