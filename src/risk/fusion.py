"""Multi-sensor geospatial risk fusion engine.

Combines satellite changes, acoustic alerts, camera trap intrusion events,
proximity metrics, and historical incidents into a normalized daily risk score [0, 1].
Includes transparent weighted fusion and an optional gradient-boosted ML model.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class MultiSensorRiskFusion:
    """Fuses multi-modal observations into grid-cell daily threat risk scores."""

    def __init__(self, risk_config_path: Path | str = "configs/risk.yaml"):
        self.config_path = Path(risk_config_path)
        with open(self.config_path, encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.weights = self.config["fusion"]["weights"]
        self.prox_cfg = self.config["fusion"]["proximity"]
        self.decay_cfg = self.config["fusion"].get("temporal_decay", {})

        # Optional ML model trained on ranger patrol ground truth
        self.ml_model = None

    def calculate_proximity_risk(
        self,
        dist_to_river_m: float,
        dist_to_road_m: float,
        dist_to_boundary_m: float,
    ) -> float:
        """Calculate spatial vulnerability based on proximity to riverbed, road, and boundary.

        Closer proximity yields higher vulnerability (normalized 0 to 1).
        """
        # Critical buffers (meters)
        river_buf = self.prox_cfg.get("river_critical_buffer_m", 300.0)
        road_buf = self.prox_cfg.get("road_critical_buffer_m", 500.0)
        bound_buf = self.prox_cfg.get("core_zone_boundary_m", 1000.0)

        # Exponential decay factor: 1.0 at 0m, ~0.37 at critical buffer
        r_risk = float(np.exp(-dist_to_river_m / river_buf))
        rd_risk = float(np.exp(-dist_to_road_m / road_buf))
        b_risk = float(np.exp(-dist_to_boundary_m / bound_buf))

        w_r = self.prox_cfg.get("weight_river_proximity", 0.4)
        w_rd = self.prox_cfg.get("weight_road_proximity", 0.3)
        w_b = self.prox_cfg.get("weight_boundary_proximity", 0.3)

        return float(np.clip(w_r * r_risk + w_rd * rd_risk + w_b * b_risk, 0.0, 1.0))

    def fuse_cell(
        self,
        cell_data: dict[str, Any],
        satellite_changes: list[dict[str, Any]],
        acoustic_events: list[dict[str, Any]],
        camera_events: list[dict[str, Any]],
        historical_incident_density: int = 0,
    ) -> dict[str, Any]:
        """Compute transparent weighted multi-sensor risk score for a single grid cell."""
        # 1. Satellite Change Signals
        sat_area_ha = sum(ch.get("area_ha", 0.0) for ch in satellite_changes)
        sat_conf = max([ch.get("confidence", 0.0) for ch in satellite_changes] or [0.0])
        has_new_track = any(ch.get("class_name") == "New Track / Access Road" for ch in satellite_changes)

        # Normalize area (saturates at 2.0 ha)
        norm_area = float(np.clip(sat_area_ha / 2.0, 0.0, 1.0))

        # 2. Acoustic Signals
        gunshot_events = [ev for ev in acoustic_events if ev.get("class") == "gunshot"]
        chainsaw_events = [ev for ev in acoustic_events if ev.get("class") == "chainsaw"]

        gunshot_count = len(gunshot_events)
        chainsaw_count = len(chainsaw_events)

        norm_gunshots = float(np.clip(gunshot_count / 2.0, 0.0, 1.0))
        norm_chainsaws = float(np.clip(chainsaw_count / 3.0, 0.0, 1.0))

        # 3. Camera Trap Intrusion Signals
        human_events = [ev for ev in camera_events if ev.get("class") == "person"]
        vehicle_events = [ev for ev in camera_events if ev.get("class") == "vehicle"]
        trap_events = [ev for ev in camera_events if ev.get("class") == "snare_or_trap"]

        intrusion_count = len(human_events) + len(vehicle_events) + (len(trap_events) * 2)
        norm_intrusion = float(np.clip(intrusion_count / 3.0, 0.0, 1.0))

        # 4. Proximity Penalty
        prox_risk = self.calculate_proximity_risk(
            dist_to_river_m=cell_data["dist_to_river_m"],
            dist_to_road_m=cell_data["dist_to_road_m"],
            dist_to_boundary_m=cell_data["dist_to_boundary_m"],
        )

        # 5. Weighted Linear Combination
        w = self.weights
        raw_score = (
            w.get("satellite_change_area", 0.30) * norm_area
            + w.get("satellite_confidence", 0.10) * sat_conf
            + w.get("acoustic_gunshot_events", 0.20) * norm_gunshots
            + w.get("acoustic_chainsaw_events", 0.15) * norm_chainsaws
            + w.get("camera_intrusion_events", 0.15) * norm_intrusion
            + w.get("track_detection", 0.05) * (1.0 if has_new_track else 0.0)
            + w.get("proximity_penalty", 0.05) * prox_risk
        )

        # Boost slightly if historical hotspot
        if historical_incident_density > 0:
            raw_score += min(0.10, historical_incident_density * 0.02)

        risk_score = float(np.clip(raw_score, 0.0, 1.0))

        # Risk level classification
        if risk_score >= 0.70:
            threat_level = "CRITICAL"
        elif risk_score >= 0.40:
            threat_level = "HIGH"
        elif risk_score >= 0.20:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"

        return {
            "cell_id": cell_data["cell_id"],
            "centroid": cell_data["centroid"],
            "risk_score": round(risk_score, 3),
            "threat_level": threat_level,
            "components": {
                "satellite_change_area_ha": round(sat_area_ha, 3),
                "satellite_confidence": round(sat_conf, 3),
                "gunshot_event_count": gunshot_count,
                "chainsaw_event_count": chainsaw_count,
                "camera_intrusion_count": intrusion_count,
                "has_new_track": has_new_track,
                "proximity_risk": round(prox_risk, 3),
                "dist_to_river_m": cell_data["dist_to_river_m"],
                "dist_to_road_m": cell_data["dist_to_road_m"],
                "dist_to_boundary_m": cell_data["dist_to_boundary_m"],
                "historical_density": historical_incident_density,
            },
            "evidence_details": {
                "satellite_changes": satellite_changes,
                "acoustic_events": acoustic_events,
                "camera_events": camera_events,
            },
        }

    def train_ml_feedback_model(
        self,
        training_records: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Train a gradient-boosted decision tree on confirmed ranger ground truth."""
        X = []
        y = []

        for rec in training_records:
            comp = rec["components"]
            features = [
                comp["satellite_change_area_ha"],
                comp["satellite_confidence"],
                comp["gunshot_event_count"],
                comp["chainsaw_event_count"],
                comp["camera_intrusion_count"],
                1.0 if comp["has_new_track"] else 0.0,
                comp["proximity_risk"],
                comp["dist_to_river_m"],
                comp["historical_density"],
            ]
            X.append(features)
            # Label: 1 if ranger confirmed alert was true, 0 if false alarm
            y.append(rec.get("ranger_confirmed_true", 0))

        X = np.array(X)
        y = np.array(y)

        # Train Random Forest
        clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
        clf.fit(X, y)
        self.ml_model = clf

        probs = clf.predict_proba(X)[:, 1]
        auc = float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 1.0
        logger.info("Trained ranger feedback ML model with ROC-AUC: %.4f", auc)
        return {"roc_auc": round(auc, 4), "samples": len(y)}
