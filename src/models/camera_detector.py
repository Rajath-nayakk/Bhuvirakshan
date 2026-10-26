"""Camera-trap detector, night/infrared augmentation, and intrusion rule engine.

Classes:
  0: person
  1: vehicle
  2: animal
  3: snare_or_trap
"""

import logging
from datetime import datetime, time
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance

logger = logging.getLogger(__name__)


def apply_night_infrared_transform(image: Image.Image) -> Image.Image:
    """Simulate camera-trap active infrared nighttime flash.

    Converts color image to monochrome, increases contrast, adds high-ISO sensor noise,
    and applies radial illumination falloff (vignetting).
    """
    gray = image.convert("L")
    arr = np.array(gray, dtype=np.float32)

    h, w = arr.shape
    y, x = np.ogrid[:h, :w]
    center_dist = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
    vignette = np.clip(1.25 - (center_dist / (w * 0.65)), 0.25, 1.25)
    arr = arr * vignette

    # High-ISO sensor noise
    noise = np.random.normal(0, 12, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)

    ir_img = Image.fromarray(arr, mode="L").convert("RGB")
    # Slight contrast bump
    enhancer = ImageEnhance.Contrast(ir_img)
    return enhancer.enhance(1.2)


class IntrusionRuleEngine:
    """Spatial and temporal rule engine evaluating security threats and wildlife privacy."""

    def __init__(
        self,
        curfew_start: str = "18:00",
        curfew_end: str = "06:00",
        restricted_zones: list[str] | None = None,
    ):
        self.curfew_start = time.fromisoformat(curfew_start)
        self.curfew_end = time.fromisoformat(curfew_end)
        self.restricted_zones = set(restricted_zones or ["CORE_RESERVE_ZONE_A", "RIVER_BUFFER_ZONE_B"])

    def is_curfew_hours(self, dt: datetime) -> bool:
        """Check if timestamp falls into nighttime curfew hours."""
        t = dt.time()
        if self.curfew_start > self.curfew_end:
            # Crosses midnight (e.g. 18:00 to 06:00)
            return t >= self.curfew_start or t < self.curfew_end
        return self.curfew_start <= t < self.curfew_end

    def evaluate_detection(
        self,
        class_name: str,
        confidence: float,
        timestamp: datetime,
        location_zone: str = "BUFFER_ZONE",
        coords: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        """Apply enforcement and wildlife privacy rules to a single vision detection."""
        is_curfew = self.is_curfew_hours(timestamp)
        is_in_restricted_zone = location_zone in self.restricted_zones

        alert_type = "STANDARD_OBSERVATION"
        threat_level = "LOW"
        is_alert = False
        requires_immediate_dispatch = False
        is_wildlife_private = False

        if class_name in ["person", "vehicle"]:
            if is_curfew and is_in_restricted_zone:
                alert_type = "CRITICAL_NIGHT_TRESPASS"
                threat_level = "CRITICAL"
                is_alert = True
                requires_immediate_dispatch = True
            elif is_curfew:
                alert_type = "NIGHT_CURFEW_VIOLATION"
                threat_level = "HIGH"
                is_alert = True
                requires_immediate_dispatch = True
            elif is_in_restricted_zone:
                alert_type = "RESTRICTED_ZONE_INCURSION"
                threat_level = "MEDIUM"
                is_alert = True

        elif class_name == "snare_or_trap":
            alert_type = "POACHING_TRAP_HAZARD"
            threat_level = "CRITICAL"
            is_alert = True
            requires_immediate_dispatch = True

        elif class_name == "animal":
            # Strict wildlife privacy guard: obfuscate sighting coordinates from non-admin
            is_wildlife_private = True
            alert_type = "WILDLIFE_SIGHTING"
            threat_level = "NONE"

        return {
            "class_name": class_name,
            "confidence": round(confidence, 3),
            "timestamp": timestamp.isoformat(),
            "location_zone": location_zone,
            "is_alert": is_alert,
            "alert_type": alert_type,
            "threat_level": threat_level,
            "requires_immediate_dispatch": requires_immediate_dispatch,
            "is_wildlife_private": is_wildlife_private,
            "coordinates": coords if not is_wildlife_private else None,
        }
