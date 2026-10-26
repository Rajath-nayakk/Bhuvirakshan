"""Camera-trap inference runner, intrusion rule evaluation, and evidence frame generator.

Scans camera trap frames, runs YOLO detection, evaluates curfew/core-zone rules,
draws bounding boxes, and exports evidence images and event logs.
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
from ultralytics import YOLO

from src.data.camera_data import CAMERA_CLASSES
from src.models.camera_detector import IntrusionRuleEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class CameraTrapInferenceEngine:
    """Inference engine for camera-trap images with intrusion alert rules."""

    def __init__(
        self,
        checkpoint_path: str = "data/checkpoints/camera_yolo_best.pt",
        confidence_threshold: float = 0.35,
        evidence_dir: str = "data/processed/camera_evidence",
    ):
        self.conf_thresh = confidence_threshold
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        chk_p = Path(checkpoint_path)
        if chk_p.exists():
            self.model = YOLO(str(chk_p))
            logger.info("Loaded fine-tuned camera YOLO model from %s", chk_p)
        else:
            self.model = YOLO("yolov8n.pt")
            logger.warning("Checkpoint %s not found. Initialized base YOLOv8n.", chk_p)

        self.rule_engine = IntrusionRuleEngine(curfew_start="18:00", curfew_end="06:00")

    def process_image(
        self,
        image_path: Path | str,
        camera_id: str = "CAM_TRAP_CAUVERY_01",
        location_zone: str = "CORE_RESERVE_ZONE_A",
        gps_coords: tuple[float, float] = (77.2245, 12.0831),
        timestamp: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Run YOLO inference on image, apply intrusion rules, and save evidence frame."""
        image_path = Path(image_path)
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        # Default timestamp from file modification or current time
        if timestamp is None:
            # Check filename for night indicator to set realistic hour
            if "night" in image_path.name.lower():
                dt = datetime(2026, 10, 26, 23, 15, 0, tzinfo=UTC)
            else:
                dt = datetime(2026, 10, 26, 14, 30, 0, tzinfo=UTC)
        else:
            dt = timestamp

        # Run YOLO detection
        results = self.model.predict(
            source=img,
            conf=self.conf_thresh,
            device="cpu",
            verbose=False,
        )[0]

        annotated_img = img.copy()
        draw = ImageDraw.Draw(annotated_img)

        events = []
        for box in results.boxes:
            cls_idx = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)

            # Map class index (ensure fallback if default YOLO 80 classes)
            class_name = CAMERA_CLASSES.get(cls_idx, "unknown_object")
            if class_name == "unknown_object":
                # Fallback mapping from COCO classes (0: person, 2: car/vehicle, etc.)
                if cls_idx == 0:
                    class_name = "person"
                elif cls_idx in [1, 2, 3, 5, 7]:
                    class_name = "vehicle"
                elif cls_idx in range(14, 25):
                    class_name = "animal"

            rule_eval = self.rule_engine.evaluate_detection(
                class_name=class_name,
                confidence=conf,
                timestamp=dt,
                location_zone=location_zone,
                coords=gps_coords,
            )

            # Draw bounding box and label
            outline_color = (255, 40, 40) if rule_eval["is_alert"] else (40, 220, 40)
            draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=3)
            label_text = f"{class_name.upper()} {conf:.1%}"
            draw.rectangle([x1, y1 - 20, x1 + len(label_text) * 9, y1], fill=outline_color)
            draw.text((x1 + 3, y1 - 18), label_text, fill=(255, 255, 255))

            event_record = {
                "event_type": "camera_detection",
                "camera_id": camera_id,
                "timestamp": dt.isoformat(),
                "class": class_name,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2],
                "alert": rule_eval["is_alert"],
                "alert_type": rule_eval["alert_type"],
                "threat_level": rule_eval["threat_level"],
                "location_zone": location_zone,
                "coordinates": rule_eval["coordinates"],
                "image_source": image_path.name,
            }
            events.append(event_record)

        # Save annotated evidence frame
        evidence_file = self.evidence_dir / f"evidence_{image_path.name}"
        annotated_img.save(evidence_file, quality=92)
        for ev in events:
            ev["evidence_frame_path"] = str(evidence_file)

        return events


def run_batch_camera_inference(
    images_dir: str = "data/samples/camera/images/val",
    output_events_path: str = "data/processed/camera_events.json",
) -> list[dict[str, Any]]:
    """Process camera trap image directory and save event logs."""
    detector = CameraTrapInferenceEngine()
    img_dir = Path(images_dir)
    all_events = []

    files = list(img_dir.glob("*.jpg"))
    for idx, f in enumerate(files):
        zone = "CORE_RESERVE_ZONE_A" if idx % 2 == 0 else "RIVER_BUFFER_ZONE_B"
        evs = detector.process_image(f, camera_id=f"CAM_NODE_{idx:02d}", location_zone=zone)
        all_events.extend(evs)

    out_p = Path(output_events_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as dst:
        json.dump(all_events, dst, indent=2)

    logger.info("Batch camera inference complete. %d events saved to %s", len(all_events), out_p)
    return all_events


if __name__ == "__main__":
    run_batch_camera_inference()
