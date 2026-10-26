"""Unit tests for Phase 5: Camera-Trap Detector, Day/Night Transforms, and Intrusion Rules."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.camera_data import (
    CAMERA_CLASSES,
    generate_synthetic_camera_trap_dataset,
)
from src.infer.run_camera import CameraTrapInferenceEngine
from src.models.camera_detector import IntrusionRuleEngine, apply_night_infrared_transform


def test_yolo_dataset_generator(tmp_path: Path):
    """Verify camera trap dataset generator creates valid images and YOLO format txt labels."""
    dataset_yaml = generate_synthetic_camera_trap_dataset(
        output_dir=tmp_path / "camera_data",
        num_train=4,
        num_val=2,
        img_size=(320, 320),
    )

    assert dataset_yaml.exists()
    train_images = list((tmp_path / "camera_data" / "images" / "train").glob("*.jpg"))
    train_labels = list((tmp_path / "camera_data" / "labels" / "train").glob("*.txt"))

    assert len(train_images) == 4
    assert len(train_labels) == 4

    # Verify YOLO annotation coordinates are normalized between 0 and 1
    with open(train_labels[0], encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) > 0
        for line in lines:
            parts = line.strip().split()
            assert len(parts) == 5
            cls_id = int(parts[0])
            assert cls_id in CAMERA_CLASSES
            coords = [float(p) for p in parts[1:]]
            for c in coords:
                assert 0.0 <= c <= 1.0, f"YOLO coordinate {c} must be normalized [0, 1]"


def test_night_infrared_transform():
    """Verify active infrared night transform produces valid monochrome image with sensor noise."""
    rgb_img = Image.new("RGB", (100, 100), color=(120, 180, 70))
    ir_img = apply_night_infrared_transform(rgb_img)

    assert ir_img.size == (100, 100)
    assert ir_img.mode == "RGB"
    arr = np.array(ir_img)
    # In monochrome conversion R, G, B channels should be identical
    assert np.allclose(arr[:, :, 0], arr[:, :, 1], atol=2)


def test_intrusion_rule_engine():
    """Verify temporal curfew, restricted zone rules, and wildlife privacy protection."""
    rule_engine = IntrusionRuleEngine(
        curfew_start="18:00",
        curfew_end="06:00",
        restricted_zones=["CORE_ZONE"],
    )

    # Test 1: Nighttime human in core zone -> Critical Alert
    night_ts = datetime(2026, 10, 26, 23, 30, tzinfo=UTC)
    res_human = rule_engine.evaluate_detection(
        class_name="person",
        confidence=0.91,
        timestamp=night_ts,
        location_zone="CORE_ZONE",
        coords=(77.20, 12.05),
    )
    assert res_human["is_alert"] is True
    assert res_human["alert_type"] == "CRITICAL_NIGHT_TRESPASS"
    assert res_human["threat_level"] == "CRITICAL"
    assert res_human["requires_immediate_dispatch"] is True

    # Test 2: Snare trap detected -> Poaching Hazard
    day_ts = datetime(2026, 10, 26, 11, 0, tzinfo=UTC)
    res_snare = rule_engine.evaluate_detection(
        class_name="snare_or_trap",
        confidence=0.85,
        timestamp=day_ts,
        location_zone="BUFFER_ZONE",
    )
    assert res_snare["is_alert"] is True
    assert res_snare["alert_type"] == "POACHING_TRAP_HAZARD"

    # Test 3: Animal detection -> Strictly obfuscate GPS coordinates for wildlife protection
    res_animal = rule_engine.evaluate_detection(
        class_name="animal",
        confidence=0.95,
        timestamp=night_ts,
        location_zone="CORE_ZONE",
        coords=(77.25, 12.10),
    )
    assert res_animal["is_wildlife_private"] is True
    assert res_animal["coordinates"] is None, "Animal coordinates must never be publicly exposed"
    assert res_animal["threat_level"] == "NONE"


def test_camera_trap_inference_engine(tmp_path: Path):
    """Verify inference engine processes image, tags rules, and creates evidence output."""
    test_img_path = tmp_path / "test_frame.jpg"
    img = Image.new("RGB", (320, 320), color=(50, 80, 50))
    img.save(test_img_path)

    evidence_dir = tmp_path / "evidence"
    detector = CameraTrapInferenceEngine(evidence_dir=str(evidence_dir))

    events = detector.process_image(test_img_path, camera_id="CAM_TEST_01")
    assert isinstance(events, list)

    evidence_files = list(evidence_dir.glob("*.jpg"))
    assert len(evidence_files) == 1
    assert evidence_files[0].name == "evidence_test_frame.jpg"
