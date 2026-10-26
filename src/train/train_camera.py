"""Training pipeline for camera-trap YOLO model (Person, Vehicle, Animal, Snare).

Fine-tunes YOLOv8n on camera-trap dataset with day/night simulations and evaluates mAP50.
"""

import json
import logging
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO

from src.data.camera_data import (
    CAMERA_CLASSES,
    generate_synthetic_camera_trap_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_camera_model(
    config_path: str = "configs/training.yaml",
    dataset_yaml_path: str = "data/samples/camera/camera_dataset.yaml",
    checkpoints_dir: str = "data/checkpoints",
    reports_dir: str = "reports/metrics",
    epochs_override: int | None = None,
) -> dict[str, Any]:
    """Train YOLOv8n on camera-trap dataset and evaluate mAP50."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["camera_detector"]

    dataset_yaml_p = Path(dataset_yaml_path)
    if not dataset_yaml_p.exists():
        logger.info("Camera dataset YAML not found. Generating synthetic camera-trap dataset.")
        dataset_yaml_p = generate_synthetic_camera_trap_dataset(dataset_yaml_p.parent)

    checkpoints_dir_p = Path(checkpoints_dir)
    checkpoints_dir_p.mkdir(parents=True, exist_ok=True)
    reports_dir_p = Path(reports_dir)
    reports_dir_p.mkdir(parents=True, exist_ok=True)

    epochs = epochs_override or 5  # Quick fine-tuning for demo/verification
    logger.info("Initializing YOLO model: %s for %d epochs", cfg["model"]["type"], epochs)

    # Initialize YOLOv8 nano
    model = YOLO(f"{cfg['model']['type']}.pt")

    # Train model
    model.train(
        data=str(dataset_yaml_p).replace("\\", "/"),
        epochs=epochs,
        imgsz=cfg["model"]["img_size"],
        batch=4,
        workers=0,  # Ensure Windows multiprocessing stability
        device="cpu",
        plots=False,
        save=True,
        project="data/checkpoints/yolo_runs",
        name="camera_trap_run",
        exist_ok=True,
    )

    # Validate model on validation split
    val_results = model.val(
        data=str(dataset_yaml_p).replace("\\", "/"),
        device="cpu",
        workers=0,
    )

    metrics_dict = {
        "mAP50": float(round(val_results.box.map50, 4)),
        "mAP50_95": float(round(val_results.box.map, 4)),
        "precision": float(round(val_results.box.mp, 4)),
        "recall": float(round(val_results.box.mr, 4)),
        "classes": {cls_id: name for cls_id, name in CAMERA_CLASSES.items()},
        "epochs": epochs,
    }

    # Save best checkpoint
    best_weights_path = Path("data/checkpoints/yolo_runs/camera_trap_run/weights/best.pt")
    target_best_path = checkpoints_dir_p / "camera_yolo_best.pt"
    if best_weights_path.exists():
        import shutil
        shutil.copy(best_weights_path, target_best_path)
        logger.info("Copied best weights to %s", target_best_path)
    else:
        # Save current model
        model.save(str(target_best_path))

    metrics_out = reports_dir_p / "camera_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    logger.info("Camera model training complete! Metrics: %s", metrics_dict)
    return metrics_dict


if __name__ == "__main__":
    train_camera_model(epochs_override=3)
