"""Camera-trap data preparation, LILA BC format converter, and YOLO annotation generator.

Classes:
  0: person
  1: vehicle
  2: animal
  3: snare_or_trap
"""

import logging
import random
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

CAMERA_CLASSES = {
    0: "person",
    1: "vehicle",
    2: "animal",
    3: "snare_or_trap",
}


def create_yolo_dataset_yaml(
    dataset_dir: Path | str,
    output_yaml_path: Path | str = "data/samples/camera/camera_dataset.yaml",
) -> Path:
    """Create YOLOv8 dataset configuration YAML file."""
    dataset_dir_p = Path(dataset_dir).resolve()
    output_yaml_p = Path(output_yaml_path)
    output_yaml_p.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "path": str(dataset_dir_p).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/val",
        "names": {cls_id: name for cls_id, name in CAMERA_CLASSES.items()},
        "nc": len(CAMERA_CLASSES),
    }

    with open(output_yaml_p, "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

    return output_yaml_p


def generate_synthetic_camera_trap_dataset(
    output_dir: Path | str = "data/samples/camera",
    num_train: int = 24,
    num_val: int = 8,
    img_size: tuple[int, int] = (640, 640),
    seed: int = 42,
) -> Path:
    """Generate physically simulated camera-trap imagery with YOLO annotations.

    Simulates:
    - Forest backgrounds (daylight foliage and nighttime IR illuminator monochrome)
    - Persons (poachers / illegal miners)
    - Vehicles (illegal sand tipper trucks / offroad motorcycles)
    - Wildlife (spotted deer, elephants, leopards)
    - Wire snares / traps anchored to tree trunks
    """
    random.seed(seed)
    np.random.seed(seed)

    out_dir = Path(output_dir)
    images_train = out_dir / "images" / "train"
    images_val = out_dir / "images" / "val"
    labels_train = out_dir / "labels" / "train"
    labels_val = out_dir / "labels" / "val"

    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    w, h = img_size

    def make_sample(idx: int, is_val: bool):
        is_night = random.random() > 0.4
        img_dir = images_val if is_val else images_train
        lbl_dir = labels_val if is_val else labels_train

        # Base background texture
        if is_night:
            # Infrared night background (grayscale with vignetting & sensor noise)
            bg_arr = np.random.normal(50, 15, (h, w)).clip(0, 255).astype(np.uint8)
            # Add center IR spotlight flash gradient
            y, x = np.ogrid[:h, :w]
            center_dist = np.sqrt((x - w / 2) ** 2 + (y - h / 2) ** 2)
            vignette = np.clip(1.2 - (center_dist / (w * 0.7)), 0.3, 1.3)
            bg_arr = np.clip(bg_arr * vignette, 10, 240).astype(np.uint8)
            img = Image.fromarray(bg_arr, mode="L").convert("RGB")
        else:
            # Daylight forest green/brown tones
            r = np.random.normal(65, 15, (h, w)).clip(0, 255)
            g = np.random.normal(95, 20, (h, w)).clip(0, 255)
            b = np.random.normal(55, 15, (h, w)).clip(0, 255)
            bg_arr = np.stack([r, g, b], axis=-1).astype(np.uint8)
            img = Image.fromarray(bg_arr, mode="RGB")

        draw = ImageDraw.Draw(img)

        # Draw foliage/trees
        for _ in range(5):
            tx = random.randint(0, w)
            draw.rectangle([tx - 15, 0, tx + 15, h], fill=(30, 25, 20) if not is_night else (35, 35, 35))

        # Select object class to place
        cls_id = random.choice([0, 1, 2, 3])
        labels = []

        if cls_id == 0:  # Person
            pw = random.randint(int(w * 0.1), int(w * 0.25))
            ph = random.randint(int(h * 0.25), int(h * 0.5))
            px = random.randint(5, max(6, w - pw - 5))
            py = random.randint(5, max(6, h - ph - 5))
            color = (180, 160, 140) if not is_night else (170, 170, 170)
            draw.rectangle([px, py, px + pw, py + ph], fill=color, outline=(0, 0, 0))
            # Head
            head_h = max(5, ph // 5)
            draw.ellipse([px + pw // 4, max(0, py - head_h), px + 3 * pw // 4, py + head_h // 2], fill=(200, 180, 160) if not is_night else (190, 190, 190))
            labels.append((cls_id, (px + pw / 2) / w, (py + ph / 2) / h, pw / w, ph / h))

        elif cls_id == 1:  # Vehicle (sand truck / jeep)
            vw = random.randint(int(w * 0.25), int(w * 0.45))
            vh = random.randint(int(h * 0.15), int(h * 0.35))
            vx = random.randint(5, max(6, w - vw - 5))
            vy = random.randint(5, max(6, h - vh - 5))
            color = (190, 120, 40) if not is_night else (160, 160, 160)
            draw.rectangle([vx, vy, vx + vw, vy + vh], fill=color, outline=(20, 20, 20))
            labels.append((cls_id, (vx + vw / 2) / w, (vy + vh / 2) / h, vw / w, vh / h))

        elif cls_id == 2:  # Animal (wild deer / elephant)
            aw = random.randint(int(w * 0.15), int(w * 0.35))
            ah = random.randint(int(h * 0.15), int(h * 0.3))
            ax = random.randint(5, max(6, w - aw - 5))
            ay = random.randint(5, max(6, h - ah - 5))
            color = (130, 85, 45) if not is_night else (120, 120, 120)
            draw.ellipse([ax, ay, ax + aw, ay + ah], fill=color)
            labels.append((cls_id, (ax + aw / 2) / w, (ay + ah / 2) / h, aw / w, ah / h))

        else:  # Snare / Trap
            sw = random.randint(int(w * 0.08), int(w * 0.18))
            sh = random.randint(int(h * 0.08), int(h * 0.18))
            sx = random.randint(5, max(6, w - sw - 5))
            sy = random.randint(5, max(6, h - sh - 5))
            outline_color = (220, 220, 220) if not is_night else (200, 200, 200)
            draw.ellipse([sx, sy, sx + sw, sy + sh], outline=outline_color, width=3)
            labels.append((cls_id, (sx + sw / 2) / w, (sy + sh / 2) / h, sw / w, sh / h))

        # Save image and label
        file_prefix = f"frame_{idx:04d}_{'night' if is_night else 'day'}"
        img_path = img_dir / f"{file_prefix}.jpg"
        lbl_path = lbl_dir / f"{file_prefix}.txt"

        img.save(img_path, quality=90)
        with open(lbl_path, "w", encoding="utf-8") as f:
            for c, x_c, y_c, bw, bh in labels:
                f.write(f"{c} {x_c:.5f} {y_c:.5f} {bw:.5f} {bh:.5f}\n")

    for i in range(num_train):
        make_sample(i, is_val=False)
    for i in range(num_val):
        make_sample(i + num_train, is_val=True)

    dataset_yaml = create_yolo_dataset_yaml(out_dir, out_dir / "camera_dataset.yaml")
    logger.info("Created camera-trap dataset with %d train, %d val samples. YAML: %s", num_train, num_val, dataset_yaml)
    return dataset_yaml
