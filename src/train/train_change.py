"""Training pipeline for the Siamese U-Net change detection model.

Uses MLflow for experiment tracking, Dice + Focal loss, and spatial validation splits.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.models.change_detect import DiceFocalLoss, SiameseUNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ChangeDetectionDataset(Dataset):
    """Dataset for paired multi-temporal satellite tiles and segmentation masks."""

    def __init__(self, tile_records: list[dict[str, Any]], augment: bool = True):
        self.records = tile_records
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def _read_raster(self, path: str) -> np.ndarray:
        with rasterio.open(path) as src:
            return src.read()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        before_arr = self._read_raster(rec["before_path"])  # (10, H, W)
        after_arr = self._read_raster(rec["after_path"])    # (10, H, W)
        mask_arr = self._read_raster(rec["mask_path"])[0]   # (H, W)

        # Synchronous spatial augmentations
        if self.augment:
            if random.random() > 0.5:
                before_arr = np.flip(before_arr, axis=2).copy()
                after_arr = np.flip(after_arr, axis=2).copy()
                mask_arr = np.flip(mask_arr, axis=1).copy()
            if random.random() > 0.5:
                before_arr = np.flip(before_arr, axis=1).copy()
                after_arr = np.flip(after_arr, axis=1).copy()
                mask_arr = np.flip(mask_arr, axis=0).copy()

        t1 = torch.from_numpy(before_arr.astype(np.float32))
        t2 = torch.from_numpy(after_arr.astype(np.float32))
        target = torch.from_numpy(mask_arr.astype(np.int64))

        return t1, t2, target


def compute_metrics(preds: np.ndarray, targets: np.ndarray, num_classes: int = 4) -> dict[str, float]:
    """Compute per-class and mean Intersection over Union (IoU) and F1 score."""
    metrics = {}
    ious = []
    f1s = []

    for c in range(num_classes):
        p_c = (preds == c)
        t_c = (targets == c)
        intersection = np.logical_and(p_c, t_c).sum()
        union = np.logical_or(p_c, t_c).sum()

        iou = (intersection + 1e-6) / (union + 1e-6)
        ious.append(iou)

        precision = (intersection + 1e-6) / (p_c.sum() + 1e-6)
        recall = (intersection + 1e-6) / (t_c.sum() + 1e-6)
        f1 = (2 * precision * recall) / (precision + recall + 1e-6)
        f1s.append(f1)

        metrics[f"iou_class_{c}"] = float(iou)
        metrics[f"f1_class_{c}"] = float(f1)

    metrics["mean_iou"] = float(np.mean(ious))
    metrics["mean_f1"] = float(np.mean(f1s))
    # Change class IoU (classes 1, 2, 3)
    metrics["change_mean_iou"] = float(np.mean(ious[1:]))
    return metrics


def train_change_model(
    config_path: str = "configs/training.yaml",
    splits_path: str = "data/labels/spatial_splits.json",
    checkpoints_dir: str = "data/checkpoints",
    epochs_override: int | None = None,
) -> dict[str, Any]:
    """Train Siamese U-Net on satellite change pairs with spatial validation."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["change_detection"]

    set_seed(cfg["training"]["random_seed"])
    checkpoints_dir_p = Path(checkpoints_dir)
    checkpoints_dir_p.mkdir(parents=True, exist_ok=True)

    with open(splits_path, encoding="utf-8") as f:
        splits = json.load(f)

    train_ds = ChangeDetectionDataset(splits["train"], augment=True)
    val_ds = ChangeDetectionDataset(splits["val"], augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training Siamese U-Net on device: %s", device)

    model = SiameseUNet(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        base_channels=16,  # Efficient lightweight base
    ).to(device)

    loss_fn = DiceFocalLoss(
        num_classes=cfg["model"]["num_classes"],
        focal_weight=cfg["training"]["loss"]["focal_weight"],
        dice_weight=cfg["training"]["loss"]["dice_weight"],
        gamma=cfg["training"]["loss"]["gamma"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    num_epochs = epochs_override or cfg["training"]["num_epochs"]
    best_val_iou = -1.0
    best_metrics = {}

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []

        for t1, t2, target in train_loader:
            t1, t2, target = t1.to(device), t2.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(t1, t2)
            loss = loss_fn(logits, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for t1, t2, target in val_loader:
                t1, t2 = t1.to(device), t2.to(device)
                logits = model(t1, t2)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.append(preds.flatten())
                val_targets.append(target.numpy().flatten())

        val_preds_arr = np.concatenate(val_preds)
        val_targets_arr = np.concatenate(val_targets)
        metrics = compute_metrics(val_preds_arr, val_targets_arr, num_classes=cfg["model"]["num_classes"])
        avg_train_loss = float(np.mean(train_losses))

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Mean IoU: %.4f | Change IoU: %.4f",
            epoch,
            num_epochs,
            avg_train_loss,
            metrics["mean_iou"],
            metrics["change_mean_iou"],
        )

        if metrics["mean_iou"] > best_val_iou:
            best_val_iou = metrics["mean_iou"]
            best_metrics = metrics
            best_model_path = checkpoints_dir_p / "siamese_unet_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_iou": best_val_iou,
                    "config": cfg,
                },
                best_model_path,
            )
            logger.info("Saved new best checkpoint with Val IoU: %.4f to %s", best_val_iou, best_model_path)

    return {
        "best_val_iou": best_val_iou,
        "best_metrics": best_metrics,
        "checkpoint_path": str(checkpoints_dir_p / "siamese_unet_best.pt"),
    }


if __name__ == "__main__":
    train_change_model(epochs_override=5)
