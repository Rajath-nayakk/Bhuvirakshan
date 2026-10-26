"""Training pipeline for acoustic threat classifier (gunshots, chainsaws, engines).

Computes per-class precision/recall, evaluates false alarm rate per hour on ambient noise,
generates noise-robustness curves across variable SNR, and exports to ONNX.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import yaml
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Dataset

from src.data.audio_data import (
    AUDIO_CLASSES,
    AudioFeatureExtractor,
    generate_synthetic_acoustic_dataset,
    mix_with_noise_at_snr,
)
from src.models.audio_classifier import AudioClassifier, export_audio_model_to_onnx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AudioDataset(Dataset):
    """Dataset loading audio files, extracting log-mel spectrograms, and applying SpecAugment."""

    def __init__(
        self,
        file_records: list[dict[str, Any]],
        feature_extractor: AudioFeatureExtractor,
        ambient_noise_pool: list[np.ndarray] | None = None,
        augment: bool = True,
        snr_range: tuple[float, float] = (-5.0, 20.0),
    ):
        self.records = file_records
        self.extractor = feature_extractor
        self.ambient_noise_pool = ambient_noise_pool or []
        self.augment = augment
        self.snr_range = snr_range

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        waveform, _ = sf.read(rec["file_path"])
        cls_id = rec["class_id"]

        # If augment and we have ambient noise pool and this is not already ambient class
        if self.augment and self.ambient_noise_pool and cls_id != 3 and random.random() > 0.4:
            target_snr = random.uniform(*self.snr_range)
            noise = random.choice(self.ambient_noise_pool)
            waveform = mix_with_noise_at_snr(waveform, noise, target_snr)

        log_mel = self.extractor.extract_log_mel(waveform)

        if self.augment and random.random() > 0.3:
            log_mel = self.extractor.apply_spec_augment(log_mel)

        return log_mel, torch.tensor(cls_id, dtype=torch.long)


def evaluate_snr_robustness(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    extractor: AudioFeatureExtractor,
    ambient_noise: np.ndarray,
    snr_levels: list[float] | None = None,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Evaluate model accuracy across different SNR levels to test noise robustness."""
    if snr_levels is None:
        snr_levels = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    model.eval()
    device = device or torch.device("cpu")
    robustness = {}

    for snr in snr_levels:
        correct = 0
        total = 0
        for rec in records:
            cls_id = rec["class_id"]
            if cls_id == 3:  # Skip ambient class for SNR signal test
                continue
            wav, _ = sf.read(rec["file_path"])
            noisy_wav = mix_with_noise_at_snr(wav, ambient_noise, snr)
            spec = extractor.extract_log_mel(noisy_wav).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(spec)
                pred = torch.argmax(logits, dim=1).item()

            if pred == cls_id:
                correct += 1
            total += 1

        acc = float(correct / total) if total > 0 else 0.0
        robustness[f"snr_{int(snr)}_db"] = round(acc, 4)

    return robustness


def evaluate_false_alarm_rate(
    model: torch.nn.Module,
    ambient_audio: np.ndarray,
    extractor: AudioFeatureExtractor,
    sample_rate: int = 16000,
    window_sec: float = 5.0,
    stride_sec: float = 2.5,
    threat_threshold: float = 0.65,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Calculate false alarms per hour on continuous background forest recording."""
    model.eval()
    device = device or torch.device("cpu")

    window_samples = int(window_sec * sample_rate)
    stride_samples = int(stride_sec * sample_rate)
    total_samples = len(ambient_audio)

    total_hours = (total_samples / sample_rate) / 3600.0
    # If ambient audio is short, extrapolate to hourly rate
    if total_hours < 0.001:
        total_hours = 0.01  # Safe minimum for calculation

    false_alarms = 0
    start = 0

    while start + window_samples <= total_samples:
        clip = ambient_audio[start : start + window_samples]
        spec = extractor.extract_log_mel(clip).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(spec), dim=1).cpu().numpy()[0]

        # Threat classes are 0 (gunshot), 1 (chainsaw), 2 (vehicle)
        max_threat_conf = max(probs[0], probs[1], probs[2])
        if max_threat_conf > threat_threshold:
            false_alarms += 1

        start += stride_samples

    fa_per_hour = round(false_alarms / total_hours, 2)
    return {
        "false_alarms_total": false_alarms,
        "test_duration_hours": round(total_hours, 3),
        "false_alarms_per_hour": fa_per_hour,
    }


def train_audio_pipeline(
    config_path: str = "configs/training.yaml",
    audio_samples_dir: str = "data/samples/audio",
    reports_dir: str = "reports",
    checkpoints_dir: str = "data/checkpoints",
    epochs: int = 12,
) -> dict[str, Any]:
    """End-to-end training and evaluation pipeline for acoustic classifier."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["audio_classifier"]

    set_seed(cfg["training"]["random_seed"])
    audio_dir = Path(audio_samples_dir)
    checkpoints_dir_p = Path(checkpoints_dir)
    checkpoints_dir_p.mkdir(parents=True, exist_ok=True)
    reports_p = Path(reports_dir)
    metrics_p = reports_p / "metrics"
    figures_p = reports_p / "figures"
    metrics_p.mkdir(parents=True, exist_ok=True)
    figures_p.mkdir(parents=True, exist_ok=True)

    # Ensure sample audio clips exist
    if not any(audio_dir.glob("*/*.wav")):
        logger.info("Audio samples not found. Generating synthetic acoustic samples.")
        generate_synthetic_acoustic_dataset(output_dir=audio_dir, num_samples_per_class=15)

    # Collect audio file records
    records = []
    for cls_id, cls_name in AUDIO_CLASSES.items():
        cls_files = list((audio_dir / cls_name).glob("*.wav"))
        for f in cls_files:
            records.append({
                "file_path": str(f),
                "class_id": cls_id,
                "class_name": cls_name,
            })

    random.shuffle(records)
    split_idx = int(0.75 * len(records))
    train_records = records[:split_idx]
    val_records = records[split_idx:]

    extractor = AudioFeatureExtractor(
        sample_rate=cfg["features"]["sample_rate"],
        clip_duration_sec=cfg["features"]["clip_duration_sec"],
        n_fft=cfg["features"]["n_fft"],
        hop_length=cfg["features"]["hop_length"],
        n_mels=cfg["features"]["n_mels"],
        f_min=cfg["features"]["f_min"],
        f_max=cfg["features"]["f_max"],
    )

    # Load ambient noise clips for mixing
    ambient_wavs = []
    for r in [rec for rec in records if rec["class_id"] == 3]:
        w, _ = sf.read(r["file_path"])
        ambient_wavs.append(w)

    train_ds = AudioDataset(train_records, extractor, ambient_noise_pool=ambient_wavs, augment=True)
    val_ds = AudioDataset(val_records, extractor, augment=False)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioClassifier(num_classes=4, in_channels=1, base_channels=24).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(specs)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.to(device)
                logits = model(specs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.numpy())

        acc = float(np.mean(np.array(val_preds) == np.array(val_targets)))
        logger.info("Audio Epoch %d/%d | Train Loss: %.4f | Val Acc: %.4f", epoch, epochs, train_loss / len(train_loader), acc)

        if acc > best_acc:
            best_acc = acc
            best_ckpt = checkpoints_dir_p / "audio_classifier_best.pt"
            torch.save(model.state_dict(), best_ckpt)

    # Load best model for full evaluation
    best_ckpt = checkpoints_dir_p / "audio_classifier_best.pt"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # Evaluate per-class precision and recall on val set
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for specs, labels in val_loader:
            specs = specs.to(device)
            preds = torch.argmax(model(specs), dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(labels.numpy())

    cls_report = classification_report(
        val_targets,
        val_preds,
        target_names=[AUDIO_CLASSES[i] for i in range(4)],
        output_dict=True,
        zero_division=0,
    )

    # Evaluate SNR Noise Robustness Curve
    ambient_ref = ambient_wavs[0] if ambient_wavs else np.zeros(80000, dtype=np.float32)
    snr_levels = [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
    snr_curve = evaluate_snr_robustness(model, val_records, extractor, ambient_ref, snr_levels, device)

    # Evaluate False Alarms Per Hour
    long_ambient_audio = np.tile(ambient_ref, 20)  # Simulate extended ambient period
    fa_metrics = evaluate_false_alarm_rate(model, long_ambient_audio, extractor, device=device)

    # Plot Noise Robustness Curve
    fig, ax = plt.subplots(figsize=(6, 4))
    snr_vals = [int(k.split("_")[1]) for k in snr_curve.keys()]
    acc_vals = [snr_curve[k] * 100 for k in snr_curve.keys()]
    ax.plot(snr_vals, acc_vals, marker="o", color="#2563eb", linewidth=2.5, markersize=7)
    ax.set_xlabel("Signal-to-Noise Ratio (SNR dB)")
    ax.set_ylabel("Threat Classification Accuracy (%)")
    ax.set_title("Acoustic Model Noise Robustness Curve")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim([0, 105])
    plt.tight_layout()
    snr_plot_path = figures_p / "audio_snr_robustness.png"
    plt.savefig(snr_plot_path, dpi=150)
    plt.close()

    # Export to ONNX
    onnx_path = export_audio_model_to_onnx(model, checkpoints_dir_p / "audio_classifier.onnx")
    logger.info("Exported ONNX model to %s", onnx_path)

    # Save comprehensive metrics summary
    full_metrics = {
        "best_accuracy": best_acc,
        "classification_report": cls_report,
        "snr_robustness_curve": snr_curve,
        "false_alarm_rate": fa_metrics,
        "onnx_exported": str(onnx_path),
    }

    metrics_out = metrics_p / "audio_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)

    logger.info("Audio evaluation complete. Metrics: %s, Figure: %s", metrics_out, snr_plot_path)
    return full_metrics


if __name__ == "__main__":
    train_audio_pipeline(epochs=10)
