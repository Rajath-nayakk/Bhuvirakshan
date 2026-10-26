"""Sliding-window acoustic inference engine for edge deployment (Raspberry Pi / Jetson).

Monitors live microphone feeds or directory of WAV recordings, applies
5-second sliding windows with 50% overlap, and emits structured security alerts.

Edge Deployment on Raspberry Pi 4 / 5:
1. Install ONNX Runtime: `pip install onnxruntime`
2. Connect USB microphone or I2S MEMS sensor array (e.g., SPH0645).
3. Run: `python -m src.infer.run_audio --input-dir /mnt/audio_buffer --sensor-id NODE_01`
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from src.data.audio_data import AUDIO_CLASSES, AudioFeatureExtractor
from src.models.audio_classifier import AudioClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class AcousticThreatDetector:
    """Real-time sliding-window acoustic threat detection engine."""

    def __init__(
        self,
        checkpoint_path: str = "data/checkpoints/audio_classifier_best.pt",
        sample_rate: int = 16000,
        window_sec: float = 5.0,
        stride_sec: float = 2.5,
        confidence_threshold: float = 0.55,
        device: str | None = None,
    ):
        self.sample_rate = sample_rate
        self.window_samples = int(window_sec * sample_rate)
        self.stride_samples = int(stride_sec * sample_rate)
        self.confidence_threshold = confidence_threshold

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.extractor = AudioFeatureExtractor(sample_rate=sample_rate, clip_duration_sec=window_sec)

        self.model = AudioClassifier(num_classes=4, in_channels=1, base_channels=24).to(self.device)
        chk_p = Path(checkpoint_path)
        if chk_p.exists():
            self.model.load_state_dict(torch.load(chk_p, map_location=self.device))
            logger.info("Loaded acoustic model from %s", chk_p)
        else:
            logger.warning("Checkpoint %s not found. Operating with initialized model.", chk_p)
        self.model.eval()

    def process_waveform(
        self,
        waveform: np.ndarray,
        sensor_id: str = "AUDIO_SENSOR_CAUVERY_01",
        base_timestamp: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Process continuous 1D audio buffer with sliding window and emit threat events."""
        base_ts = base_timestamp or datetime.now(UTC)
        total_samples = len(waveform)
        events = []

        start = 0
        while start + self.window_samples <= total_samples:
            window_audio = waveform[start : start + self.window_samples]
            spec = self.extractor.extract_log_mel(window_audio).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(spec)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            predicted_cls_id = int(np.argmax(probs))
            confidence = float(probs[predicted_cls_id])
            predicted_class = AUDIO_CLASSES[predicted_cls_id]

            # Offset timestamp for this window
            window_offset_sec = start / self.sample_rate
            event_timestamp = datetime.fromtimestamp(
                base_ts.timestamp() + window_offset_sec, tz=UTC
            ).isoformat()

            # Ignore ambient forest noise; only emit actionable threats
            if predicted_class != "ambient_forest" and confidence >= self.confidence_threshold:
                threat_level = "CRITICAL" if predicted_class in ["gunshot", "chainsaw"] else "WARNING"
                event = {
                    "event_type": "acoustic_threat",
                    "sensor_id": sensor_id,
                    "timestamp": event_timestamp,
                    "class": predicted_class,
                    "confidence": round(confidence, 3),
                    "threat_level": threat_level,
                    "window_start_sec": round(window_offset_sec, 2),
                    "window_duration_sec": 5.0,
                    "all_probabilities": {
                        AUDIO_CLASSES[i]: round(float(probs[i]), 3) for i in range(4)
                    },
                }
                events.append(event)
                logger.warning(
                    "[ACOUSTIC ALERT] %s detected by %s (Confidence: %.1f%%, Time: %s)",
                    predicted_class.upper(),
                    sensor_id,
                    confidence * 100,
                    event_timestamp,
                )

            start += self.stride_samples

        return events

    def process_file(
        self,
        file_path: Path | str,
        sensor_id: str = "AUDIO_SENSOR_01",
    ) -> list[dict[str, Any]]:
        """Process an audio file and return detected acoustic events."""
        data, sr = sf.read(file_path)
        if sr != self.sample_rate:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=self.sample_rate)
        return self.process_waveform(data, sensor_id=sensor_id)


def run_batch_audio_inference(
    audio_dir: str = "data/samples/audio",
    output_events_path: str = "data/processed/acoustic_events.json",
) -> list[dict[str, Any]]:
    """Process directory of audio files and save consolidated event stream."""
    detector = AcousticThreatDetector()
    audio_dir_p = Path(audio_dir)
    all_events = []

    files = list(audio_dir_p.glob("*/*.wav"))
    for f in files:
        events = detector.process_file(f, sensor_id=f"MIC_{f.parent.name.upper()}_01")
        all_events.extend(events)

    out_p = Path(output_events_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as dst:
        json.dump(all_events, dst, indent=2)

    logger.info("Batch audio inference complete. %d events saved to %s", len(all_events), out_p)
    return all_events


if __name__ == "__main__":
    run_batch_audio_inference()
