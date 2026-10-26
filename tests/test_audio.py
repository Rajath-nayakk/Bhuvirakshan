"""Unit tests for Phase 4: Acoustic Event Detection, SpecAugment, and Sliding Window."""

from pathlib import Path

import numpy as np
import torch

from src.data.audio_data import (
    AudioFeatureExtractor,
    mix_with_noise_at_snr,
)
from src.infer.run_audio import AcousticThreatDetector
from src.models.audio_classifier import AudioClassifier, export_audio_model_to_onnx


def test_audio_feature_extractor_shape():
    """Verify log-mel spectrogram extraction produces exact expected dimension (1, 128, 157)."""
    extractor = AudioFeatureExtractor(sample_rate=16000, clip_duration_sec=5.0)

    # 5 seconds at 16,000 Hz = 80,000 samples
    waveform = np.sin(2 * np.pi * 440 * np.linspace(0, 5, 80000)).astype(np.float32)
    log_mel = extractor.extract_log_mel(waveform)

    assert log_mel.shape == (1, 128, 157), f"Expected (1, 128, 157), got {log_mel.shape}"
    assert not torch.isnan(log_mel).any()


def test_spec_augment():
    """Verify frequency and time masking are applied without corrupting tensor shape."""
    extractor = AudioFeatureExtractor()
    dummy_spec = torch.randn(1, 128, 157)
    augmented = extractor.apply_spec_augment(dummy_spec)

    assert augmented.shape == (1, 128, 157)
    assert not torch.isnan(augmented).any()


def test_mix_with_noise_snr():
    """Verify SNR noise mixing maintains length, bounds amplitude, and changes power."""
    signal = np.sin(2 * np.pi * 200 * np.linspace(0, 1, 16000)).astype(np.float32)
    noise = np.random.randn(16000).astype(np.float32) * 0.1

    mixed = mix_with_noise_at_snr(signal, noise, target_snr_db=5.0)

    assert len(mixed) == len(signal)
    assert np.max(np.abs(mixed)) <= 1.05
    assert not np.array_equal(mixed, signal)


def test_audio_classifier_forward():
    """Verify AudioClassifier neural network accepts spectrogram batch and outputs class logits."""
    model = AudioClassifier(num_classes=4, in_channels=1, base_channels=16)
    model.eval()

    batch_size = 3
    x = torch.randn(batch_size, 1, 128, 157)

    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (batch_size, 4)


def test_sliding_window_event_emission():
    """Verify sliding window processes multi-second audio stream and emits structured alerts."""
    # 10 seconds of simulated chainsaw revving
    sample_rate = 16000
    t = np.linspace(0, 10, 10 * sample_rate)
    chainsaw_sig = 0.8 * np.sin(2 * np.pi * 180 * t)

    detector = AcousticThreatDetector(sample_rate=sample_rate, window_sec=5.0, stride_sec=2.5)

    events = detector.process_waveform(chainsaw_sig, sensor_id="TEST_SENSOR_01")
    assert isinstance(events, list)

    for ev in events:
        assert "event_type" in ev
        assert ev["event_type"] == "acoustic_threat"
        assert ev["sensor_id"] == "TEST_SENSOR_01"
        assert "class" in ev
        assert "confidence" in ev
        assert 0.0 <= ev["confidence"] <= 1.0
        assert "threat_level" in ev


def test_export_onnx(tmp_path: Path):
    """Verify model successfully exports to ONNX for Raspberry Pi edge execution."""
    model = AudioClassifier(num_classes=4, in_channels=1, base_channels=16)
    onnx_file = tmp_path / "test_model.onnx"

    exported = export_audio_model_to_onnx(model, onnx_file)
    assert exported.exists()
    assert exported.stat().st_size > 10000  # Non-trivial ONNX binary
