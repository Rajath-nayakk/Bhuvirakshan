"""Audio data pipeline: feature extraction, log-mel spectrograms, SpecAugment, and SNR noise mixing.

Handles 5-second audio clips at 16kHz, computes 128-mel band spectrograms,
and generates synthetic physical acoustic events for offline benchmarking.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.transforms as T

logger = logging.getLogger(__name__)

AUDIO_CLASSES = {
    0: "gunshot",
    1: "chainsaw",
    2: "vehicle",
    3: "ambient_forest",
}


class AudioFeatureExtractor:
    """Extracts log-mel spectrograms and applies SpecAugment."""

    def __init__(
        self,
        sample_rate: int = 16000,
        clip_duration_sec: float = 5.0,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 50.0,
        f_max: float = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.clip_samples = int(sample_rate * clip_duration_sec)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        self.freq_mask = T.FrequencyMasking(freq_mask_param=16)
        self.time_mask = T.TimeMasking(time_mask_param=20)

    def extract_log_mel(self, waveform: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Convert 1D audio waveform into 2D log-mel spectrogram tensor.

        Input: (samples,) or (1, samples)
        Output: Tensor of shape (1, n_mels, time_steps)
        """
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Pad or crop to exact clip duration
        if waveform.size(-1) < self.clip_samples:
            pad_amount = self.clip_samples - waveform.size(-1)
            waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
        elif waveform.size(-1) > self.clip_samples:
            waveform = waveform[:, : self.clip_samples]

        # Compute mel power spectrogram and convert to decibels
        mel = self.mel_transform(waveform)
        log_mel = torch.log(torch.clamp(mel, min=1e-6))
        # Normalize
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        return log_mel

    def apply_spec_augment(self, log_mel: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment (frequency and time masking)."""
        augmented = self.freq_mask(log_mel)
        augmented = self.time_mask(augmented)
        return augmented


def mix_with_noise_at_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float,
) -> np.ndarray:
    """Mix audio signal with background environmental noise at target SNR (dB).

    Formula:
        SNR_dB = 10 * log10(P_signal / P_noise)
        scaled_noise = noise * sqrt(P_signal / (10^(SNR_dB/10) * P_noise))
    """
    sig_len = len(signal)
    if len(noise) < sig_len:
        reps = int(np.ceil(sig_len / len(noise)))
        noise = np.tile(noise, reps)[:sig_len]
    else:
        noise = noise[:sig_len]

    p_sig = np.mean(signal**2) + 1e-9
    p_noise = np.mean(noise**2) + 1e-9

    snr_linear = 10.0 ** (target_snr_db / 10.0)
    scale = np.sqrt(p_sig / (snr_linear * p_noise))
    mixed = signal + scale * noise

    # Normalize to avoid clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 1.0:
        mixed = mixed / max_val
    return mixed.astype(np.float32)


def generate_synthetic_acoustic_dataset(
    output_dir: Path | str = "data/samples/audio",
    num_samples_per_class: int = 15,
    sample_rate: int = 16000,
    duration_sec: float = 5.0,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Generate physically synthesized acoustic signals for gunshots, chainsaws, engines, and ambient noise.

    - Gunshot: Short explosive acoustic impulse (5-20 ms) with steep rise, reverberation tail and high-frequency blast.
    - Chainsaw: Fast periodic two-stroke engine harmonics (120-350 Hz base harmonic series) with revving modulation.
    - Vehicle: Low-frequency heavy diesel combustion thrum (40-120 Hz) and broadband rolling tire road noise.
    - Ambient Forest: Pink noise wind turbulence, intermittent cicada/insect chirps (4-6 kHz), and gentle river flow.
    """
    rng = np.random.default_rng(seed)
    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)
    records = []

    for cls_id, cls_name in AUDIO_CLASSES.items():
        cls_dir = output_dir_p / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples_per_class):
            if cls_name == "gunshot":
                # Gunshot: Sharp impulse blast at ~1.5s + exponential echo decay
                sig = rng.normal(0, 0.05, n_samples)
                blast_idx = int(sample_rate * (1.2 + 0.5 * rng.random()))
                blast_dur = int(sample_rate * 0.4)
                t_blast = np.linspace(0, 1, blast_dur)
                impulse = np.exp(-12.0 * t_blast) * np.sin(2 * np.pi * 850 * t_blast)
                noise_burst = rng.normal(0, 0.9, blast_dur) * np.exp(-15.0 * t_blast)
                sig[blast_idx : blast_idx + blast_dur] += impulse + noise_burst
                # Low-pass ground echo
                sig += np.convolve(sig, np.ones(20) / 20.0, mode="same") * 0.3

            elif cls_name == "chainsaw":
                # Chainsaw: 2-stroke engine harmonic buzz + revving throttle
                f0 = 160.0 + 30.0 * np.sin(2 * np.pi * 0.8 * t)  # 130-190 Hz revving
                harmonics = np.zeros(n_samples)
                for h in range(1, 8):
                    harmonics += (1.0 / h) * np.sin(2 * np.pi * h * f0 * t)
                buzz = harmonics + rng.normal(0, 0.15, n_samples)
                sig = buzz * 0.6

            elif cls_name == "vehicle":
                # Heavy diesel engine low idle: 45 Hz fundamental + rumble
                engine = np.sin(2 * np.pi * 48 * t) + 0.6 * np.sin(2 * np.pi * 96 * t)
                rumble = rng.normal(0, 0.25, n_samples)
                rumble = np.convolve(rumble, np.ones(50) / 50.0, mode="same")
                sig = (engine * 0.5 + rumble * 0.5)

            else:  # ambient_forest
                # Ambient wind pink noise + intermittent bird chirps
                wind = rng.normal(0, 0.1, n_samples)
                wind = np.convolve(wind, np.ones(80) / 80.0, mode="same")
                # Insect / cicada 5kHz carrier
                insects = 0.04 * np.sin(2 * np.pi * 5200 * t) * (np.sin(2 * np.pi * 8 * t) > 0)
                sig = wind + insects

            # Normalize peak
            sig = sig / (np.max(np.abs(sig)) + 1e-6) * 0.9
            sig = sig.astype(np.float32)

            file_path = cls_dir / f"{cls_name}_{i:03d}.wav"
            sf.write(file_path, sig, sample_rate)

            records.append({
                "file_path": str(file_path),
                "class_id": cls_id,
                "class_name": cls_name,
                "duration_sec": duration_sec,
                "sample_rate": sample_rate,
            })

    logger.info("Generated %d synthetic audio clips across %d classes in %s", len(records), len(AUDIO_CLASSES), output_dir)
    return {"samples": records}
