"""Script to generate exploratory and results notebooks."""

import json
from pathlib import Path

nb_dir = Path("notebooks")
nb_dir.mkdir(parents=True, exist_ok=True)

# 01_explore_data.ipynb
nb_01 = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Forest & River Guardian: Exploratory Data Analysis\n",
                "This notebook explores multi-temporal satellite imagery (Sentinel-2 and Sentinel-1 SAR),\n",
                "acoustic audio samples, and camera-trap frames."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import rasterio\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from pathlib import Path\n",
                "\n",
                "# Inspect sample satellite tile\n",
                "tile_path = Path('../data/samples/satellite/tile_000_2023_before.tif')\n",
                "if tile_path.exists():\n",
                "    with rasterio.open(tile_path) as src:\n",
                "        print('Tile CRS:', src.crs)\n",
                "        print('Band Count:', src.count)\n",
                "        print('Dimensions:', src.shape)\n",
                "        b4 = src.read(3)\n",
                "        b8 = src.read(4)\n",
                "        ndvi = (b8 - b4) / (b8 + b4 + 1e-6)\n",
                "        \n",
                "    plt.figure(figsize=(6, 5))\n",
                "    plt.imshow(ndvi, cmap='RdYlGn')\n",
                "    plt.title('NDVI Vegetation Index')\n",
                "    plt.colorbar(label='NDVI')\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Inspect Audio Spectrogram\n",
                "import soundfile as sf\n",
                "import torch\n",
                "from src.data.audio_data import AudioFeatureExtractor\n",
                "\n",
                "wav_path = Path('../data/samples/audio/gunshot/gunshot_000.wav')\n",
                "if wav_path.exists():\n",
                "    data, sr = sf.read(str(wav_path))\n",
                "    extractor = AudioFeatureExtractor()\n",
                "    spec = extractor.extract_log_mel(torch.from_numpy(data).float())\n",
                "    print('Spectrogram Shape:', spec.shape)\n",
                "    \n",
                "    plt.figure(figsize=(7, 3))\n",
                "    plt.imshow(spec.squeeze().numpy(), origin='lower', aspect='auto', cmap='magma')\n",
                "    plt.title('Gunshot Audio Spectrogram')\n",
                "    plt.colorbar(label='Log Mel Power')\n",
                "    plt.show()"
            ]
        }
    ],
    "metadata": {
        "language_info": {"name": "python", "version": "3.11"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(nb_dir / "01_explore_data.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb_01, f, indent=2)

# 02_results.ipynb
nb_02 = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Forest & River Guardian: Results, Metrics & Visualizations\n",
                "Analysis of change-detection models, acoustic classifier robustness, and 500m risk fusion."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [],
            "source": [
                "import json\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "from pathlib import Path\n",
                "\n",
                "# Load Evaluation Metrics\n",
                "metrics_dir = Path('../reports/metrics')\n",
                "with open(metrics_dir / 'change_detection_metrics.json') as f:\n",
                "    change_metrics = json.load(f)\n",
                "with open(metrics_dir / 'audio_metrics.json') as f:\n",
                "    audio_metrics = json.load(f)\n",
                "with open(metrics_dir / 'camera_metrics.json') as f:\n",
                "    camera_metrics = json.load(f)\n",
                "\n",
                "print('=== Change Detection Comparison ===')\n",
                "print(pd.DataFrame(change_metrics['model_comparison']).T)\n",
                "\n",
                "print('\\n=== Camera Trap mAP@0.50 ===', camera_metrics['mAP50'])\n",
                "print('=== Audio Classifier Accuracy ===', audio_metrics['best_accuracy'])"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot SNR Robustness Curve\n",
                "snr_data = audio_metrics['snr_robustness_curve']\n",
                "snrs = [-10, -5, 0, 5, 10, 15, 20]\n",
                "accs = [snr_data[f'snr_{s}_db'] for s in snrs]\n",
                "\n",
                "plt.figure(figsize=(7, 4))\n",
                "plt.plot(snrs, [a * 100 for a in accs], marker='o', linewidth=2, color='#10b981')\n",
                "plt.title('Audio Classifier Noise Robustness Across SNR (dB)')\n",
                "plt.xlabel('Signal-to-Noise Ratio (dB)')\n",
                "plt.ylabel('Classification Accuracy (%)')\n",
                "plt.grid(True, linestyle='--', alpha=0.5)\n",
                "plt.ylim(30, 105)\n",
                "plt.show()"
            ]
        }
    ],
    "metadata": {
        "language_info": {"name": "python", "version": "3.11"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(nb_dir / "02_results.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb_02, f, indent=2)

print("Generated notebooks successfully.")
