# Forest & River Guardian: Implementation Plan

An AI system detecting illegal sand mining, illegal tree felling, and poaching activity along river corridors and protected forests, providing enforcement teams with an explainable, ranked daily patrol plan.

---

## Phase Status Summary

- [x] **Phase 0: Project Setup & Repository Skeleton**
- [x] **Phase 1: Satellite Data Pipeline (Earth Engine, Sentinel-1/2, GeoTIFF Tiling)**
- [x] **Phase 2: Labeling System & Spatial Partitioning**
- [x] **Phase 3: Change-Detection Deep Learning Model (Siamese U-Net)**
- [ ] **Phase 4: Acoustic Event Detection (Gunshots, Chainsaws, Engines)**
- [ ] **Phase 5: Camera-Trap Vision Detector (YOLO Intrusion Engine)**
- [ ] **Phase 6: Geospatial Risk Fusion & Explainable Ranking Engine**
- [ ] **Phase 7: Production Backend API & Secure Ingestion**
- [ ] **Phase 8: Interactive Enforcement Dashboard**
- [ ] **Phase 9: Comprehensive Evaluation, Ablation Studies & Final Report**
- [ ] **Phase 10: Packaging, Containerization & Offline Demo Mode**

---

## Detailed Phases & Acceptance Criteria

### Phase 0: Project Setup & Repository Skeleton
- [x] Create standardized repository layout (`src/`, `configs/`, `data/`, `frontend/`, `tests/`, `scripts/`, `reports/`, `notebooks/`).
- [x] Setup Python 3.11 virtual environment with `uv` and pinned dependencies (`requirements.txt`).
- [x] Configure `pyproject.toml` / `ruff.toml` for linting and code formatting.
- [x] Setup `pytest.ini` with unit test structure.
- [x] Provide `.env.example` and `.gitignore`.
- [x] Create base configurations: `configs/region.yaml`, `configs/training.yaml`, `configs/risk.yaml`.
- [x] **Acceptance Criteria**: Virtual environment active, all core imports verified without error, clean linting pass with `ruff check .`, test runner runs.

### Phase 1: Satellite Data Pipeline
- [x] Implement `src/data/earth_engine.py`: EE authentication, AOI loader from `configs/region.yaml`.
- [x] Fetch Sentinel-2 Surface Reflectance (harmonized cloud masking via SCL / QA60) and Sentinel-1 GRD (VV/VH, speckle filtered).
- [x] Compute monthly median composites and before/after temporal pairs across 10m spatial resolution.
- [x] Band stack calculation: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2), NDVI, NDWI, S1 VV, S1 VH.
- [x] Implement GeoTIFF exporter preserving affine geotransform, CRS, and nodata masks.
- [x] Build offline cached sample generator in `data/samples/` for seamless demo/testing without EE credentials.
- [x] **Acceptance Criteria**: Unit tests verifying tile dimensions (256x256), correct CRS and transform preservation, non-empty reflectance values, and graceful fallback to offline mode when credentials are not present.

### Phase 2: Labeling System & Spatial Partitioning
- [x] Implement `src/data/labeling.py` for mask generation and label management.
- [x] Define standardized class hierarchy:
  - 0: Background (Stable forest, water, established settlements)
  - 1: Sand Mining / Riverbed Extraction
  - 2: Forest Clearing / Illegal Logging
  - 3: New Track / Access Road
- [x] GeoJSON/Shapefile importer to rasterize ground truth or reported GPS incident points into raster masks.
- [x] Rule-based weak supervision generator (NDVI loss drop $>0.25$, MNDWI / Bare Soil index shifts inside riverbed buffer) marked with `weak_label=True`.
- [x] Spatial cross-validation / train-val-test partitioning ensuring strict non-overlapping geographical blocks to prevent spatial autocorrelation leakage.
- [x] **Acceptance Criteria**: Tests verifying mask rasterization alignment, weak label generation logic, and non-overlapping spatial boundaries between train/val/test splits.

### Phase 3: Change-Detection Deep Learning Model
- [x] Implement Siamese U-Net architecture (`src/models/change_detect.py`) using `segmentation-models-pytorch` backbone with difference & concatenation feature bridges.
- [x] Implement Combined Dice + Focal Loss to handle severe class imbalance (small illegal clearings vs large background).
- [x] Coupled spatial augmentations (identical affine, flip, and rotation transforms on pre- and post-event images).
- [x] Time-series seasonal change suppression filter comparing current diff against historical same-season baseline to suppress false positives from natural phenology / dry-season leaf drop.
- [x] Training script (`src/train/train_change.py`) with MLflow experiment logging, validation IoU tracking, and checkpointing.
- [x] Vectorization pipeline (`src/infer/run_change.py`) converting raster prediction masks into GeoJSON polygons with area (ha) and mean confidence.
- [x] Quantitative baseline comparison: Deep Siamese U-Net vs standard NDVI threshold differencing.
- [x] **Acceptance Criteria**: F1/IoU metrics calculated on held-out spatial test set, baseline comparison table generated, GeoJSON polygon output verified against shape & CRS standards.

### Phase 4: Acoustic Event Detection (Gunshots, Chainsaws, Engines)
- [ ] Implement audio feature pipeline (`src/data/audio_data.py`): 5-second windowing, log-mel spectrogram extraction (128 mel bins, 16kHz).
- [ ] Acoustic data augmentation: SpecAugment (time/frequency masking) and realistic background environmental noise mixing at variable SNR (-5 dB to 20 dB).
- [ ] Acoustic classifier (`src/models/audio_classifier.py`): Efficient CNN classifier fine-tuned for classes: `gunshot`, `chainsaw`, `vehicle_engine`, `ambient_forest`.
- [ ] Training script (`src/train/train_audio.py`) with per-class precision/recall, confusion matrix, and SNR degradation curves.
- [ ] Stream/file sliding window inference engine (`src/infer/run_audio.py`) emitting structured acoustic alarm events (timestamp, sensor_id, class, confidence).
- [ ] Edge deployment artifact: ONNX / TFLite export with benchmark inference latency.
- [ ] **Acceptance Criteria**: Validation accuracy > 88% on acoustic events, false alarms per hour evaluated on continuous ambient noise, ONNX model generated and verified.

### Phase 5: Camera-Trap Vision Detector (YOLO Intrusion Engine)
- [ ] Dataset formatting and conversion script (`src/data/camera_data.py`) supporting standard camera-trap formats (classes: `person`, `vehicle`, `animal`, `snare_or_trap`).
- [ ] Train/fine-tune YOLO model (`src/models/camera_detector.py`, `src/train/train_camera.py`) with day/night thermal simulation transforms.
- [ ] Temporal & spatial rule engine: person or vehicle detected during curfew hours (night) or inside restricted core zones flags critical intrusion alerts.
- [ ] Batch & stream inference pipeline (`src/infer/run_camera.py`) producing bounding-box annotated evidence frames and JSON detection records.
- [ ] **Acceptance Criteria**: mAP50 evaluation on held-out camera trap test split, intrusion rule triggering verified, annotated evidence image output saved.

### Phase 6: Geospatial Risk Fusion & Explainable Ranking Engine
- [ ] Implement `src/risk/grid.py`: Create regular 500m x 500m hexagonal or square grid cells across the study region.
- [ ] Implement `src/risk/fusion.py`: Multi-source bayesian/weighted daily fusion combining:
  1. Satellite change detection area (ha) and confidence.
  2. Acoustic alerts (gunshot/chainsaw count and recency).
  3. Camera trap intrusion events (humans/vehicles).
  4. Proximity penalty: distance to river channels, forest boundaries, access roads.
  5. Historical incident density / ranger patrol track coverage.
- [ ] Gradient-boosted ranking model support (`LightGBM` / `RandomForest`) trained on ranger feedback labels when available.
- [ ] Implement `src/risk/ranking.py`: Rank cells daily and generate natural-language patrol rationales (e.g., "Cell #142: High Risk (0.89) - 1.4 ha riverbed extraction + 2 nighttime vehicle detections").
- [ ] **Acceptance Criteria**: Deterministic grid generation over AOI, risk score normalization [0, 1], explainability string generation for Top-N patrol recommendations.

### Phase 7: Production Backend API & Secure Ingestion
- [ ] FastAPI application in `src/api/main.py` with SQLite relational database (`src/api/db.py`).
- [ ] Role-Based Access Control (RBAC): JWT authentication with `admin` and `ranger` roles (`src/api/auth.py`).
- [ ] Privacy & Safety Guard: Animal sightings and sensitive coordinates obfuscated/restricted from unauthorized endpoints.
- [ ] Core endpoints:
  - `POST /api/v1/auth/login`: Issue secure JWT tokens.
  - `GET /api/v1/alerts/top`: Retrieve daily Top-N ranked patrol cells with evidence.
  - `GET /api/v1/cells/{id}`: Detailed cell history, sensor logs, and change overlays.
  - `GET /api/v1/detections`: Filterable detections (satellite, audio, camera).
  - `POST /api/v1/events`: Ingestion endpoint for remote IoT/acoustic/camera sensors.
  - `POST /api/v1/feedback`: Ranger field patrol validation (confirmed true alert vs false alarm).
- [ ] Comprehensive pytest test suite covering authentication, RBAC, input validation, and alert filtering.
- [ ] **Acceptance Criteria**: OpenAPI schema generated, all test client requests passing with appropriate status codes (200, 401, 403, 422).

### Phase 8: Enforcement Dashboard
- [ ] Modern, high-performance web dashboard in `frontend/` (responsive for desktop and mobile tablets).
- [ ] Interactive map (Leaflet / MapLibre) with satellite imagery, 500m risk grid choropleth, detected change polygons, sensor markers.
- [ ] Daily Top-N patrol inspection panel with evidence cards:
  - Optical & Radar before/after comparison slider.
  - Acoustic spectrogram player & detection waveform.
  - Camera-trap evidence frame with bounding boxes.
- [ ] One-click Ranger Feedback workflow (Confirm Alert / False Positive) sending real-time updates to backend.
- [ ] Layer control toggles (Satellite Changes, Sensor Locations, Riverbed Buffers, Curfew Zones).
- [ ] **Acceptance Criteria**: Smooth rendering of GeoJSON grid, functional before/after image comparison, feedback button updates DB status dynamically, JWT session management.

### Phase 9: Evaluation, Ablation Studies & Final Report
- [ ] Execute full end-to-end pipeline on held-out test region and timeline.
- [ ] Generate comprehensive evaluation artifacts in `reports/`:
  - Per-model classification & detection metrics.
  - Siamese U-Net vs NDVI baseline comparison table.
  - Audio model noise-robustness curves across varying SNR.
  - Ablation study: Optical-only vs Optical + Sentinel-1 Radar vs Seasonal Filtering.
- [ ] Author `reports/final_report.md` detailing methodology, empirical results, ethical safeguards, and deployment guidelines.
- [ ] **Acceptance Criteria**: Complete reproducible report with real numeric metrics, confusion matrix plots, and ablation tables.

### Phase 10: Packaging, Containerization & Offline Demo Mode
- [ ] Multi-stage `Dockerfile` and `docker-compose.yml` for zero-configuration deployment.
- [ ] Unified pipeline runner script `scripts/run_pipeline.sh`.
- [ ] Comprehensive `README.md` with architectural Mermaid diagrams, quickstart instructions, and API references.
- [ ] One-command Offline Demo Mode running entirely on verified sample data without requiring cloud credentials.
- [ ] **Acceptance Criteria**: Container builds successfully, `scripts/run_pipeline.sh` executes clean demonstration, end-to-end demo launches dashboard seamlessly.
