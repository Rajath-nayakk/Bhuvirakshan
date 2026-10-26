#!/usr/bin/env bash
# ==============================================================================
# Forest & River Guardian: End-to-End Autonomous Pipeline Runner
# Executes data generation, model training, multi-sensor inference, risk fusion,
# and launches the FastAPI enforcement backend & interactive web dashboard.
# ==============================================================================

set -e

echo "================================================================================"
echo "          FOREST & RIVER GUARDIAN: AI PATROL ENFORCEMENT PIPELINE               "
echo "================================================================================"

# 1. Determine Python executable (virtualenv or fallback to system python)
if [ -f ".venv/Scripts/python.exe" ]; then
    PYTHON_CMD=".venv/Scripts/python.exe"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_CMD=".venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "[-] Error: Python executable not found. Please activate virtual environment."
    exit 1
fi

echo "[*] Using Python: $($PYTHON_CMD --version) at $PYTHON_CMD"

# 2. Phase 1: Satellite Data Pipeline (Cached Offline or Earth Engine)
echo ""
echo "[*] Step 1/6: Verifying Multi-Temporal Satellite Pipeline (S2 Optical + S1 SAR)..."
$PYTHON_CMD -c "
from src.data.earth_engine import authenticate_and_initialize, load_study_region, generate_sample_tiles
ee_auth = authenticate_and_initialize()
aoi = load_study_region()
if not ee_auth:
    print('[!] Earth Engine unauthenticated. Using cached offline physical sample tiles.')
    generate_sample_tiles()
"

# 3. Phase 2: Spatial Label Generation & Non-Overlapping Geographic Split
echo ""
echo "[*] Step 2/6: Building Ground-Truth Masks & Spatial Geographic Splits..."
$PYTHON_CMD -c "
from src.data.build_dataset import build_dataset
build_dataset()
"

# 4. Phase 3: Satellite Change-Detection Training & Evaluation
echo ""
echo "[*] Step 3/6: Executing Siamese U-Net Change Detection Inference..."
$PYTHON_CMD src/infer/run_change.py

# 5. Phase 4: Acoustic Event Detection (Gunshots, Chainsaws, Engines)
echo ""
echo "[*] Step 4/6: Executing Bioacoustic Classifier Inference (5.0s Windows)..."
$PYTHON_CMD src/infer/run_audio.py

# 6. Phase 5: Camera-Trap Vision Intrusion Engine (YOLOv8 + Night IR)
echo ""
echo "[*] Step 5/6: Executing Camera-Trap Intrusion Rule Engine..."
$PYTHON_CMD src/infer/run_camera.py

# 7. Phase 6: Multi-Sensor 500m Geospatial Risk Fusion & Daily Ranking
echo ""
echo "[*] Step 6/6: Fusing Multi-Sensor Feeds & Generating Top-N Patrol Ranking..."
$PYTHON_CMD -c "
from src.risk.ranking import generate_daily_ranking_pipeline
generate_daily_ranking_pipeline()
"

echo ""
echo "================================================================================"
echo "[+] SUCCESS: Multi-Sensor Pipeline Completed Successfully!"
echo "    - 500m Risk Grid: data/processed/fused_risk_grid.geojson"
echo "    - Daily Patrol Alerts: data/processed/daily_top_alerts.json"
echo "    - Scientific Report: reports/final_report.md"
echo "================================================================================"
echo ""
echo "[*] Launching Forest & River Guardian Enforcement Dashboard..."
echo "    - API URL: http://localhost:8000"
echo "    - Web Dashboard: http://localhost:8000/dashboard/"
echo "    - OpenAPI Docs: http://localhost:8000/docs"
echo "    - Default Ranger: ranger@guardian.internal / RangerPatrol2026!"
echo "    - Default Admin:  admin@guardian.internal  / ForestGuardianAdmin2026!"
echo "================================================================================"

$PYTHON_CMD -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
