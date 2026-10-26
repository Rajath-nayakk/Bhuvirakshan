"""Tests for FastAPI backend, RBAC, wildlife privacy redaction, and endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.db import init_db
from src.api.main import app

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    """Ensure database is seeded before running tests."""
    init_db()


def test_unauthenticated_access_blocked():
    """Verify protected endpoints reject requests without a valid token."""
    response = client.get("/api/v1/alerts/top")
    assert response.status_code == 401

    response = client.get("/api/v1/detections")
    assert response.status_code == 401


def test_login_invalid_credentials():
    """Verify failed login with incorrect password."""
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "ranger@guardian.internal", "password": "WrongPassword123"},
    )
    assert response.status_code == 401


def test_login_ranger_success():
    """Verify ranger authentication produces valid JWT token with ranger role."""
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "ranger@guardian.internal", "password": "RangerPatrol2026!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["role"] == "ranger"
    assert data["token_type"] == "bearer"


def test_login_admin_success():
    """Verify admin authentication produces valid JWT token with admin role."""
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@guardian.internal", "password": "ForestGuardianAdmin2026!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["role"] == "admin"


def test_get_top_alerts():
    """Verify ranger can retrieve daily top alerts."""
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "ranger@guardian.internal", "password": "RangerPatrol2026!"},
    )
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    response = client.get("/api/v1/alerts/top?limit=5", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "alerts" in data
    assert isinstance(data["alerts"], list)
    if data["alerts"]:
        first = data["alerts"][0]
        assert "cell_id" in first
        assert "threat_level" in first
        assert "explanation" in first


def test_get_cell_details():
    """Verify detailed risk breakdown for a 500m grid cell."""
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "ranger@guardian.internal", "password": "RangerPatrol2026!"},
    )
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    response = client.get("/api/v1/cells/CELL_0012", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["cell_id"] == "CELL_0012"
    assert "risk_score" in data
    assert "explanation" in data
    assert "recent_detections" in data


def test_sensor_event_ingestion():
    """Verify IoT acoustic and camera trap sensor event ingestion."""
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "ranger@guardian.internal", "password": "RangerPatrol2026!"},
    )
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    payload = {
        "source": "acoustic",
        "cell_id": "CELL_0045",
        "class_name": "gunshot",
        "confidence": 0.96,
        "threat_level": "CRITICAL",
        "coordinates": [77.23, 12.08],
    }
    response = client.post("/api/v1/events", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["cell_id"] == "CELL_0045"
    assert "detection_id" in data


def test_wildlife_coordinate_privacy_enforcement():
    """CRITICAL PRIVACY TEST: Verify animal sighting coordinates are masked for rangers, but visible for admins."""
    # Step 1: Ingest an animal detection with sensitive coordinates
    admin_login = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@guardian.internal", "password": "ForestGuardianAdmin2026!"},
    )
    admin_token = admin_login.json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    ingest_payload = {
        "source": "camera",
        "cell_id": "CELL_0099",
        "class_name": "animal",
        "confidence": 0.91,
        "threat_level": "LOW",
        "coordinates": [77.215, 12.072],
    }
    client.post("/api/v1/events", json=ingest_payload, headers=admin_headers)

    # Step 2: Query detections as Ranger
    ranger_login = client.post(
        "/api/v1/auth/login",
        json={"email": "ranger@guardian.internal", "password": "RangerPatrol2026!"},
    )
    ranger_token = ranger_login.json()["access_token"]
    ranger_headers = {"Authorization": f"Bearer {ranger_token}"}

    ranger_resp = client.get("/api/v1/detections?cell_id=CELL_0099", headers=ranger_headers)
    assert ranger_resp.status_code == 200
    ranger_dets = ranger_resp.json()
    assert len(ranger_dets) >= 1
    for d in ranger_dets:
        if d["class_name"] == "animal":
            # Must be redacted to protect endangered species from poaching
            assert d["coordinates"] is None
            assert d["is_wildlife_private"] is True

    # Step 3: Query detections as Admin
    admin_resp = client.get("/api/v1/detections?cell_id=CELL_0099", headers=admin_headers)
    assert admin_resp.status_code == 200
    admin_dets = admin_resp.json()
    animal_dets = [d for d in admin_dets if d["class_name"] == "animal"]
    assert len(animal_dets) >= 1
    # Authorized admin should see coordinates
    assert animal_dets[0]["coordinates"] == [77.215, 12.072]


def test_ranger_feedback_submission():
    """Verify ranger can confirm true alert or flag false alarm."""
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "ranger@guardian.internal", "password": "RangerPatrol2026!"},
    )
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    feedback_payload = {
        "alert_id": 1,
        "cell_id": "CELL_0012",
        "is_true_alert": True,
        "notes": "Field patrol confirmed illegal excavator along north bank.",
    }
    response = client.post("/api/v1/feedback", json=feedback_payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["is_true_alert"] is True
    assert data["status_updated"] == "confirmed_true"
    assert data["notes"] == "Field patrol confirmed illegal excavator along north bank."


def test_geojson_endpoints():
    """Verify GeoJSON grid and change endpoints."""
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "ranger@guardian.internal", "password": "RangerPatrol2026!"},
    )
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    grid_resp = client.get("/api/v1/grid/geojson", headers=headers)
    assert grid_resp.status_code == 200
    grid_data = grid_resp.json()
    assert grid_data["type"] == "FeatureCollection"
    assert "features" in grid_data

    change_resp = client.get("/api/v1/change/geojson", headers=headers)
    assert change_resp.status_code == 200
    change_data = change_resp.json()
    assert change_data["type"] == "FeatureCollection"
