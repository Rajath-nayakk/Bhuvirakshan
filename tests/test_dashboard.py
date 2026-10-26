"""Tests for frontend static file delivery and dashboard asset routing."""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_dashboard_html_served():
    """Verify frontend HTML dashboard index is reachable."""
    response = client.get("/dashboard/")
    assert response.status_code == 200
    assert "Forest & River Guardian" in response.text
    assert "mapContainer" in response.text


def test_dashboard_css_served():
    """Verify frontend stylesheet is served."""
    response = client.get("/dashboard/styles.css")
    assert response.status_code == 200
    assert "--accent-emerald" in response.text


def test_dashboard_js_served():
    """Verify frontend JavaScript controller is served."""
    response = client.get("/dashboard/app.js")
    assert response.status_code == 200
    assert "fetchTopAlerts" in response.text


def test_evidence_figures_served():
    """Verify generated satellite and acoustic evidence figures are served."""
    sat_resp = client.get("/figures/satellite_before_after.png")
    assert sat_resp.status_code == 200

    audio_resp = client.get("/figures/audio_spectrogram_sample.png")
    assert audio_resp.status_code == 200
