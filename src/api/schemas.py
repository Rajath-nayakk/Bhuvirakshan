"""Pydantic schemas for API request validation and structured responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# --- Auth Schemas ---
class LoginRequest(BaseModel):
    email: str = Field(..., json_schema_extra={"example": "ranger@guardian.internal"})
    password: str = Field(..., json_schema_extra={"example": "RangerPatrol2026!"})


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_email: str
    role: str
    full_name: str | None = None


class UserProfileResponse(BaseModel):
    id: int
    email: str
    role: str
    full_name: str | None


# --- Alert Schemas ---
class AlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    cell_id: str
    date: str
    threat_level: str
    risk_score: float
    daily_rank: int
    explanation: str
    status: str
    created_at: datetime


class TopAlertsResponse(BaseModel):
    total_alerts: int
    date: str
    alerts: list[AlertResponse]


# --- Grid Cell Schemas ---
class GridScoreResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    cell_id: str
    date: str
    risk_score: float
    threat_level: str
    explanation: str | None
    centroid_lon: float
    centroid_lat: float
    components: dict[str, Any] | None = None


class CellDetailResponse(BaseModel):
    cell_id: str
    date: str
    risk_score: float
    threat_level: str
    explanation: str
    centroid_lon: float
    centroid_lat: float
    components: dict[str, Any]
    recent_detections: list[dict[str, Any]]
    associated_alerts: list[AlertResponse]


# --- Detection Schemas ---
class DetectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source: str
    cell_id: str
    timestamp: datetime
    class_name: str
    confidence: float
    threat_level: str
    coordinates: Any | None = None  # Obfuscated or None if wildlife private and user not admin
    evidence_path: str | None = None
    is_wildlife_private: bool


# --- Ingestion & Feedback Schemas ---
class EventIngestRequest(BaseModel):
    source: str = Field(..., json_schema_extra={"example": "acoustic"}, description="satellite, acoustic, or camera")
    cell_id: str = Field(..., json_schema_extra={"example": "CELL_0012"})
    timestamp: datetime | None = None
    class_name: str = Field(..., json_schema_extra={"example": "gunshot"})
    confidence: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.94})
    threat_level: str | None = Field("HIGH", json_schema_extra={"example": "CRITICAL"})
    coordinates: list[float] | None = Field(None, json_schema_extra={"example": [77.225, 12.085]})
    evidence_path: str | None = None
    is_wildlife_private: bool | None = False


class IngestResponse(BaseModel):
    status: str = "success"
    message: str
    detection_id: int
    cell_id: str


class FeedbackRequest(BaseModel):
    alert_id: int | None = Field(None, json_schema_extra={"example": 1})
    cell_id: str = Field(..., json_schema_extra={"example": "CELL_0012"})
    is_true_alert: bool = Field(..., description="True if confirmed illegal activity, False if false alarm")
    notes: str | None = Field(None, json_schema_extra={"example": "Patrol inspected riverbank; verified unauthorized hydraulic sand dredge."})


class FeedbackResponse(BaseModel):
    id: int
    cell_id: str
    is_true_alert: bool
    notes: str | None
    ranger_id: int | None
    status_updated: str
    created_at: datetime
