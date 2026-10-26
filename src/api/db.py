"""Database models and SQLite connection using SQLAlchemy.

Tables:
- users: authentication and role-based access control (admin, ranger)
- grid_scores: 500m cell daily risk scores and multi-sensor components
- alerts: priority patrol recommendations and enforcement status
- detections: raw and processed multi-sensor events (satellite, audio, camera)
- feedback: field validation by forest rangers (true alert vs false alarm)
"""

import json
import logging
import os
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path

import bcrypt
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

logger = logging.getLogger(__name__)

DB_PATH = Path("data/guardian.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def hash_password(password: str) -> str:
    """Hash password using native bcrypt."""
    pw_bytes = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt()).decode("utf-8")


def check_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password using native bcrypt."""
    pw_bytes = plain_password.encode("utf-8")[:72]
    return bcrypt.checkpw(pw_bytes, hashed_password.encode("utf-8"))


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(120), unique=True, index=True, nullable=False)
    hashed_password = Column(String(200), nullable=False)
    role = Column(String(50), default="ranger", nullable=False)  # 'admin' or 'ranger'
    full_name = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    feedback_entries = relationship("Feedback", back_populates="ranger")


class GridScore(Base):
    __tablename__ = "grid_scores"

    id = Column(Integer, primary_key=True, index=True)
    cell_id = Column(String(50), index=True, nullable=False)
    date = Column(String(20), index=True, nullable=False)
    risk_score = Column(Float, nullable=False)
    threat_level = Column(String(50), nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    explanation = Column(Text, nullable=True)
    components_json = Column(Text, nullable=True)
    centroid_lon = Column(Float, nullable=False)
    centroid_lat = Column(Float, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    cell_id = Column(String(50), index=True, nullable=False)
    date = Column(String(20), index=True, nullable=False)
    threat_level = Column(String(50), nullable=False)
    risk_score = Column(Float, nullable=False)
    daily_rank = Column(Integer, nullable=False)
    explanation = Column(Text, nullable=False)
    status = Column(String(50), default="pending")  # 'pending', 'confirmed_true', 'false_alarm'
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    feedback_records = relationship("Feedback", back_populates="alert")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(50), index=True, nullable=False)  # 'satellite', 'acoustic', 'camera'
    cell_id = Column(String(50), index=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    class_name = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    threat_level = Column(String(50), default="LOW")
    coordinates_json = Column(Text, nullable=True)  # [lon, lat] or geojson geometry
    evidence_path = Column(String(250), nullable=True)
    is_wildlife_private = Column(Boolean, default=False)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"), nullable=True)
    cell_id = Column(String(50), nullable=False)
    ranger_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    is_true_alert = Column(Boolean, nullable=False)  # True if confirmed illegal extraction, False if false alarm
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    alert = relationship("Alert", back_populates="feedback_records")
    ranger = relationship("User", back_populates="feedback_entries")


def get_db() -> Generator:
    """FastAPI database session dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables and seed initial users and sample alerts."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # Seed Admin User if not present
    admin_email = os.getenv("INITIAL_ADMIN_EMAIL", "admin@guardian.internal")
    admin_user = db.query(User).filter(User.email == admin_email).first()
    if not admin_user:
        hashed = hash_password(os.getenv("INITIAL_ADMIN_PASSWORD", "ForestGuardianAdmin2026!"))
        admin = User(
            email=admin_email,
            hashed_password=hashed,
            role="admin",
            full_name="HQ Enforcement Administrator",
        )
        db.add(admin)

    # Seed Ranger User if not present
    ranger_email = os.getenv("INITIAL_RANGER_EMAIL", "ranger@guardian.internal")
    ranger_user = db.query(User).filter(User.email == ranger_email).first()
    if not ranger_user:
        hashed_ranger = hash_password(os.getenv("INITIAL_RANGER_PASSWORD", "RangerPatrol2026!"))
        ranger = User(
            email=ranger_email,
            hashed_password=hashed_ranger,
            role="ranger",
            full_name="Forest Patrol Ranger - Cauvery Division",
        )
        db.add(ranger)

    # Populate initial alerts from processed files if table is empty
    alert_count = db.query(Alert).count()
    if alert_count == 0:
        daily_alerts_file = Path("data/processed/daily_top_alerts.json")
        if daily_alerts_file.exists():
            with open(daily_alerts_file, encoding="utf-8") as f:
                data = json.load(f)
                for alert_data in data.get("top_alerts", []):
                    alert = Alert(
                        cell_id=alert_data["cell_id"],
                        date=alert_data.get("date", datetime.now(UTC).strftime("%Y-%m-%d")),
                        threat_level=alert_data["threat_level"],
                        risk_score=alert_data["risk_score"],
                        daily_rank=alert_data.get("daily_rank", 1),
                        explanation=alert_data["explanation"],
                        status="pending",
                    )
                    db.add(alert)

    # Populate initial detections if empty
    det_count = db.query(Detection).count()
    if det_count == 0:
        # Acoustic detections
        ac_file = Path("data/processed/acoustic_events.json")
        if ac_file.exists():
            with open(ac_file, encoding="utf-8") as f:
                events = json.load(f)
                for ev in events[:15]:
                    d = Detection(
                        source="acoustic",
                        cell_id="CELL_0012",
                        timestamp=datetime.fromisoformat(ev["timestamp"]),
                        class_name=ev["class"],
                        confidence=ev["confidence"],
                        threat_level=ev.get("threat_level", "HIGH"),
                        coordinates_json=json.dumps([77.225, 12.085]),
                        is_wildlife_private=False,
                    )
                    db.add(d)

        # Camera detections
        cam_file = Path("data/processed/camera_events.json")
        if cam_file.exists():
            with open(cam_file, encoding="utf-8") as f:
                events = json.load(f)
                for ev in events[:15]:
                    is_animal = (ev["class"] == "animal")
                    d = Detection(
                        source="camera",
                        cell_id="CELL_0021" if is_animal else "CELL_0003",
                        timestamp=datetime.fromisoformat(ev["timestamp"]),
                        class_name=ev["class"],
                        confidence=ev["confidence"],
                        threat_level=ev.get("threat_level", "MEDIUM"),
                        coordinates_json=json.dumps(ev.get("coordinates") or [77.218, 12.091]),
                        evidence_path=ev.get("evidence_frame_path"),
                        is_wildlife_private=is_animal,
                    )
                    db.add(d)

    db.commit()
    db.close()
    logger.info("Database initialized successfully at %s", DATABASE_URL)
