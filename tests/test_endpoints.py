import pytest
import asyncio
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_healthz_endpoint():
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "chunks_indexed" in data

def test_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["target_urls_count"] == 7

def test_lead_capture_endpoint():
    payload = {
        "full_name": "Test Candidate",
        "email": "candidate@example.com",
        "phone": "+1-555-0100",
        "treatment_interest": "ITC Tijuana Cancer Package",
        "session_id": "test_unit_session"
    }
    response = client.post("/api/leads", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["lead_id"] > 0

def test_admin_gaps_endpoint():
    response = client.get("/api/admin/gaps")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_out_of_scope_guardrail():
    response = client.post("/api/chat", json={
        "message": "What is the weather in Cancun?",
        "session_id": "test_out_of_scope"
    })
    assert response.status_code == 200
    data = response.json()
    assert "weather" in data["answer"].lower() or "cannot provide general information" in data["answer"].lower()

def test_direct_booking_guardrail():
    response = client.post("/api/chat", json={
        "message": "Can you book me an appointment for tomorrow?",
        "session_id": "test_booking"
    })
    assert response.status_code == 200
    data = response.json()
    assert "cannot directly book" in data["answer"].lower() or "free quote" in data["answer"].lower()
