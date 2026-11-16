import pytest
from fastapi.testclient import TestClient
from backend.main import app
from backend.specialist_mapping import get_specialist_for_disease, SPECIALIST_RULES
from backend.doctor_recommendation import find_doctors

client = TestClient(app)

def test_health_endpoint():
    """Verify /health returns 200, healthy status, and model loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["total_diseases"] >= 20
    assert data["total_symptoms"] >= 30

def test_symptoms_catalog_endpoint():
    """Verify /symptoms returns categorized list of recognized symptoms."""
    response = client.get("/symptoms")
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert "all_symptoms" in data
    assert len(data["all_symptoms"]) >= 30
    assert "Respiratory & Throat" in data["categories"]

def test_prediction_endpoint_valid_cold():
    """Verify /predict correctly predicts respiratory condition for cold symptoms."""
    payload = {
        "age": 28,
        "gender": "Male",
        "symptoms": ["runny_nose", "sneezing", "sore_throat", "cough", "nasal_congestion"],
        "location": "New York"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["predicted_disease"] in ["Common Cold", "Allergic Rhinitis"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert "%" in data["confidence_percentage"]
    assert "specialist_recommendation" in data
    assert "recommended_doctors" in data
    assert "disclaimer" in data
    assert "educational and informational" in data["disclaimer"].lower()
    assert len(data["key_contributing_features"]) > 0

def test_prediction_endpoint_valid_asthma():
    """Verify /predict with asthma symptoms."""
    payload = {
        "age": 22,
        "gender": "Female",
        "symptoms": ["shortness_of_breath", "wheezing", "chest_tightness", "cough"],
        "location": "San Francisco"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_disease"] == "Asthma"
    assert "Pulmonologist" in data["specialist_recommendation"]["specialist_type"]
    assert data["is_location_exact_match"] is True

def test_prediction_invalid_age():
    """Verify validation error when age is out of boundary."""
    # Negative age
    response = client.post("/predict", json={
        "age": -5,
        "gender": "Male",
        "symptoms": ["cough"],
        "location": "New York"
    })
    assert response.status_code == 422
    
    # Age over 120
    response2 = client.post("/predict", json={
        "age": 145,
        "gender": "Male",
        "symptoms": ["cough"],
        "location": "New York"
    })
    assert response2.status_code == 422

def test_prediction_empty_symptoms():
    """Verify error when symptoms list is empty."""
    response = client.post("/predict", json={
        "age": 30,
        "gender": "Female",
        "symptoms": [],
        "location": "Chicago"
    })
    assert response.status_code == 422

def test_prediction_unknown_symptoms():
    """Verify bad request when provided symptoms do not match any known feature."""
    response = client.post("/predict", json={
        "age": 30,
        "gender": "Female",
        "symptoms": ["alien_tentacles_growing", "quantum_teleportation_flu"],
        "location": "Chicago"
    })
    assert response.status_code == 400
    assert "None of the provided symptoms were recognized" in response.json()["detail"]

def test_specialist_mapping_coverage():
    """Verify all diseases in specialist rules have non-empty specialist metadata."""
    assert len(SPECIALIST_RULES) >= 20
    for disease, meta in SPECIALIST_RULES.items():
        assert "specialist_type" in meta
        assert "clinical_focus" in meta
        assert "urgency_guidance" in meta
        assert "recommended_action" in meta
        rec = get_specialist_for_disease(disease)
        assert rec["specialist_type"] == meta["specialist_type"]

def test_doctor_filtering_exact_match():
    """Verify doctor recommendation matches both specialty and location."""
    doctors, is_exact = find_doctors("Pulmonologist / Allergist", "New York")
    assert len(doctors) > 0
    assert is_exact is True
    assert any(doc["city"] == "New York" for doc in doctors)

def test_doctor_filtering_fallback():
    """Verify doctor recommendation falls back to available telehealth/regional doctors when city is unknown."""
    doctors, is_exact = find_doctors("Cardiologist", "UnknownRemoteTown12345")
    assert len(doctors) > 0
    assert is_exact is False

def test_doctors_query_endpoint():
    """Verify GET /doctors endpoint returns matching doctor records."""
    response = client.get("/doctors?specialist=Dermatologist&location=London")
    assert response.status_code == 200
    data = response.json()
    assert data["total_found"] > 0
    assert data["is_exact_location_match"] is True
    assert "Dermatologist" in data["doctors"][0]["specialization"]

def test_frontend_static_serving():
    """Verify that root / serves index.html and static assets are accessible."""
    root_res = client.get("/")
    assert root_res.status_code == 200
    assert "AI Healthcare Assistant" in root_res.text

    css_res = client.get("/static/style.css")
    assert css_res.status_code == 200

    js_res = client.get("/static/script.js")
    assert js_res.status_code == 200
