"""
FastAPI Application Entry Point
================================
Main web service exposing REST APIs for disease prediction, specialist mapping,
doctor recommendations, symptom catalog discovery, and static web UI hosting.
"""

import os
from typing import Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.schemas import (
    HealthResponse,
    SymptomsListResponse,
    PredictionRequest,
    PredictionResponse,
    DifferentialCondition,
    FeatureImportanceContribution,
    SpecialistRecommendation,
    DoctorProfile,
    DoctorQueryResponse
)
from backend.predictor import DiseasePredictor
from backend.specialist_mapping import get_specialist_for_disease
from backend.doctor_recommendation import find_doctors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

DISCLAIMER_TEXT = (
    "This application is intended for educational and informational purposes only. "
    "It does not provide medical diagnosis or treatment advice. "
    "Please consult a qualified healthcare professional for medical concerns."
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        predictor = DiseasePredictor.get_instance()
        print(f"Startup: Successfully loaded model with {len(predictor.disease_classes)} classes and {len(predictor.symptom_features)} features.")
    except Exception as e:
        print(f"Startup Warning: Could not initialize model on startup ({e}). Ensure 'python ml/train.py' was run.")
    yield

app = FastAPI(
    title="AI Healthcare Assistance System",
    description="Educational AI/ML healthcare assistance system predicting probable disease conditions from symptoms, with rule-based specialist recommendations and location-filtered doctor directory.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for local testing and cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns service health status, model load status, and feature counts."""
    try:
        predictor = DiseasePredictor.get_instance()
        is_ready = predictor.is_loaded()
        total_diseases = len(predictor.disease_classes) if is_ready else 0
        total_symptoms = len(predictor.symptom_features) if is_ready else 0
    except Exception:
        is_ready = False
        total_diseases = 0
        total_symptoms = 0
        
    return HealthResponse(
        status="healthy" if is_ready else "degraded (model not loaded)",
        model_loaded=is_ready,
        version="1.0.0",
        total_diseases=total_diseases,
        total_symptoms=total_symptoms
    )

@app.get("/symptoms", response_model=SymptomsListResponse, tags=["Catalog"])
def get_symptoms():
    """Returns list of all recognized symptoms categorized by clinical domain for frontend UI selection."""
    try:
        predictor = DiseasePredictor.get_instance()
        catalog = predictor.get_symptom_catalog()
        return SymptomsListResponse(**catalog)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load symptom catalog: {str(e)}"
        )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_disease(request: PredictionRequest):
    """
    Predicts probable condition from age, gender, and selected symptoms.
    Maps predicted disease to recommended medical specialist and filters
    recommended doctors by user location.
    """
    try:
        predictor = DiseasePredictor.get_instance()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Prediction model is unavailable: {str(e)}"
        )
        
    try:
        # Step 1: Execute Random Forest ML Prediction
        ml_result = predictor.predict(
            age=request.age,
            gender=request.gender,
            user_symptoms=request.symptoms
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(val_err)
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(err)}"
        )
        
    predicted_disease = ml_result["predicted_disease"]
    confidence = ml_result["confidence"]
    confidence_pct = ml_result["confidence_percentage"]
    
    # Step 2: Separate Rule-Based Specialist Mapping
    specialist_meta = get_specialist_for_disease(predicted_disease)
    specialist_rec = SpecialistRecommendation(**specialist_meta)
    
    # Step 3: Location and Specialist Filtered Doctor Recommendations
    doctors_raw, is_exact_match = find_doctors(
        specialist_type=specialist_meta["specialist_type"],
        user_location=request.location,
        max_results=4
    )
    doctors_list = [DoctorProfile(**doc) for doc in doctors_raw]
    
    # Step 4: Assemble Differential Conditions
    differentials = [
        DifferentialCondition(**diff) for diff in ml_result.get("differentials", [])
    ]
    
    # Step 5: Assemble Feature Importance Explainability
    explainability = [
        FeatureImportanceContribution(**f) for f in ml_result.get("explainability", [])
    ]
    
    return PredictionResponse(
        predicted_disease=predicted_disease,
        confidence=confidence,
        confidence_percentage=confidence_pct,
        differential_diagnoses=differentials,
        key_contributing_features=explainability,
        specialist_recommendation=specialist_rec,
        recommended_doctors=doctors_list,
        location_queried=request.location,
        doctor_count=len(doctors_list),
        is_location_exact_match=is_exact_match,
        disclaimer=DISCLAIMER_TEXT
    )

@app.get("/doctors", response_model=DoctorQueryResponse, tags=["Directory"])
def query_doctors(
    specialist: Optional[str] = Query(None, description="Specialist title e.g. Pulmonologist, Cardiologist"),
    location: Optional[str] = Query(None, description="City or region name e.g. New York, Chicago")
):
    """Query doctors directory filtered by specialization and/or city."""
    spec_query = specialist if specialist else "General Physician"
    loc_query = location if location else ""
    
    doctors_raw, is_exact = find_doctors(
        specialist_type=spec_query,
        user_location=loc_query,
        max_results=10
    )
    
    return DoctorQueryResponse(
        total_found=len(doctors_raw),
        specialist=specialist,
        location=location,
        is_exact_location_match=is_exact,
        doctors=[DoctorProfile(**d) for d in doctors_raw],
        disclaimer=DISCLAIMER_TEXT
    )

# Serve Frontend static assets if available
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend_root():
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return JSONResponse({"message": "Frontend index.html not found, visit /docs for API documentation."})

    @app.get("/style.css", include_in_schema=False)
    def serve_style():
        return FileResponse(os.path.join(FRONTEND_DIR, "style.css"))

    @app.get("/script.js", include_in_schema=False)
    def serve_script():
        return FileResponse(os.path.join(FRONTEND_DIR, "script.js"))
