from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    total_diseases: int = Field(..., json_schema_extra={"example": 25})
    total_symptoms: int = Field(..., json_schema_extra={"example": 42})

class SymptomCategory(BaseModel):
    category: str
    symptoms: List[str]

class SymptomsListResponse(BaseModel):
    total_symptoms: int
    categories: dict[str, List[dict[str, str]]]
    all_symptoms: List[dict[str, str]]

class PredictionRequest(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Patient age between 1 and 120 years", json_schema_extra={"example": 32})
    gender: str = Field(..., description="Patient gender (Male, Female, Other)", json_schema_extra={"example": "Female"})
    symptoms: List[str] = Field(..., min_length=1, description="List of reported symptom identifiers", json_schema_extra={"example": ["fever", "cough", "fatigue"]})
    location: str = Field(..., min_length=2, description="City or metropolitan area for doctor matching", json_schema_extra={"example": "New York"})

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        formatted = v.strip().capitalize()
        if formatted not in ["Male", "Female", "Other"]:
            raise ValueError("Gender must be 'Male', 'Female', or 'Other'")
        return formatted

    @field_validator("symptoms")
    @classmethod
    def validate_symptoms_list(cls, v: List[str]) -> List[str]:
        cleaned = [s.strip().lower() for s in v if s.strip()]
        if not cleaned:
            raise ValueError("At least one valid symptom must be selected")
        return cleaned

    @field_validator("location")
    @classmethod
    def validate_location(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned) < 2:
            raise ValueError("Location must contain at least 2 characters")
        return cleaned

class DifferentialCondition(BaseModel):
    disease: str
    probability: float
    confidence_percentage: str

class FeatureImportanceContribution(BaseModel):
    symptom: str
    symptom_display: str
    relative_importance: float
    importance_percentage: str

class SpecialistRecommendation(BaseModel):
    specialist_type: str
    clinical_focus: str
    urgency_guidance: str
    recommended_action: str

class DoctorProfile(BaseModel):
    id: str
    name: str
    specialization: str
    hospital_clinic: str
    city: str
    state_country: str
    contact: str
    email: str
    consultation_hours: str
    experience_years: int
    rating: float
    telehealth_available: bool
    is_sample_data: bool = True

class PredictionResponse(BaseModel):
    predicted_disease: str
    confidence: float
    confidence_percentage: str
    differential_diagnoses: List[DifferentialCondition]
    key_contributing_features: List[FeatureImportanceContribution]
    specialist_recommendation: SpecialistRecommendation
    recommended_doctors: List[DoctorProfile]
    location_queried: str
    doctor_count: int
    is_location_exact_match: bool
    disclaimer: str

class DoctorQueryResponse(BaseModel):
    total_found: int
    specialist: Optional[str]
    location: Optional[str]
    is_exact_location_match: bool
    doctors: List[DoctorProfile]
    disclaimer: str
