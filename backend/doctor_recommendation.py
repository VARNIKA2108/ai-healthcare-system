"""
Doctor Recommendation Engine
=============================
Filters educational sample doctors and clinic records by medical specialization
and geographical location.

IMPORTANT DISCLAIMER:
The doctor profiles returned by this module are structured demonstration records
for educational and portfolio purposes. They are not live verified clinical
schedules or medical affiliations.
"""

import os
import json
from typing import List, Tuple, Dict, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCTORS_FILE = os.path.join(BASE_DIR, "data", "doctors.json")

def load_doctors() -> List[Dict[str, Any]]:
    """Load doctor profiles from structured JSON storage."""
    if not os.path.exists(DOCTORS_FILE):
        return []
    with open(DOCTORS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_str(val: str) -> str:
    return val.strip().lower()

def matches_specialist(doctor_spec: str, target_spec: str) -> bool:
    """
    Checks if doctor specialization matches target specialist.
    Performs case-insensitive keyword intersection to account for composite
    titles (e.g., 'Pulmonologist / Allergist').
    """
    doc_tokens = set(clean_str(doctor_spec).replace("/", " ").replace("-", " ").split())
    target_tokens = set(clean_str(target_spec).replace("/", " ").replace("-", " ").split())
    
    # Exclude common stopwords
    stopwords = {"and", "or", "specialist", "medicine", "doctor", "physician", "care"}
    doc_core = doc_tokens - stopwords
    target_core = target_tokens - stopwords
    
    # Check if any core specialty term intersects (e.g., 'pulmonologist', 'cardiologist')
    intersection = doc_core.intersection(target_core)
    return len(intersection) > 0

def find_doctors(
    specialist_type: str,
    user_location: str,
    max_results: int = 5
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Filters doctors based on specialist type and user location.
    
    Returns:
        (doctors_list, is_exact_location_match)
    """
    all_doctors = load_doctors()
    if not all_doctors:
        return [], False
        
    cleaned_loc = clean_str(user_location)
    
    # Step 1: Match specialist first
    matching_specialists = [
        doc for doc in all_doctors
        if matches_specialist(doc.get("specialization", ""), specialist_type)
    ]
    
    # If no specific specialist found, fallback to general physicians
    if not matching_specialists:
        matching_specialists = [
            doc for doc in all_doctors
            if "general physician" in clean_str(doc.get("specialization", ""))
        ]
        
    # Step 2: Exact or partial match on city / region
    city_matches = [
        doc for doc in matching_specialists
        if cleaned_loc in clean_str(doc.get("city", "")) or
           cleaned_loc in clean_str(doc.get("state_country", ""))
    ]
    
    if city_matches:
        return city_matches[:max_results], True
        
    # Step 3: Location fallback - prioritize telehealth available doctors of that specialty
    telehealth_matches = [
        doc for doc in matching_specialists
        if doc.get("telehealth_available", False)
    ]
    
    fallback_candidates = telehealth_matches if telehealth_matches else matching_specialists
    return fallback_candidates[:max_results], False
