"""
Model Inference & Feature Explainability Engine
================================================
Loads the persisted RandomForestClassifier model and encoders bundle, validates
input features against the training schema, generates genuine probabilities via
predict_proba(), and computes feature importance explainability.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Models live in ../ml/
ML_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "ml"))
MODEL_PATH = os.path.join(ML_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(ML_DIR, "encoders.pkl")

class DiseasePredictor:
    _instance = None
    
    def __init__(self):
        self.model = None
        self.encoders = None
        self.feature_names = []
        self.symptom_features = []
        self.gender_mapping = {}
        self.label_encoder = None
        self.disease_classes = []
        self.symptom_categories = {}
        self.symptom_display_map = {}
        self.load_artifacts()
        
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DiseasePredictor()
        return cls._instance

    def load_artifacts(self):
        if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
            raise FileNotFoundError(
                f"Model artifacts not found at {MODEL_PATH} or {ENCODERS_PATH}. "
                "Please run 'python ml/train.py' to generate model artifacts."
            )
        
        self.model = joblib.load(MODEL_PATH)
        self.encoders = joblib.load(ENCODERS_PATH)
        
        self.feature_names = self.encoders["feature_names"]
        self.symptom_features = self.encoders["symptom_features"]
        self.gender_mapping = self.encoders["gender_mapping"]
        self.label_encoder = self.encoders["label_encoder"]
        self.disease_classes = self.encoders["disease_classes"]
        self.symptom_categories = self.encoders.get("symptom_categories", {})
        self.symptom_display_map = self.encoders.get("symptom_display_map", {})

    def is_loaded(self) -> bool:
        return self.model is not None and self.encoders is not None

    def get_symptom_catalog(self) -> Dict[str, Any]:
        """Returns organized catalog of all recognized symptoms for UI selection."""
        formatted_categories = {}
        for category, syms in self.symptom_categories.items():
            formatted_categories[category] = [
                {"id": s, "label": self.symptom_display_map.get(s, s.replace('_', ' ').title())}
                for s in syms if s in self.symptom_features
            ]
            
        all_symptoms = [
            {"id": s, "label": self.symptom_display_map.get(s, s.replace('_', ' ').title())}
            for s in self.symptom_features
        ]
        
        return {
            "total_symptoms": len(self.symptom_features),
            "categories": formatted_categories,
            "all_symptoms": sorted(all_symptoms, key=lambda x: x["label"])
        }

    def predict(
        self,
        age: int,
        gender: str,
        user_symptoms: List[str]
    ) -> Dict[str, Any]:
        """
        Executes genuine ML prediction using trained Random Forest model.
        Returns top prediction, true model probability, differentials, and explainability.
        """
        if not self.is_loaded():
            self.load_artifacts()
            
        # Normalize and filter user symptoms
        cleaned_user_symptoms = set(s.strip().lower().replace(" ", "_") for s in user_symptoms)
        
        # Check that at least one symptom matches known features
        matched_symptoms = [s for s in self.symptom_features if s in cleaned_user_symptoms]
        if not matched_symptoms:
            raise ValueError(
                "None of the provided symptoms were recognized in the trained model catalog. "
                f"Please choose from available symptoms: {', '.join(self.symptom_features[:8])}..."
            )
            
        # 1. Build input feature vector in strict training column order
        gender_code = self.gender_mapping.get(gender.strip().capitalize(), 2)
        
        row_dict = {
            "age": age,
            "gender_encoded": gender_code
        }
        
        for sym in self.symptom_features:
            row_dict[sym] = 1 if sym in cleaned_user_symptoms else 0
            
        # Create DataFrame to preserve column names and prevent feature ordering mismatch
        input_df = pd.DataFrame([row_dict])[self.feature_names]
        
        # 2. Compute prediction and genuine class probabilities via predict_proba()
        probabilities = self.model.predict_proba(input_df)[0]
        
        # Sort class indices by descending probability
        sorted_indices = np.argsort(probabilities)[::-1]
        top_idx = sorted_indices[0]
        
        top_disease = self.label_encoder.inverse_transform([top_idx])[0]
        top_prob = float(probabilities[top_idx])
        
        # 3. Compile top differential conditions (next 2-3 most probable diseases)
        differentials = []
        for rank in range(1, min(4, len(sorted_indices))):
            d_idx = sorted_indices[rank]
            d_prob = float(probabilities[d_idx])
            # Only include if probability has a non-trivial value (e.g. > 1%)
            if d_prob > 0.01:
                differentials.append({
                    "disease": self.label_encoder.inverse_transform([d_idx])[0],
                    "probability": round(d_prob, 4),
                    "confidence_percentage": f"{d_prob * 100:.1f}%"
                })
                
        # 4. Feature Importance Explainability
        # Extract Random Forest Gini feature importances for user's active symptoms
        feature_importance_lookup = dict(zip(self.feature_names, self.model.feature_importances_))
        
        active_importances = []
        for sym in matched_symptoms:
            imp_val = float(feature_importance_lookup.get(sym, 0.0))
            active_importances.append((sym, imp_val))
            
        # Sort by importance descending
        active_importances.sort(key=lambda x: x[1], reverse=True)
        
        total_active_imp = sum(val for _, val in active_importances) or 1.0
        explainability = []
        for sym, val in active_importances[:5]:
            rel_weight = round(val / total_active_imp, 4)
            explainability.append({
                "symptom": sym,
                "symptom_display": self.symptom_display_map.get(sym, sym.replace('_', ' ').title()),
                "relative_importance": rel_weight,
                "importance_percentage": f"{rel_weight * 100:.1f}%"
            })
            
        return {
            "predicted_disease": top_disease,
            "confidence": round(top_prob, 4),
            "confidence_percentage": f"{top_prob * 100:.1f}%",
            "differentials": differentials,
            "explainability": explainability,
            "matched_symptoms_count": len(matched_symptoms)
        }
