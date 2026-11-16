import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")

# Categorized symptom metadata for the user interface
SYMPTOM_CATEGORIES = {
    "Respiratory & Throat": [
        "cough", "dry_cough", "productive_cough", "shortness_of_breath",
        "wheezing", "chest_tightness", "runny_nose", "sneezing",
        "sore_throat", "nasal_congestion", "loss_of_taste_smell"
    ],
    "General & Systemic": [
        "fever", "high_fever", "chills", "fatigue", "malaise",
        "dizziness", "unexplained_weight_loss", "rapid_heartbeat", "pale_skin"
    ],
    "Neurological & Head": [
        "headache", "throbbing_headache", "sensitivity_to_light"
    ],
    "Cardiovascular & Chest": [
        "chest_pain", "sharp_chest_pain"
    ],
    "Gastrointestinal & Digestive": [
        "heartburn", "nausea", "vomiting", "watery_diarrhea",
        "abdominal_cramps", "burning_stomach_pain", "lower_right_abdominal_pain"
    ],
    "Endocrine & Urinary": [
        "excessive_thirst", "frequent_urination", "burning_urination", "pelvic_pain"
    ],
    "Musculoskeletal & Joints": [
        "joint_pain", "joint_stiffness", "joint_swelling"
    ],
    "Skin & Dermatological": [
        "skin_rash", "itchy_skin", "red_silvery_skin_patches"
    ]
}

def clean_feature_name(name: str) -> str:
    """Format technical snake_case into readable display text."""
    return name.replace("_", " ").title()

def train_model():
    print(f"Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of duplicate rows: {df.duplicated().sum()}")
    print(f"Missing values:\n{df.isnull().sum().max()} missing values found.")
    
    # Handle missing values if any exist
    df = df.dropna().reset_index(drop=True)
    
    # Define symptom columns (all columns except age, gender, disease)
    symptom_columns = [col for col in df.columns if col not in ["age", "gender", "disease"]]
    
    # Encode categorical gender
    gender_mapping = {"Male": 0, "Female": 1, "Other": 2}
    df["gender_encoded"] = df["gender"].map(gender_mapping).fillna(2).astype(int)
    
    # Define feature list in exact order
    feature_names = ["age", "gender_encoded"] + symptom_columns
    
    X = df[feature_names]
    y_raw = df["disease"]
    
    # Encode target disease labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    
    print(f"Number of target classes (diseases): {len(label_encoder.classes_)}")
    print(f"Total features: {len(feature_names)} (2 demographic + {len(symptom_columns)} symptoms)")
    
    # Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train RandomForestClassifier with specified hyperparameters
    print("\nTraining RandomForestClassifier(n_estimators=100, random_state=42)...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = rf_model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    print("\n" + "=" * 50)
    print("      ACTUAL TEST SET EVALUATION METRICS       ")
    print("=" * 50)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}% (weighted)")
    print(f"Recall:    {rec * 100:.2f}% (weighted)")
    print(f"F1-Score:  {f1 * 100:.2f}% (weighted)")
    print("=" * 50)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Bundle metadata and encoders
    bundle = {
        "feature_names": feature_names,
        "symptom_features": symptom_columns,
        "gender_mapping": gender_mapping,
        "label_encoder": label_encoder,
        "disease_classes": list(label_encoder.classes_),
        "symptom_categories": SYMPTOM_CATEGORIES,
        "symptom_display_map": {sym: clean_feature_name(sym) for sym in symptom_columns},
        "test_metrics": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "test_samples": int(len(y_test))
        }
    }
    
    # Persist model and encoders
    print(f"\nSaving model to: {MODEL_PATH}")
    joblib.dump(rf_model, MODEL_PATH)
    
    print(f"Saving encoders and metadata to: {ENCODERS_PATH}")
    joblib.dump(bundle, ENCODERS_PATH)
    
    print("Model training and artifact persistence completed successfully!")

if __name__ == "__main__":
    train_model()
