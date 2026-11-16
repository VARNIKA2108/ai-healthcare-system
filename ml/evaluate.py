import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "encoders.pkl")

def evaluate_saved_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
        raise FileNotFoundError("Model or encoder artifacts not found. Run train.py first.")
    
    print("Loading model and metadata artifacts...")
    model = joblib.load(MODEL_PATH)
    bundle = joblib.load(ENCODERS_PATH)
    
    feature_names = bundle["feature_names"]
    label_encoder = bundle["label_encoder"]
    disease_classes = bundle["disease_classes"]
    gender_mapping = bundle["gender_mapping"]
    
    print(f"Loading test split from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH).dropna().reset_index(drop=True)
    df["gender_encoded"] = df["gender"].map(gender_mapping).fillna(2).astype(int)
    
    X = df[feature_names]
    y = label_encoder.transform(df["disease"])
    
    # Recreate the exact same test split used during training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    print("\n" + "=" * 60)
    print("          SAVED MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"Evaluated Test Samples: {len(y_test)}")
    print(f"Test Accuracy:         {acc * 100:.2f}%")
    print(f"Weighted F1 Score:     {f1 * 100:.2f}%")
    print("=" * 60)
    
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=disease_classes))
    
    print("\n--- Top 10 Most Important Features Across Trees ---")
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    for idx, row in feature_imp_df.head(10).iterrows():
        print(f"{idx + 1:2d}. {row['Feature']:<30} : {row['Importance'] * 100:.2f}%")
        
    print("\n--- Confusion Matrix Shape ---")
    cm = confusion_matrix(y_test, y_pred)
    print(f"Matrix Dimension: {cm.shape[0]}x{cm.shape[1]} (Diagonal matches: {cm.diagonal().sum()}/{len(y_test)})")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_saved_model()
