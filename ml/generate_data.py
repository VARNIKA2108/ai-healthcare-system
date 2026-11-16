import os
import random
import numpy as np
import pandas as pd

# Define base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset.csv")

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# List of 42 symptom features
SYMPTOMS = [
    "fever",
    "high_fever",
    "chills",
    "fatigue",
    "malaise",
    "headache",
    "throbbing_headache",
    "cough",
    "dry_cough",
    "productive_cough",
    "shortness_of_breath",
    "wheezing",
    "chest_tightness",
    "chest_pain",
    "sharp_chest_pain",
    "heartburn",
    "runny_nose",
    "sneezing",
    "sore_throat",
    "nasal_congestion",
    "loss_of_taste_smell",
    "nausea",
    "vomiting",
    "watery_diarrhea",
    "abdominal_cramps",
    "burning_stomach_pain",
    "lower_right_abdominal_pain",
    "dizziness",
    "excessive_thirst",
    "frequent_urination",
    "burning_urination",
    "pelvic_pain",
    "unexplained_weight_loss",
    "joint_pain",
    "joint_stiffness",
    "joint_swelling",
    "skin_rash",
    "itchy_skin",
    "red_silvery_skin_patches",
    "sensitivity_to_light",
    "rapid_heartbeat",
    "pale_skin"
]

# Disease profiles with core symptoms (high probability), secondary symptoms (moderate probability),
# typical age distributions, and gender distribution
DISEASE_PROFILES = {
    "Common Cold": {
        "core": ["runny_nose", "sneezing", "sore_throat", "cough", "nasal_congestion"],
        "secondary": ["fever", "fatigue", "headache", "malaise"],
        "age_range": (5, 80),
        "gender_ratio": 0.5,
    },
    "Influenza (Flu)": {
        "core": ["high_fever", "chills", "fatigue", "headache", "cough"],
        "secondary": ["sore_throat", "runny_nose", "malaise", "chest_tightness"],
        "age_range": (5, 85),
        "gender_ratio": 0.5,
    },
    "COVID-19": {
        "core": ["fever", "dry_cough", "fatigue", "loss_of_taste_smell"],
        "secondary": ["shortness_of_breath", "headache", "sore_throat", "chills", "malaise"],
        "age_range": (10, 85),
        "gender_ratio": 0.5,
    },
    "Asthma": {
        "core": ["shortness_of_breath", "wheezing", "chest_tightness", "cough"],
        "secondary": ["fatigue"],
        "age_range": (5, 65),
        "gender_ratio": 0.48,
    },
    "Bronchitis": {
        "core": ["productive_cough", "cough", "shortness_of_breath", "fatigue"],
        "secondary": ["chest_tightness", "fever", "chills", "malaise", "sore_throat"],
        "age_range": (18, 75),
        "gender_ratio": 0.52,
    },
    "Pneumonia": {
        "core": ["high_fever", "productive_cough", "sharp_chest_pain", "shortness_of_breath", "chills"],
        "secondary": ["fatigue", "malaise", "nausea", "headache"],
        "age_range": (15, 85),
        "gender_ratio": 0.53,
    },
    "Allergic Rhinitis": {
        "core": ["sneezing", "runny_nose", "nasal_congestion", "itchy_skin"],
        "secondary": ["headache", "fatigue"],
        "age_range": (8, 65),
        "gender_ratio": 0.49,
    },
    "Sinusitis": {
        "core": ["nasal_congestion", "headache", "runny_nose", "loss_of_taste_smell"],
        "secondary": ["sore_throat", "cough", "fatigue", "fever"],
        "age_range": (12, 70),
        "gender_ratio": 0.47,
    },
    "Migraine": {
        "core": ["throbbing_headache", "nausea", "sensitivity_to_light"],
        "secondary": ["vomiting", "dizziness", "fatigue"],
        "age_range": (15, 60),
        "gender_ratio": 0.30, # More common in females
    },
    "Tension Headache": {
        "core": ["headache", "fatigue"],
        "secondary": ["dizziness", "malaise"],
        "age_range": (18, 65),
        "gender_ratio": 0.45,
    },
    "Hypertension": {
        "core": ["dizziness", "headache", "rapid_heartbeat"],
        "secondary": ["fatigue", "chest_pain"],
        "age_range": (35, 85),
        "gender_ratio": 0.55,
    },
    "Type 2 Diabetes": {
        "core": ["excessive_thirst", "frequent_urination", "unexplained_weight_loss", "fatigue"],
        "secondary": ["dizziness", "malaise"],
        "age_range": (30, 85),
        "gender_ratio": 0.55,
    },
    "GERD (Acid Reflux)": {
        "core": ["heartburn", "chest_pain", "burning_stomach_pain"],
        "secondary": ["nausea", "cough", "sore_throat"],
        "age_range": (20, 75),
        "gender_ratio": 0.52,
    },
    "Gastroenteritis": {
        "core": ["watery_diarrhea", "vomiting", "nausea", "abdominal_cramps"],
        "secondary": ["fever", "chills", "fatigue", "malaise"],
        "age_range": (5, 80),
        "gender_ratio": 0.5,
    },
    "Peptic Ulcer Disease": {
        "core": ["burning_stomach_pain", "heartburn", "nausea"],
        "secondary": ["abdominal_cramps", "vomiting", "unexplained_weight_loss"],
        "age_range": (25, 75),
        "gender_ratio": 0.54,
    },
    "Appendicitis": {
        "core": ["lower_right_abdominal_pain", "nausea", "vomiting", "fever"],
        "secondary": ["abdominal_cramps", "chills", "fatigue"],
        "age_range": (10, 45),
        "gender_ratio": 0.55,
    },
    "Urinary Tract Infection (UTI)": {
        "core": ["burning_urination", "frequent_urination", "pelvic_pain"],
        "secondary": ["fever", "nausea", "fatigue", "malaise"],
        "age_range": (15, 80),
        "gender_ratio": 0.20, # Significantly more common in females
    },
    "Osteoarthritis": {
        "core": ["joint_pain", "joint_stiffness"],
        "secondary": ["joint_swelling", "fatigue"],
        "age_range": (45, 85),
        "gender_ratio": 0.40,
    },
    "Rheumatoid Arthritis": {
        "core": ["joint_pain", "joint_stiffness", "joint_swelling", "fatigue"],
        "secondary": ["fever", "malaise", "unexplained_weight_loss"],
        "age_range": (30, 75),
        "gender_ratio": 0.30,
    },
    "Eczema (Dermatitis)": {
        "core": ["skin_rash", "itchy_skin"],
        "secondary": ["fatigue"],
        "age_range": (5, 65),
        "gender_ratio": 0.48,
    },
    "Psoriasis": {
        "core": ["red_silvery_skin_patches", "itchy_skin", "skin_rash"],
        "secondary": ["joint_pain", "joint_stiffness"],
        "age_range": (18, 70),
        "gender_ratio": 0.50,
    },
    "Hypothyroidism": {
        "core": ["fatigue", "unexplained_weight_loss", "malaise"], # Note: metabolic slowing
        "secondary": ["dizziness", "joint_pain", "pale_skin"],
        "age_range": (20, 75),
        "gender_ratio": 0.20, # More common in females
    },
    "Hyperthyroidism": {
        "core": ["rapid_heartbeat", "unexplained_weight_loss", "fatigue"],
        "secondary": ["excessive_thirst", "frequent_urination", "dizziness"],
        "age_range": (20, 65),
        "gender_ratio": 0.25,
    },
    "Iron Deficiency Anemia": {
        "core": ["fatigue", "pale_skin", "dizziness", "shortness_of_breath"],
        "secondary": ["headache", "rapid_heartbeat", "malaise"],
        "age_range": (12, 65),
        "gender_ratio": 0.25, # More common in females
    },
    "Gout": {
        "core": ["joint_pain", "joint_swelling", "sharp_chest_pain"], # Severe focal joint swelling/pain
        "secondary": ["fever", "chills", "joint_stiffness"],
        "age_range": (35, 80),
        "gender_ratio": 0.80, # More common in males
    }
}

def generate_dataset(samples_per_disease=50):
    rows = []
    
    for disease, profile in DISEASE_PROFILES.items():
        core_syms = profile["core"]
        sec_syms = profile["secondary"]
        age_min, age_max = profile["age_range"]
        gender_male_prob = profile["gender_ratio"]
        
        for _ in range(samples_per_disease):
            record = {}
            # Generate Age
            age = int(np.clip(np.random.normal(loc=(age_min + age_max) / 2, scale=(age_max - age_min) / 4), age_min, age_max))
            record["age"] = age
            
            # Generate Gender
            gender = "Male" if random.random() < gender_male_prob else "Female"
            record["gender"] = gender
            
            # Core symptoms: 85-98% chance of presence
            # Secondary symptoms: 30-60% chance of presence
            # Noise symptoms: 2-5% chance of random spurious presence
            for sym in SYMPTOMS:
                if sym in core_syms:
                    prob = random.uniform(0.85, 0.99)
                elif sym in sec_syms:
                    prob = random.uniform(0.35, 0.65)
                else:
                    prob = random.uniform(0.01, 0.04) # clinical background noise
                record[sym] = 1 if random.random() < prob else 0
            
            # Ensure at least 2 symptoms are present so no empty patient records exist
            active_symptoms = [s for s in SYMPTOMS if record[s] == 1]
            if len(active_symptoms) < 2:
                for c_sym in random.sample(core_syms, min(2, len(core_syms))):
                    record[c_sym] = 1
                    
            record["disease"] = disease
            rows.append(record)
            
    df = pd.DataFrame(rows)
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(DATASET_PATH, index=False)
    print(f"Generated {len(df)} samples across {len(DISEASE_PROFILES)} diseases.")
    print(f"Saved dataset to {DATASET_PATH}")
    return df

if __name__ == "__main__":
    generate_dataset()
