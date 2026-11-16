"""
Rule-Based Specialist Mapping Layer
====================================
ARCHITECTURAL DISTINCTION:
This module is strictly decoupled from the Random Forest Machine Learning model.

The Random Forest model is purely responsible for:
    Input Features (Age, Gender, Symptoms) -> Probable Disease Prediction

Once the disease label is determined by the ML model, this separate, deterministic
clinical rule-based layer maps that disease to the appropriate medical specialist,
clinical scope, urgency tier, and safe consultation advice.
"""

from typing import Dict, Any

# Deterministic mapping dictionary from Disease to Specialist Metadata
SPECIALIST_RULES: Dict[str, Dict[str, str]] = {
    "Common Cold": {
        "specialist_type": "General Physician / Family Medicine",
        "clinical_focus": "Upper respiratory viral infections and supportive symptom management.",
        "urgency_guidance": "Routine consultation if symptoms persist over 10 days or worsen.",
        "recommended_action": "Rest, hydration, and over-the-counter supportive relief. Consult a primary care physician if high fever develops."
    },
    "Influenza (Flu)": {
        "specialist_type": "General Physician / Infectious Disease Specialist",
        "clinical_focus": "Systemic viral influenza diagnosis, supportive care, and antiviral interventions.",
        "urgency_guidance": "Prompt consultation within 48 hours for vulnerable individuals (elderly, young children, immunocompromised).",
        "recommended_action": "Seek evaluation for potential prescription antiviral medications if within 48 hours of symptom onset."
    },
    "COVID-19": {
        "specialist_type": "Pulmonologist / Infectious Disease Specialist",
        "clinical_focus": "SARS-CoV-2 viral syndrome, respiratory monitoring, and post-viral complications.",
        "urgency_guidance": "Prompt evaluation; seek urgent medical attention if oxygen saturation drops or breathing becomes difficult.",
        "recommended_action": "Isolate, monitor respiratory effort and oxygen saturation, and consult a physician or tele-health provider."
    },
    "Asthma": {
        "specialist_type": "Pulmonologist / Allergist",
        "clinical_focus": "Chronic airway inflammation, bronchial hyperreactivity, and spirometry.",
        "urgency_guidance": "Prompt routine review; urgent emergency care if rescue inhaler provides no relief.",
        "recommended_action": "Schedule an asthma action plan evaluation and pulmonary function testing."
    },
    "Bronchitis": {
        "specialist_type": "Pulmonologist / General Physician",
        "clinical_focus": "Tracheobronchial inflammation, chronic sputum clearance, and differential diagnosis.",
        "urgency_guidance": "Routine consultation if cough lasts over 3 weeks or is accompanied by blood.",
        "recommended_action": "Undergo chest auscultation to rule out lower respiratory tract infections or pneumonia."
    },
    "Pneumonia": {
        "specialist_type": "Pulmonologist / Critical Care Specialist",
        "clinical_focus": "Alveolar parenchymal lung infection, antibiotic stewardship, and oxygen therapy.",
        "urgency_guidance": "Urgent consultation required within 12-24 hours; emergency room if severe dyspnea.",
        "recommended_action": "Obtain urgent medical evaluation, chest radiography (X-ray), and sputum/blood work."
    },
    "Allergic Rhinitis": {
        "specialist_type": "Allergist / Immunologist / ENT Specialist",
        "clinical_focus": "Environmental allergen sensitization, IgE-mediated rhinitis, and immunotherapy.",
        "urgency_guidance": "Routine elective consultation.",
        "recommended_action": "Consider comprehensive allergy skin prick testing or allergen-specific IgE blood tests."
    },
    "Sinusitis": {
        "specialist_type": "ENT (Otolaryngologist)",
        "clinical_focus": "Paranasal sinus mucosal inflammation, drainage obstruction, and endoscopic assessment.",
        "urgency_guidance": "Routine to prompt consultation if severe facial pain persists over 10 days.",
        "recommended_action": "Consult an ENT specialist for nasal endoscopy or diagnostic imaging if unresolved."
    },
    "Migraine": {
        "specialist_type": "Neurologist / Headache Specialist",
        "clinical_focus": "Neurovascular headache disorders, triptan therapy, and prophylactic management.",
        "urgency_guidance": "Prompt consultation for recurrent disabling attacks; urgent if sudden 'thunderclap' onset.",
        "recommended_action": "Keep a headache log (triggers, duration, intensity) and see a neurologist for personalized therapy."
    },
    "Tension Headache": {
        "specialist_type": "Neurologist / General Physician",
        "clinical_focus": "Musculoskeletal and pericranial muscle tension, stress-induced cefalea.",
        "urgency_guidance": "Routine consultation if unresponsive to basic analgesics or lifestyle adjustments.",
        "recommended_action": "Evaluate ergonomic stress, hydration, sleep quality, and physical therapy options."
    },
    "Hypertension": {
        "specialist_type": "Cardiologist / Internal Medicine Specialist",
        "clinical_focus": "Vascular hemodynamics, target-organ protection, and antihypertensive medication.",
        "urgency_guidance": "Prompt consultation; immediate emergency department visit if blood pressure exceeds 180/120 mmHg.",
        "recommended_action": "Obtain ambulatory blood pressure monitoring, baseline ECG, and renal function tests."
    },
    "Type 2 Diabetes": {
        "specialist_type": "Endocrinologist / Diabetologist",
        "clinical_focus": "Pancreatic beta-cell dynamics, glycemic regulation, HbA1c control, and metabolic health.",
        "urgency_guidance": "Prompt consultation within a week to assess fasting glucose and HbA1c.",
        "recommended_action": "Obtain fasting plasma glucose, oral glucose tolerance test, and dietary consultation."
    },
    "GERD (Acid Reflux)": {
        "specialist_type": "Gastroenterologist",
        "clinical_focus": "Lower esophageal sphincter incompetence, acid suppression, and upper endoscopy.",
        "urgency_guidance": "Routine consultation; prompt if experiencing dysphagia (difficulty swallowing) or hematemesis.",
        "recommended_action": "Avoid late meals, elevate head of bed, and consult a gastroenterologist for endoscopic evaluation."
    },
    "Gastroenteritis": {
        "specialist_type": "Gastroenterologist / General Physician",
        "clinical_focus": "Enteric pathogens, electrolyte stabilization, and mucosal restitution.",
        "urgency_guidance": "Prompt consultation if inability to tolerate fluids, severe dehydration, or bloody stools.",
        "recommended_action": "Focus on oral rehydration salts (ORS) and seek immediate medical evaluation if vomiting persists."
    },
    "Peptic Ulcer Disease": {
        "specialist_type": "Gastroenterologist",
        "clinical_focus": "Gastric/duodenal mucosal erosions, H. pylori eradication, and acid mitigation.",
        "urgency_guidance": "Prompt consultation; urgent emergency care if black/tarry stools or severe acute pain.",
        "recommended_action": "Undergo H. pylori urea breath/stool test and upper GI endoscopy (EGD)."
    },
    "Appendicitis": {
        "specialist_type": "General Surgeon / Emergency Medicine Physician",
        "clinical_focus": "Acute appendiceal luminal obstruction, peritonitis prevention, and surgical appendectomy.",
        "urgency_guidance": "URGENT / EMERGENCY: Immediate hospital emergency department evaluation required.",
        "recommended_action": "Do NOT take laxatives or pain medication that may mask symptoms; go directly to the nearest emergency center."
    },
    "Urinary Tract Infection (UTI)": {
        "specialist_type": "Urologist / General Physician",
        "clinical_focus": "Uropathogenic bacterial infections, urinalysis, and targeted antimicrobials.",
        "urgency_guidance": "Prompt consultation within 24 hours to prevent ascending pyelonephritis (kidney infection).",
        "recommended_action": "Obtain a clean-catch urine analysis and urine culture before beginning antibiotics."
    },
    "Osteoarthritis": {
        "specialist_type": "Orthopedic Specialist / Rheumatologist",
        "clinical_focus": "Degenerative articular cartilage loss, joint preservation, and mobility rehabilitation.",
        "urgency_guidance": "Routine consultation for chronic joint stiffness and pain management.",
        "recommended_action": "Schedule weight-bearing joint radiographs and consult an orthopedic clinician or physical therapist."
    },
    "Rheumatoid Arthritis": {
        "specialist_type": "Rheumatologist",
        "clinical_focus": "Autoimmune synovial inflammation, disease-modifying anti-rheumatic drugs (DMARDs), and joint protection.",
        "urgency_guidance": "Prompt early consultation within weeks to prevent irreversible joint erosion.",
        "recommended_action": "Request rheumatologic serology (RF, anti-CCP antibodies, ESR, CRP) from a specialist."
    },
    "Eczema (Dermatitis)": {
        "specialist_type": "Dermatologist",
        "clinical_focus": "Epidermal barrier disruption, topical immunomodulators, and allergen avoidance.",
        "urgency_guidance": "Routine consultation; prompt if secondary yellow bacterial crusting appears.",
        "recommended_action": "Maintain skin barrier moisturization, avoid harsh soaps, and seek dermatologist prescription creams."
    },
    "Psoriasis": {
        "specialist_type": "Dermatologist",
        "clinical_focus": "Hyperproliferative autoimmune keratinocyte kinetics, biologic therapies, and psoriatic arthritis surveillance.",
        "urgency_guidance": "Routine consultation for disease severity scoring (PASI) and systemic therapy evaluation.",
        "recommended_action": "Consult a board-certified dermatologist for tailored topical, phototherapy, or biologic management."
    },
    "Hypothyroidism": {
        "specialist_type": "Endocrinologist",
        "clinical_focus": "Thyroid hormone deficiency, levothyroxine titration, and pituitary-thyroid axis.",
        "urgency_guidance": "Routine consultation for serum thyroid stimulating hormone (TSH) testing.",
        "recommended_action": "Order a serum TSH and free T4 blood panel to establish accurate hormonal baseline."
    },
    "Hyperthyroidism": {
        "specialist_type": "Endocrinologist",
        "clinical_focus": "Thyrotoxicosis, Graves' disease, antithyroid medications, and cardiovascular protection.",
        "urgency_guidance": "Prompt consultation to avoid thyroid storm and tachycardia complications.",
        "recommended_action": "Undergo thyroid hormone testing (TSH, free T3/T4, TSI antibodies) and cardiac evaluation."
    },
    "Iron Deficiency Anemia": {
        "specialist_type": "Hematologist / Internal Medicine Specialist",
        "clinical_focus": "Erythrocyte kinetics, iron metabolism, ferritin levels, and occult blood loss investigation.",
        "urgency_guidance": "Prompt consultation to ascertain root cause of deficiency.",
        "recommended_action": "Complete a full blood count (CBC), serum ferritin, iron panel, and dietary assessment."
    },
    "Gout": {
        "specialist_type": "Rheumatologist",
        "clinical_focus": "Monosodium urate crystal arthropathy, acute flare termination, and urate-lowering therapy.",
        "urgency_guidance": "Prompt consultation for acute flare relief within 24-48 hours.",
        "recommended_action": "Check serum uric acid levels, avoid purine-rich foods and alcohol, and see a rheumatologist."
    }
}

DEFAULT_SPECIALIST = {
    "specialist_type": "General Physician / Internal Medicine",
    "clinical_focus": "Comprehensive medical evaluation and preliminary diagnostic workup.",
    "urgency_guidance": "Routine consultation with a primary care physician.",
    "recommended_action": "Schedule an appointment with a primary care doctor for clinical evaluation."
}

def get_specialist_for_disease(disease_name: str) -> Dict[str, str]:
    """
    Given a disease predicted by the Random Forest model, returns the rule-based
    specialist recommendation metadata.
    """
    return SPECIALIST_RULES.get(disease_name, DEFAULT_SPECIALIST)
