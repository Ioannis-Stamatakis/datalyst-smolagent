import random

import numpy as np
import pandas as pd


def generate_healthcare_data(output_path: str, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    n = 350

    genders = ["Male", "Female", "Other"]
    gender_weights = [0.48, 0.48, 0.04]

    smoker_values = ["No", "Yes", "Former"]
    smoker_weights = [0.55, 0.25, 0.20]

    rows = []

    for i in range(n):
        patient_id = f"PAT_{1000 + i}"

        # Age: slight right skew — more middle-aged/older patients
        age = int(np.clip(np.random.normal(45, 18), 18, 90))

        gender = random.choices(genders, weights=gender_weights)[0]

        # BMI: bimodal — normal-weight cluster (18–25) and overweight cluster (27–35)
        if random.random() < 0.45:
            bmi = round(random.uniform(18, 25), 1)
        else:
            bmi = round(random.uniform(27, 45), 1)
        if random.random() < 0.04:
            bmi = None

        # Systolic BP: correlated with age and bmi
        bmi_val = bmi if bmi is not None else 28.0
        systolic_bp = int(np.clip(
            90 + 0.6 * age + 0.8 * (bmi_val - 22) + np.random.normal(0, 12),
            90, 200
        ))

        # Diastolic BP: correlated with systolic
        diastolic_bp = int(np.clip(
            0.55 * systolic_bp + np.random.normal(10, 6),
            60, 120
        ))

        # Cholesterol: ~5% NaN
        if random.random() < 0.05:
            cholesterol_mgdl = None
        else:
            cholesterol_mgdl = round(random.uniform(120, 320), 1)

        # Blood glucose: mostly 70–180; ~15 diabetic patients with 300–400
        is_diabetic = random.random() < (15 / n)
        if is_diabetic:
            blood_glucose_mgdl = round(random.uniform(300, 400), 1)
        else:
            blood_glucose_mgdl = round(random.uniform(70, 180), 1)

        smoker = random.choices(smoker_values, weights=smoker_weights)[0]

        # Exercise hours: ~8% NaN; lower for older patients
        if random.random() < 0.08:
            exercise_hours_week = None
        else:
            base_exercise = max(0, 10 - 0.1 * age)
            exercise_hours_week = round(
                np.clip(base_exercise + np.random.normal(0, 3), 0, 15), 1
            )

        # Diagnosis: derived from clinical stats
        if is_diabetic or blood_glucose_mgdl > 250:
            diagnosis = "Diabetes"
        elif systolic_bp >= 140 or diastolic_bp >= 90:
            diagnosis = "Hypertension"
        elif bmi_val >= 35:
            diagnosis = "Obesity"
        elif systolic_bp >= 130 and cholesterol_mgdl is not None and cholesterol_mgdl > 240:
            diagnosis = "Heart Disease"
        else:
            diagnosis = "Healthy"

        # Hospital visits: correlated with diagnosis severity
        severity_map = {
            "Healthy": (0, 2),
            "Hypertension": (1, 5),
            "Obesity": (1, 4),
            "Diabetes": (2, 8),
            "Heart Disease": (3, 12),
        }
        lo, hi = severity_map[diagnosis]
        hospital_visits_year = random.randint(lo, hi)

        rows.append({
            "patient_id": patient_id,
            "age": age,
            "gender": gender,
            "bmi": bmi,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "cholesterol_mgdl": cholesterol_mgdl,
            "blood_glucose_mgdl": blood_glucose_mgdl,
            "smoker": smoker,
            "exercise_hours_week": exercise_hours_week,
            "diagnosis": diagnosis,
            "hospital_visits_year": hospital_visits_year,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = generate_healthcare_data("healthcare_data.csv")
    print(df.head())
    print(df.describe())
