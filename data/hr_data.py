import random

import numpy as np
import pandas as pd


def generate_hr_data(output_path: str, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    n = 300

    departments = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"]
    job_levels = ["Junior", "Mid", "Senior", "Lead", "Manager"]
    genders = ["Male", "Female", "Non-binary"]
    gender_weights = [0.46, 0.46, 0.08]

    # Base salary by department (annual USD)
    dept_base_salary = {
        "Engineering": 85000,
        "Finance": 80000,
        "Sales": 65000,
        "Marketing": 70000,
        "HR": 60000,
        "Operations": 58000,
    }

    # Multiplier by job level
    level_multiplier = {
        "Junior": 0.75,
        "Mid": 1.0,
        "Senior": 1.35,
        "Lead": 1.65,
        "Manager": 2.0,
    }

    # Tenure range by level
    tenure_range = {
        "Junior": (0.5, 3.0),
        "Mid": (2.0, 7.0),
        "Senior": (5.0, 12.0),
        "Lead": (7.0, 16.0),
        "Manager": (8.0, 20.0),
    }

    # Remote work % tendency by department
    remote_tendency = {
        "Engineering": (40, 100),
        "Marketing": (20, 80),
        "Finance": (10, 60),
        "Sales": (10, 50),
        "HR": (10, 50),
        "Operations": (0, 20),
    }

    # Training hours: higher for Junior/Mid
    training_range = {
        "Junior": (30, 80),
        "Mid": (20, 60),
        "Senior": (10, 40),
        "Lead": (5, 30),
        "Manager": (5, 25),
    }

    rows = []

    for i in range(n):
        employee_id = f"EMP_{1000 + i}"

        department = random.choice(departments)

        # Job level: weighted toward junior/mid
        level_weights = [0.30, 0.30, 0.20, 0.12, 0.08]
        job_level = random.choices(job_levels, weights=level_weights)[0]

        gender = random.choices(genders, weights=gender_weights)[0]

        # Age correlated with job level
        age_base = {"Junior": 25, "Mid": 30, "Senior": 37, "Lead": 42, "Manager": 46}
        age = int(np.clip(np.random.normal(age_base[job_level], 4), 22, 60))

        lo, hi = tenure_range[job_level]
        tenure_years = round(random.uniform(lo, hi), 1)

        base = dept_base_salary[department]
        mult = level_multiplier[job_level]
        salary_usd = round(base * mult * random.uniform(0.90, 1.10), 2)

        # Performance score: slight positive skew, 1–5
        performance_score = round(np.clip(np.random.normal(3.4, 0.8), 1.0, 5.0), 1)

        lo_t, hi_t = training_range[job_level]
        training_hours_year = random.randint(lo_t, hi_t)

        lo_r, hi_r = remote_tendency[department]
        remote_work_pct = round(random.uniform(lo_r, hi_r), 1)

        # Satisfaction score: ~6% NaN; lower for high-attrition candidates
        # We'll determine attrition first based on performance + relative salary
        dept_avg_salary = dept_base_salary[department] * 1.0  # rough midpoint
        salary_below_avg = salary_usd < dept_avg_salary

        # Attrition ~15%: more likely for low performance or below-avg salary
        attrition_prob = 0.07
        if performance_score < 2.5:
            attrition_prob += 0.15
        if salary_below_avg:
            attrition_prob += 0.08
        attrition = 1 if random.random() < attrition_prob else 0

        if random.random() < 0.06:
            satisfaction_score = None
        else:
            if attrition == 1:
                satisfaction_score = round(random.uniform(1.0, 5.0), 1)
            else:
                satisfaction_score = round(random.uniform(4.0, 10.0), 1)

        rows.append({
            "employee_id": employee_id,
            "department": department,
            "job_level": job_level,
            "gender": gender,
            "age": age,
            "tenure_years": tenure_years,
            "salary_usd": salary_usd,
            "performance_score": performance_score,
            "training_hours_year": training_hours_year,
            "remote_work_pct": remote_work_pct,
            "attrition": attrition,
            "satisfaction_score": satisfaction_score,
        })

    df = pd.DataFrame(rows)

    # Inject 5 executive salary outliers (>200k)
    outlier_indices = random.sample(range(n), 5)
    for idx in outlier_indices:
        df.at[idx, "salary_usd"] = round(random.uniform(200000, 350000), 2)
        df.at[idx, "job_level"] = "Manager"

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = generate_hr_data("hr_data.csv")
    print(df.head())
    print(df.describe())
