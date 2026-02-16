import random
import math
import pandas as pd
import numpy as np
from datetime import date, timedelta


def generate_sales_data(output_path: str, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    n = 300
    regions = ["North", "South", "East", "West"]
    categories = ["Electronics", "Clothing", "Food", "Home", "Sports"]
    reps = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry"]

    base_prices = {
        "Electronics": 250.0,
        "Clothing": 60.0,
        "Food": 25.0,
        "Home": 90.0,
        "Sports": 75.0,
    }

    start_date = date(2023, 1, 1)
    rows = []
    for i in range(n):
        d = start_date + timedelta(days=random.randint(0, 364))
        region = random.choice(regions)
        category = random.choice(categories)
        rep = random.choice(reps)

        # Sinusoidal price noise per category
        day_of_year = d.timetuple().tm_yday
        price_noise = 1.0 + 0.15 * math.sin(2 * math.pi * day_of_year / 365)
        unit_price = round(base_prices[category] * price_noise * random.uniform(0.85, 1.15), 2)

        units_sold = random.randint(1, 200)
        revenue = round(unit_price * units_sold, 2)

        # ~5% NaN in discount_pct
        if random.random() < 0.05:
            discount_pct = None
        else:
            discount_pct = round(random.uniform(0, 30), 1)

        # ~3% NaN in customer_satisfaction
        if random.random() < 0.03:
            customer_satisfaction = None
        else:
            customer_satisfaction = round(random.uniform(1, 5), 1)

        rows.append({
            "date": d.isoformat(),
            "region": region,
            "product_category": category,
            "sales_rep": rep,
            "units_sold": units_sold,
            "unit_price": unit_price,
            "revenue": revenue,
            "discount_pct": discount_pct,
            "customer_satisfaction": customer_satisfaction,
        })

    df = pd.DataFrame(rows)

    # Inject 8 intentional outliers in units_sold
    outlier_indices = random.sample(range(n), 8)
    for idx in outlier_indices:
        df.at[idx, "units_sold"] = random.randint(500, 900)
        df.at[idx, "revenue"] = round(df.at[idx, "unit_price"] * df.at[idx, "units_sold"], 2)

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = generate_sales_data("sales_data.csv")
    print(df.head())
    print(df.describe())
