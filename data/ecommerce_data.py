import math
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd


def generate_ecommerce_data(output_path: str, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    n = 400

    categories = ["Electronics", "Clothing", "Books", "Home & Kitchen", "Sports", "Beauty", "Toys"]

    # ~30 distinct product names mapped to categories
    products_by_category = {
        "Electronics": ["Laptop", "Smartphone", "Headphones", "Tablet", "Smartwatch"],
        "Clothing": ["T-Shirt", "Jeans", "Winter Jacket", "Sneakers", "Dress"],
        "Books": ["Python Programming", "Data Science Handbook", "Fiction Novel", "Biography"],
        "Home & Kitchen": ["Coffee Maker", "Blender", "Cookware Set", "Air Fryer"],
        "Sports": ["Yoga Mat", "Dumbbells", "Running Shoes", "Bicycle Helmet"],
        "Beauty": ["Face Serum", "Moisturizer", "Lipstick", "Perfume"],
        "Toys": ["LEGO Set", "Action Figure", "Board Game", "Puzzle"],
    }

    # Price ranges per category
    price_ranges = {
        "Electronics": (50, 2000),
        "Clothing": (15, 150),
        "Books": (8, 40),
        "Home & Kitchen": (20, 300),
        "Sports": (10, 250),
        "Beauty": (5, 120),
        "Toys": (8, 100),
    }

    payment_methods = ["Credit Card", "PayPal", "Debit Card", "Gift Card"]

    # ~200 unique customers for repeat buyers
    customer_ids = [f"CUST_{1000 + i}" for i in range(200)]

    start_date = date(2023, 1, 1)
    rows = []

    for i in range(n):
        # Sinusoidal seasonal volume: higher in Nov-Dec (holiday spike)
        # Sample a day weighted by seasonal curve
        day_offset = random.randint(0, 364)
        d = start_date + timedelta(days=day_offset)
        day_of_year = d.timetuple().tm_yday
        # Holiday boost: peak around day 340 (early Dec)
        seasonal_weight = 1.0 + 0.4 * math.sin(math.pi * (day_of_year - 80) / 365)
        # Re-roll if seasonal weight check fails (simple rejection sampling for skew)
        for _ in range(5):
            if random.random() < seasonal_weight / 1.4:
                break
            day_offset = random.randint(0, 364)
            d = start_date + timedelta(days=day_offset)
            day_of_year = d.timetuple().tm_yday
            seasonal_weight = 1.0 + 0.4 * math.sin(math.pi * (day_of_year - 80) / 365)

        customer_id = random.choice(customer_ids)
        category = random.choice(categories)
        product_name = random.choice(products_by_category[category])

        quantity = random.randint(1, 5)
        lo, hi = price_ranges[category]
        unit_price = round(random.uniform(lo, hi), 2)
        total_price = round(quantity * unit_price, 2)

        # ~30% of orders have a discount
        discount_applied = 1 if random.random() < 0.30 else 0

        # Shipping days: mostly 1–14, ~10 outlier orders with 20–45 days
        shipping_days = random.randint(1, 14)

        # ~10% return rate overall; Electronics has higher rate (~20%)
        return_rate = 0.20 if category == "Electronics" else 0.08
        return_flag = 1 if random.random() < return_rate else 0

        # Customer rating 1–5, ~8% NaN; returns correlate with lower ratings
        if random.random() < 0.08:
            customer_rating = None
        else:
            if return_flag == 1:
                customer_rating = round(random.uniform(1.0, 3.0), 1)
            else:
                customer_rating = round(random.uniform(2.5, 5.0), 1)

        payment_method = random.choice(payment_methods)

        rows.append({
            "order_date": d.isoformat(),
            "customer_id": customer_id,
            "product_category": category,
            "product_name": product_name,
            "quantity": quantity,
            "unit_price": unit_price,
            "total_price": total_price,
            "discount_applied": discount_applied,
            "shipping_days": shipping_days,
            "return_flag": return_flag,
            "customer_rating": customer_rating,
            "payment_method": payment_method,
        })

    df = pd.DataFrame(rows)

    # Inject ~10 outlier shipping days (20–45 days)
    outlier_indices = random.sample(range(n), 10)
    for idx in outlier_indices:
        df.at[idx, "shipping_days"] = random.randint(20, 45)

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = generate_ecommerce_data("ecommerce_data.csv")
    print(df.head())
    print(df.describe())
