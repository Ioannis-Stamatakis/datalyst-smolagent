import random
import math
import pandas as pd
import numpy as np
from datetime import date, timedelta


def generate_weather_data(output_path: str, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    cities = ["New York", "London", "Tokyo", "Sydney", "Cairo"]
    city_offsets = {"New York": 12, "London": 10, "Tokyo": 15, "Sydney": -5, "Cairo": 22}

    start_date = date(2023, 1, 1)
    rows = []

    for day_num in range(365):
        d = start_date + timedelta(days=day_num)
        city = cities[day_num % len(cities)]
        offset = city_offsets[city]

        # Sinusoidal seasonal temperature
        day_of_year = day_num + 1
        base_temp = offset + 10 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
        temp_high = round(base_temp + random.uniform(2, 8), 1)
        temp_low = round(base_temp - random.uniform(2, 8), 1)

        # ~5 NaN values in humidity_pct
        if random.random() < 0.014:
            humidity_pct = None
        else:
            humidity_pct = round(random.uniform(30, 95), 1)

        # ~5 NaN values in wind_speed_kmh
        if random.random() < 0.014:
            wind_speed_kmh = None
        else:
            wind_speed_kmh = round(random.uniform(5, 60), 1)

        # precipitation_mm: 0 on ~60% of days, exponential otherwise
        if random.random() < 0.60:
            precipitation_mm = 0.0
        else:
            precipitation_mm = round(random.expovariate(1 / 8), 1)

        conditions = ["Sunny", "Cloudy", "Rainy", "Windy", "Foggy", "Snowy"]
        if precipitation_mm > 5:
            weather_condition = "Rainy"
        elif abs(base_temp) < 0:
            weather_condition = "Snowy"
        else:
            weather_condition = random.choice(conditions[:4])

        uv_index = round(max(0, 5 + 4 * math.sin(2 * math.pi * (day_of_year - 80) / 365) + random.uniform(-1, 1)), 1)

        rows.append({
            "date": d.isoformat(),
            "city": city,
            "temp_high_c": temp_high,
            "temp_low_c": temp_low,
            "humidity_pct": humidity_pct,
            "wind_speed_kmh": wind_speed_kmh,
            "precipitation_mm": precipitation_mm,
            "weather_condition": weather_condition,
            "uv_index": uv_index,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = generate_weather_data("weather_data.csv")
    print(df.head())
    print(df.describe())
