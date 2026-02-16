import random
import pandas as pd
import numpy as np


def generate_population_data(output_path: str, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    continents = {
        "Africa": {
            "countries": ["Nigeria", "Ethiopia", "Kenya", "Ghana", "Tanzania"],
            "gdp_range": (500, 3000),
            "literacy_range": (55, 85),
        },
        "Asia": {
            "countries": ["China", "India", "Japan", "Indonesia", "Vietnam"],
            "gdp_range": (2000, 40000),
            "literacy_range": (75, 99),
        },
        "Europe": {
            "countries": ["Germany", "France", "UK", "Italy", "Spain"],
            "gdp_range": (25000, 55000),
            "literacy_range": (97, 99),
        },
        "North America": {
            "countries": ["USA", "Canada", "Mexico", "Cuba", "Guatemala"],
            "gdp_range": (4000, 65000),
            "literacy_range": (75, 99),
        },
        "South America": {
            "countries": ["Brazil", "Argentina", "Colombia", "Chile", "Peru"],
            "gdp_range": (5000, 20000),
            "literacy_range": (90, 99),
        },
        "Oceania": {
            "countries": ["Australia", "New Zealand", "Fiji", "Papua New Guinea", "Samoa"],
            "gdp_range": (3000, 60000),
            "literacy_range": (60, 99),
        },
    }

    cities_per_country = {
        "Nigeria": ["Lagos", "Abuja", "Kano", "Ibadan", "Kaduna"],
        "Ethiopia": ["Addis Ababa", "Dire Dawa", "Mekele", "Gondar", "Adama"],
        "Kenya": ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret"],
        "Ghana": ["Accra", "Kumasi", "Tamale", "Takoradi", "Sunyani"],
        "Tanzania": ["Dar es Salaam", "Dodoma", "Mwanza", "Arusha", "Mbeya"],
        "China": ["Shanghai", "Beijing", "Guangzhou", "Shenzhen", "Chengdu"],
        "India": ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai"],
        "Japan": ["Tokyo", "Osaka", "Nagoya", "Sapporo", "Fukuoka"],
        "Indonesia": ["Jakarta", "Surabaya", "Bandung", "Bekasi", "Medan"],
        "Vietnam": ["Ho Chi Minh City", "Hanoi", "Da Nang", "Hai Phong", "Can Tho"],
        "Germany": ["Berlin", "Hamburg", "Munich", "Cologne", "Frankfurt"],
        "France": ["Paris", "Marseille", "Lyon", "Toulouse", "Nice"],
        "UK": ["London", "Birmingham", "Leeds", "Glasgow", "Sheffield"],
        "Italy": ["Rome", "Milan", "Naples", "Turin", "Palermo"],
        "Spain": ["Madrid", "Barcelona", "Valencia", "Seville", "Zaragoza"],
        "USA": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
        "Canada": ["Toronto", "Montreal", "Vancouver", "Calgary", "Edmonton"],
        "Mexico": ["Mexico City", "Guadalajara", "Monterrey", "Puebla", "Tijuana"],
        "Cuba": ["Havana", "Santiago de Cuba", "Camagüey", "Holguín", "Santa Clara"],
        "Guatemala": ["Guatemala City", "Mixco", "Villa Nueva", "Petapa", "Quetzaltenango"],
        "Brazil": ["São Paulo", "Rio de Janeiro", "Brasília", "Salvador", "Fortaleza"],
        "Argentina": ["Buenos Aires", "Córdoba", "Rosario", "Mendoza", "Tucumán"],
        "Colombia": ["Bogotá", "Medellín", "Cali", "Barranquilla", "Cartagena"],
        "Chile": ["Santiago", "Valparaíso", "Concepción", "La Serena", "Antofagasta"],
        "Peru": ["Lima", "Arequipa", "Trujillo", "Chiclayo", "Iquitos"],
        "Australia": ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"],
        "New Zealand": ["Auckland", "Wellington", "Christchurch", "Hamilton", "Tauranga"],
        "Fiji": ["Suva", "Lautoka", "Nadi", "Labasa", "Ba"],
        "Papua New Guinea": ["Port Moresby", "Lae", "Madang", "Mount Hagen", "Goroka"],
        "Samoa": ["Apia", "Asau", "Mulifanua", "Afega", "Faleolo"],
    }

    rows = []
    for continent, info in continents.items():
        for country in info["countries"]:
            cities = cities_per_country[country]
            n_cities = min(5, len(cities))
            for i in range(n_cities):
                city = cities[i]

                # Right-skewed population
                pop_2020 = int(np.random.lognormal(mean=12.5, sigma=1.2))
                pop_2020 = max(50000, min(pop_2020, 15_000_000))
                pop_2023 = int(pop_2020 * random.uniform(1.01, 1.05))

                area_km2 = round(random.uniform(100, 3000), 1)

                # gdp correlated with continent
                gdp_base = random.uniform(*info["gdp_range"])
                # A few NaN for developing nations
                if continent == "Africa" and random.random() < 0.15:
                    gdp_per_capita = None
                else:
                    gdp_per_capita = round(gdp_base, 0)

                # Weak correlation with gdp
                literacy_base = random.uniform(*info["literacy_range"])
                literacy_rate_pct = round(min(99.9, max(40, literacy_base)), 1)

                urban_pct = round(random.uniform(20, 95), 1)

                rows.append({
                    "country": country,
                    "city": city,
                    "population_2020": pop_2020,
                    "population_2023": pop_2023,
                    "area_km2": area_km2,
                    "gdp_per_capita_usd": gdp_per_capita,
                    "literacy_rate_pct": literacy_rate_pct,
                    "urban_pct": urban_pct,
                    "region": continent,
                    "continent": continent,
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = generate_population_data("population_data.csv")
    print(df.head())
    print(df.describe())
