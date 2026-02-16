import json
import pandas as pd
import numpy as np
from smolagents import tool


@tool
def compute_descriptive_stats(filepath: str, columns: str) -> str:
    """Compute descriptive statistics (mean, median, std, min, max, q25, q75) for numeric columns.

    Args:
        filepath: Path to the CSV file.
        columns: Comma-separated list of column names, or 'ALL_NUMERIC' to auto-select all numeric columns.

    Returns:
        JSON string mapping column names to their descriptive statistics.
    """
    df = pd.read_csv(filepath)

    if columns.strip().upper() == "ALL_NUMERIC":
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = [c.strip() for c in columns.split(",")]

    result = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
        result[col] = {
            "count": int(series.count()),
            "missing": int(df[col].isnull().sum()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "q25": float(series.quantile(0.25)),
            "q75": float(series.quantile(0.75)),
            "skewness": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
        }

    return json.dumps(result, default=str)


@tool
def detect_outliers_iqr(filepath: str, column: str) -> str:
    """Detect outliers in a numeric column using the IQR method.

    Args:
        filepath: Path to the CSV file.
        column: Name of the numeric column to analyze for outliers.

    Returns:
        JSON string with q1, q3, iqr, lower_fence, upper_fence, outlier_count, and outlier_sample.
    """
    df = pd.read_csv(filepath)
    if column not in df.columns:
        return json.dumps({"error": f"Column '{column}' not found"})

    series = df[column].dropna()
    if not pd.api.types.is_numeric_dtype(series):
        return json.dumps({"error": f"Column '{column}' is not numeric"})

    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outliers = series[(series < lower_fence) | (series > upper_fence)]

    result = {
        "column": column,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "outlier_count": int(len(outliers)),
        "outlier_pct": round(len(outliers) / len(series) * 100, 2),
        "outlier_sample": outliers.head(10).tolist(),
    }

    return json.dumps(result, default=str)


@tool
def compute_value_counts(filepath: str, column: str, top_n: int) -> str:
    """Compute value counts and frequency percentages for a categorical column.

    Args:
        filepath: Path to the CSV file.
        column: Name of the column to compute value counts for.
        top_n: Number of top values to return.

    Returns:
        JSON string with list of {value, count, pct} dicts for top N values.
    """
    df = pd.read_csv(filepath)
    if column not in df.columns:
        return json.dumps({"error": f"Column '{column}' not found"})

    counts = df[column].value_counts().head(top_n)
    total = len(df[column].dropna())

    result = [
        {
            "value": str(val),
            "count": int(cnt),
            "pct": round(cnt / total * 100, 2),
        }
        for val, cnt in counts.items()
    ]

    return json.dumps({"column": column, "total_non_null": total, "top_values": result}, default=str)


@tool
def compute_correlation_matrix(filepath: str) -> str:
    """Compute the Pearson correlation matrix for all numeric columns in the CSV.

    Args:
        filepath: Path to the CSV file.

    Returns:
        JSON string with a dict-of-dicts representing the correlation matrix.
    """
    df = pd.read_csv(filepath)
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        return json.dumps({"error": "Fewer than 2 numeric columns found"})

    corr = numeric_df.corr(method="pearson")

    result = {
        "columns": corr.columns.tolist(),
        "matrix": corr.to_dict(),
    }

    return json.dumps(result, default=str)
