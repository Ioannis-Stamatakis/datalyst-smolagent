import json
import re
import pandas as pd
from smolagents import tool


@tool
def load_csv_file(filepath: str) -> str:
    """Load a CSV file and return a JSON string with shape, columns, dtypes, first 5 rows, and missing value counts.

    Args:
        filepath: Path to the CSV file to load.

    Returns:
        JSON string containing shape, columns, dtypes, head_5, and missing_counts.
    """
    df = pd.read_csv(filepath)
    result = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "head_5": df.head(5).to_dict(orient="records"),
        "missing_counts": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
    }
    return json.dumps(result, default=str)


@tool
def get_column_schema(filepath: str) -> str:
    """Classify each column as numeric, categorical, datetime, or text and return schema info.

    Args:
        filepath: Path to the CSV file.

    Returns:
        JSON string with column name mapped to type and unique_values count.
    """
    df = pd.read_csv(filepath)
    date_pattern = re.compile(
        r"^\d{4}[-/]\d{2}[-/]\d{2}$|^\d{2}[-/]\d{2}[-/]\d{4}$"
    )

    schema = {}
    for col in df.columns:
        col_data = df[col].dropna()
        unique_count = int(df[col].nunique())

        if pd.api.types.is_numeric_dtype(df[col]):
            col_type = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_type = "datetime"
        elif pd.api.types.is_string_dtype(df[col]) or col_data.dtype == object:
            # Check if values look like dates
            sample = col_data.head(20).astype(str)
            if sample.apply(lambda x: bool(date_pattern.match(x))).mean() > 0.8:
                col_type = "datetime"
            elif unique_count <= max(20, int(len(df) * 0.05)):
                col_type = "categorical"
            else:
                col_type = "text"
        else:
            col_type = "categorical"

        schema[col] = {
            "type": col_type,
            "unique_values": unique_count,
            "sample_values": col_data.head(5).tolist(),
        }

    return json.dumps(schema, default=str)
