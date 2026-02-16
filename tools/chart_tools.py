import json
import os
import matplotlib
matplotlib.use("Agg")  # Must be before any pyplot import — headless WSL2
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from smolagents import tool


@tool
def save_distribution_histograms(filepath: str, output_dir: str) -> str:
    """Generate and save histogram with KDE for every numeric column in the CSV.

    Args:
        filepath: Path to the CSV file.
        output_dir: Directory where PNG files will be saved.

    Returns:
        JSON string with list of saved PNG file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(filepath)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    saved_paths = []
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(series, bins=30, density=True, alpha=0.6, color="steelblue", edgecolor="white")

        # KDE overlay using numpy
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(series)
            x_range = np.linspace(series.min(), series.max(), 200)
            ax.plot(x_range, kde(x_range), color="darkblue", linewidth=2)
        except ImportError:
            # Manual KDE with numpy (Gaussian kernel)
            x_range = np.linspace(series.min(), series.max(), 200)
            bw = series.std() * (len(series) ** -0.2)
            if bw > 0:
                kde_vals = np.array([
                    np.mean(np.exp(-0.5 * ((x - series.values) / bw) ** 2) / (bw * np.sqrt(2 * np.pi)))
                    for x in x_range
                ])
                ax.plot(x_range, kde_vals, color="darkblue", linewidth=2)

        ax.set_title(f"Distribution of {col}", fontsize=13)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        plt.tight_layout()

        filename = f"hist_{col.replace(' ', '_').replace('/', '_')}.png"
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        saved_paths.append(out_path)

    return json.dumps({"saved_histograms": saved_paths})


@tool
def save_correlation_heatmap(filepath: str, output_dir: str) -> str:
    """Generate and save an annotated seaborn correlation heatmap for numeric columns.

    Args:
        filepath: Path to the CSV file.
        output_dir: Directory where the PNG file will be saved.

    Returns:
        JSON string with the saved PNG file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(filepath)
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        return json.dumps({"error": "Fewer than 2 numeric columns — heatmap skipped"})

    corr = numeric_df.corr(method="pearson")
    n = corr.shape[0]
    fig_size = max(6, n * 1.2)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Pearson Correlation Heatmap", fontsize=14)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "correlation_heatmap.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return json.dumps({"saved_heatmap": out_path})


@tool
def save_categorical_bar_charts(filepath: str, output_dir: str, top_n: int) -> str:
    """Generate and save bar charts for the top N value frequencies of each categorical column.

    Args:
        filepath: Path to the CSV file.
        output_dir: Directory where PNG files will be saved.
        top_n: Number of top categories to show per chart.

    Returns:
        JSON string with list of saved PNG file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(filepath)

    # Select categorical columns (string/object dtype with reasonable cardinality)
    cat_cols = []
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) and df[col].nunique() <= max(20, int(len(df) * 0.1)):
            cat_cols.append(col)

    saved_paths = []
    for col in cat_cols:
        counts = df[col].value_counts().head(top_n)
        if len(counts) == 0:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(counts) * 0.8), 5))
        counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(f"Top {top_n} Values — {col}", fontsize=13)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()

        filename = f"bar_{col.replace(' ', '_').replace('/', '_')}.png"
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        saved_paths.append(out_path)

    return json.dumps({"saved_bar_charts": saved_paths})


@tool
def save_missing_values_chart(filepath: str, output_dir: str) -> str:
    """Generate and save a bar chart showing missing value percentage per column.

    Args:
        filepath: Path to the CSV file.
        output_dir: Directory where the PNG file will be saved.

    Returns:
        JSON string with saved PNG path, or 'NO_MISSING_VALUES' if no missing data exists.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(filepath)

    missing_pct = (df.isnull().mean() * 100).round(2)
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

    if len(missing_pct) == 0:
        return json.dumps({"result": "NO_MISSING_VALUES"})

    fig, ax = plt.subplots(figsize=(max(8, len(missing_pct) * 1.2), 5))
    missing_pct.plot(kind="bar", ax=ax, color="tomato", edgecolor="white")
    ax.set_title("Missing Values by Column (%)", fontsize=13)
    ax.set_xlabel("Column")
    ax.set_ylabel("Missing %")
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(missing_pct):
        ax.text(i, v + 0.2, f"{v:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()

    out_path = os.path.join(output_dir, "missing_values.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return json.dumps({"saved_missing_chart": out_path})
