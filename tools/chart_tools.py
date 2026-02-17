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


@tool
def save_pie_charts(filepath: str, output_dir: str, max_categories: int) -> str:
    """Generate and save donut (pie) charts for low-cardinality categorical columns.

    Args:
        filepath: Path to the CSV file.
        output_dir: Directory where PNG files will be saved.
        max_categories: Maximum number of unique values a column may have to be included.

    Returns:
        JSON string with lists of saved PNG paths and skipped columns.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(filepath)

    saved_paths = []
    skipped = []

    for col in df.columns:
        if not pd.api.types.is_string_dtype(df[col]):
            continue
        n_unique = df[col].nunique()
        if not (2 <= n_unique <= max_categories):
            skipped.append(col)
            continue

        counts = df[col].value_counts()
        if len(counts) > 8:
            top = counts.iloc[:8]
            other = pd.Series({"Other": counts.iloc[8:].sum()})
            counts = pd.concat([top, other])

        colors = [plt.cm.tab10(i / len(counts)) for i in range(len(counts))]

        fig, ax = plt.subplots(figsize=(7, 7))
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.85,
            colors=colors,
        )
        # Donut hole
        circle = plt.Circle((0, 0), 0.60, fc="white")
        ax.add_patch(circle)

        for at in autotexts:
            at.set_fontsize(9)

        ax.set_title(f"Distribution of {col}", fontsize=13)
        plt.tight_layout()

        col_sanitized = col.replace(" ", "_").replace("/", "_")
        filename = f"pie_{col_sanitized}.png"
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        saved_paths.append(out_path)

    return json.dumps({"saved_pie_charts": saved_paths, "skipped": skipped})


@tool
def save_stacked_bar_chart(
    filepath: str,
    output_dir: str,
    group_col: str,
    stack_col: str,
    value_col: str,
    agg_func: str,
) -> str:
    """Generate and save a stacked bar chart aggregating a value column by two categorical columns.

    Args:
        filepath: Path to the CSV file.
        output_dir: Directory where the PNG file will be saved.
        group_col: Column for the X axis (fewer unique values).
        stack_col: Column whose categories become stacked segments (3–8 unique values).
        value_col: Numeric column to aggregate, or 'COUNT' to count rows.
        agg_func: Aggregation function — 'sum', 'mean', or 'count'.

    Returns:
        JSON string with the saved PNG path or an error message.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(filepath)

    if group_col not in df.columns:
        return json.dumps({"error": f"group_col '{group_col}' not found in columns"})
    if stack_col not in df.columns:
        return json.dumps({"error": f"stack_col '{stack_col}' not found in columns"})
    if agg_func not in {"sum", "mean", "count"}:
        return json.dumps({"error": f"agg_func must be 'sum', 'mean', or 'count'"})

    if df[group_col].nunique() > 15:
        return json.dumps({"error": f"group_col '{group_col}' has >15 unique values — skipped"})
    if df[stack_col].nunique() > 15:
        return json.dumps({"error": f"stack_col '{stack_col}' has >15 unique values — skipped"})

    if value_col == "COUNT":
        pivot = pd.crosstab(df[group_col], df[stack_col])
    else:
        if value_col not in df.columns:
            return json.dumps({"error": f"value_col '{value_col}' not found in columns"})
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            return json.dumps({"error": f"value_col '{value_col}' is not numeric"})
        pivot = df.groupby([group_col, stack_col])[value_col].agg(agg_func).unstack(fill_value=0)

    n_stacks = pivot.shape[1]
    colors = [plt.cm.Set2(i / max(n_stacks, 1)) for i in range(n_stacks)]

    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[0] * 0.9), 6))
    pivot.plot(kind="bar", stacked=True, color=colors, edgecolor="white", ax=ax)

    ax.set_title(f"{group_col} by {stack_col}" + (f" ({agg_func} of {value_col})" if value_col != "COUNT" else " (count)"), fontsize=13)
    ax.set_xlabel(group_col)
    ax.set_ylabel(value_col if value_col != "COUNT" else "Count")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title=stack_col, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()

    gc_san = group_col.replace(" ", "_").replace("/", "_")
    sc_san = stack_col.replace(" ", "_").replace("/", "_")
    filename = f"stacked_{gc_san}_by_{sc_san}.png"
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return json.dumps({
        "saved_stacked_bar": out_path,
        "group_col": group_col,
        "stack_col": stack_col,
        "agg_func": agg_func,
    })


@tool
def save_box_plots(filepath: str, output_dir: str, group_col: str) -> str:
    """Generate and save box plots for every numeric column, optionally grouped by a categorical column.

    Args:
        filepath: Path to the CSV file.
        output_dir: Directory where PNG files will be saved.
        group_col: Categorical column to group by, or 'NONE' for ungrouped plots.

    Returns:
        JSON string with list of saved PNG paths and the group column used.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(filepath)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    saved_paths = []

    use_groups = (
        group_col != "NONE"
        and group_col in df.columns
        and df[group_col].nunique() <= 15
    )

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 2:
            continue

        col_san = col.replace(" ", "_").replace("/", "_")

        if use_groups:
            gc_san = group_col.replace(" ", "_").replace("/", "_")
            groups = df.groupby(group_col)[col].apply(lambda s: s.dropna().tolist())
            labels = [str(k) for k in groups.index]
            data = [groups[k] for k in groups.index]

            fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(plt.cm.Set3(i / max(len(labels), 1)))

            ax.set_title(f"{col} by {group_col}", fontsize=13)
            ax.set_xlabel(group_col)
            ax.set_ylabel(col)
            if len(labels) > 5:
                ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()

            filename = f"box_{col_san}_by_{gc_san}.png"
        else:
            fig, ax = plt.subplots(figsize=(6, 5))
            bp = ax.boxplot(series.tolist(), patch_artist=True)
            bp["boxes"][0].set_facecolor("steelblue")
            ax.set_title(f"Box Plot — {col}", fontsize=13)
            ax.set_ylabel(col)
            ax.set_xticks([])
            plt.tight_layout()

            filename = f"box_{col_san}.png"

        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        saved_paths.append(out_path)

    return json.dumps({
        "saved_box_plots": saved_paths,
        "group_col_used": group_col if use_groups else "NONE",
    })


@tool
def save_time_series_chart(
    filepath: str,
    output_dir: str,
    date_col: str,
    value_cols: str,
    agg_period: str,
) -> str:
    """Generate and save a time series line chart for one or more numeric columns.

    Args:
        filepath: Path to the CSV file.
        output_dir: Directory where the PNG file will be saved.
        date_col: Name of the datetime column to use as the index.
        value_cols: Comma-separated numeric column names to plot, or 'ALL_NUMERIC' for all (capped at 5).
        agg_period: Resampling period — 'ME' (monthly), 'W' (weekly), 'D' (daily), or 'NONE' to skip resampling.

    Returns:
        JSON string with the saved PNG path or an error message.
    """
    import matplotlib.dates as mdates

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(filepath)

    if date_col not in df.columns:
        return json.dumps({"error": f"date_col '{date_col}' not found in columns"})

    parsed_dates = pd.to_datetime(df[date_col], errors="coerce")
    nat_frac = parsed_dates.isna().mean()
    if nat_frac > 0.20:
        return json.dumps({"error": f"date_col '{date_col}' has {nat_frac:.0%} unparseable values — skipped"})

    df = df.copy()
    df[date_col] = parsed_dates
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df = df.set_index(date_col)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if value_cols == "ALL_NUMERIC":
        cols_to_plot = numeric_cols[:5]
    else:
        cols_to_plot = [c.strip() for c in value_cols.split(",") if c.strip() in numeric_cols]

    if not cols_to_plot:
        return json.dumps({"error": "No valid numeric columns to plot"})

    plot_df = df[cols_to_plot]
    if agg_period != "NONE":
        plot_df = plot_df.resample(agg_period).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    add_markers = len(plot_df) <= 50

    for i, col in enumerate(cols_to_plot):
        color = plt.cm.tab10(i / max(len(cols_to_plot), 1))
        ax.plot(
            plot_df.index,
            plot_df[col],
            label=col,
            color=color,
            marker="o" if add_markers else None,
            markersize=4,
        )

    ax.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    ax.set_title(f"Time Series — {', '.join(cols_to_plot)}", fontsize=13)
    ax.set_xlabel(date_col)
    ax.set_ylabel("Value")
    ax.legend()
    plt.tight_layout()

    dc_san = date_col.replace(" ", "_").replace("/", "_")
    filename = f"timeseries_{dc_san}.png"
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return json.dumps({
        "saved_time_series": out_path,
        "date_col": date_col,
        "agg_period": agg_period,
        "columns_plotted": cols_to_plot,
    })


@tool
def save_scatter_with_regression(
    filepath: str,
    output_dir: str,
    x_col: str,
    y_col: str,
    color_col: str,
) -> str:
    """Generate and save a scatter plot with a regression line and R² annotation.

    Args:
        filepath: Path to the CSV file.
        output_dir: Directory where the PNG file will be saved.
        x_col: Numeric column for the X axis.
        y_col: Numeric column for the Y axis.
        color_col: Categorical column to color points by, or 'NONE' for a single color.

    Returns:
        JSON string with the saved PNG path, R², slope, and intercept, or an error message.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(filepath)

    if x_col not in df.columns or not pd.api.types.is_numeric_dtype(df[x_col]):
        return json.dumps({"error": f"x_col '{x_col}' is not a numeric column"})
    if y_col not in df.columns or not pd.api.types.is_numeric_dtype(df[y_col]):
        return json.dumps({"error": f"y_col '{y_col}' is not a numeric column"})

    subset_cols = [x_col, y_col]
    if color_col != "NONE" and color_col in df.columns:
        subset_cols.append(color_col)

    plot_df = df[subset_cols].dropna()
    if len(plot_df) < 5:
        return json.dumps({"error": "Fewer than 5 non-null rows after dropping NAs — skipped"})

    x = plot_df[x_col].values.astype(float)
    y = plot_df[y_col].values.astype(float)

    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    fig, ax = plt.subplots(figsize=(8, 6))

    use_color = (
        color_col != "NONE"
        and color_col in df.columns
        and pd.api.types.is_string_dtype(df[color_col])
        and df[color_col].nunique() <= 12
    )

    if use_color:
        sns.scatterplot(
            data=plot_df,
            x=x_col,
            y=y_col,
            hue=color_col,
            palette="tab10",
            s=50,
            alpha=0.65,
            ax=ax,
        )
    else:
        ax.scatter(x, y, alpha=0.5, color="steelblue", s=40)

    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color="crimson", linewidth=2, label="Regression")

    ax.text(
        0.05, 0.95,
        f"R² = {r_squared:.3f}\nslope = {slope:.4f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        fontsize=10,
    )

    ax.set_title(f"{x_col} vs {y_col}", fontsize=13)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    plt.tight_layout()

    x_san = x_col.replace(" ", "_").replace("/", "_")
    y_san = y_col.replace(" ", "_").replace("/", "_")
    filename = f"scatter_{x_san}_vs_{y_san}.png"
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    return json.dumps({
        "saved_scatter": out_path,
        "r_squared": round(r_squared, 4),
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
    })
