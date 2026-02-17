import os
import time
from smolagents import CodeAgent, LiteLLMModel


class GeminiLiteLLMModel(LiteLLMModel):
    """LiteLLMModel with a fixed delay before each API call to avoid rate limits."""

    def __init__(self, *args, delay_seconds: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_seconds = delay_seconds

    def generate(self, *args, **kwargs):
        time.sleep(self.delay_seconds)
        return super().generate(*args, **kwargs)
from tools import (
    load_csv_file,
    get_column_schema,
    compute_descriptive_stats,
    detect_outliers_iqr,
    compute_value_counts,
    compute_correlation_matrix,
    save_distribution_histograms,
    save_correlation_heatmap,
    save_categorical_bar_charts,
    save_missing_values_chart,
    save_pie_charts,
    save_stacked_bar_chart,
    save_box_plots,
    save_time_series_chart,
    save_scatter_with_regression,
    write_analysis_summary,
)

SYSTEM_INSTRUCTIONS = """
You are an expert data analyst. Analyze the CSV file at {csv_path} by following this protocol in order:

1. load_csv_file({csv_path!r}) — understand shape, columns, dtypes, missing counts
2. get_column_schema({csv_path!r}) — classify every column as numeric/categorical/datetime/text
3. compute_descriptive_stats({csv_path!r}, 'ALL_NUMERIC') — stats for all numeric columns
4. detect_outliers_iqr({csv_path!r}, column) — run for EACH numeric column individually
5. compute_value_counts({csv_path!r}, column, 10) — run for EACH categorical column
6. compute_correlation_matrix({csv_path!r}) — Pearson correlations
7. save_distribution_histograms({csv_path!r}, {output_dir!r}) — histograms + KDE
8. save_correlation_heatmap({csv_path!r}, {output_dir!r}) — correlation heatmap
9. save_categorical_bar_charts({csv_path!r}, {output_dir!r}, 10) — bar charts
10. save_missing_values_chart({csv_path!r}, {output_dir!r}) — missing values chart
11. save_pie_charts({csv_path!r}, {output_dir!r}, 10) — donut charts for low-cardinality categoricals
12. save_box_plots({csv_path!r}, {output_dir!r}, group_col) — box plots; choose the most meaningful
    categorical column with 3–8 unique values as group_col; pass 'NONE' if none qualifies
13. [CONDITIONAL] save_stacked_bar_chart — ONLY if ≥2 categorical columns exist with ≤12 unique values each
    - group_col: the categorical column with fewer unique values (X axis)
    - stack_col: a categorical column with 3–8 unique values (stack segments)
    - value_col: most meaningful numeric column, or 'COUNT'
    - agg_func: 'sum' for revenue/totals, 'mean' for rates/prices, 'count' for frequencies
14. [CONDITIONAL] save_time_series_chart — ONLY if a datetime column was found in step 2; otherwise skip entirely
    - date_col: the datetime column identified in step 2
    - value_cols: 2–3 most important numeric columns, comma-separated (e.g. 'revenue,units_sold')
    - agg_period: 'ME' for ≥90-day span, 'W' for 14–89 days, 'D' for <14 days, 'NONE' to skip resampling
15. [CONDITIONAL] save_scatter_with_regression — ONLY if any |r| ≥ 0.3 was found in step 6; otherwise skip
    - x_col/y_col: the numeric pair with the strongest absolute correlation
    - color_col: best low-cardinality categorical column (2–10 unique values), or 'NONE'
16. write_analysis_summary(summary_text, {output_dir!r}) — final structured report (always last)

Rules:
- Always execute steps 1–12 in order; never skip them
- Steps 13, 14, and 15 are conditional — only call those tools when the stated condition is met
- Step 16 is always the final step
- For steps 4 and 5, call the tool once per column (loop through all relevant columns)
- Include specific numbers (counts, percentages, values) in the final summary
- The summary must have these sections: Dataset Overview, Numeric Analysis, Categorical Analysis,
  Correlations, Outliers, Missing Data, Key Findings, Chart Paths
- List all saved PNG file paths at the end of the summary
- Output directory: {output_dir}
"""


def build_agent(csv_path: str, output_dir: str = "outputs") -> CodeAgent:
    model = GeminiLiteLLMModel(
        model_id="gemini/gemini-2.5-flash",
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=0.1,
        max_tokens=8192,
        delay_seconds=10,
    )

    instructions = SYSTEM_INSTRUCTIONS.format(
        csv_path=csv_path,
        output_dir=output_dir,
    )

    return CodeAgent(
        tools=[
            load_csv_file,
            get_column_schema,
            compute_descriptive_stats,
            detect_outliers_iqr,
            compute_value_counts,
            compute_correlation_matrix,
            save_distribution_histograms,
            save_correlation_heatmap,
            save_categorical_bar_charts,
            save_missing_values_chart,
            save_pie_charts,
            save_stacked_bar_chart,
            save_box_plots,
            save_time_series_chart,
            save_scatter_with_regression,
            write_analysis_summary,
        ],
        model=model,
        additional_authorized_imports=[
            "pandas",
            "numpy",
            "matplotlib",
            "matplotlib.pyplot",
            "seaborn",
            "json",
            "os",
            "datetime",
            "re",
            "statistics",
            "collections",
        ],
        instructions=instructions,
        planning_interval=None,
        max_steps=45,
        verbosity_level=1,
    )
