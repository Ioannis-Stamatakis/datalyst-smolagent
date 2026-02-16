import os
import time
from smolagents import CodeAgent, LiteLLMModel


class ThrottledLiteLLMModel(LiteLLMModel):
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
11. write_analysis_summary(summary_text, {output_dir!r}) — write structured report

Rules:
- Never skip steps; complete every step in order
- For steps 4 and 5, call the tool once per column (loop through all relevant columns)
- Include specific numbers (counts, percentages, values) in the final summary
- The summary must have these sections: Dataset Overview, Numeric Analysis, Categorical Analysis,
  Correlations, Outliers, Missing Data, Key Findings, Chart Paths
- List all saved PNG file paths at the end of the summary
- Output directory: {output_dir}
"""


def build_agent(csv_path: str, output_dir: str = "outputs") -> CodeAgent:
    model = ThrottledLiteLLMModel(
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
        max_steps=30,
        verbosity_level=1,
    )
