import json
import os
from datetime import datetime
from smolagents import tool


@tool
def write_analysis_summary(summary_text: str, output_dir: str) -> str:
    """Write the final analysis summary to a text file in the output directory.

    Args:
        summary_text: The complete analysis summary text to write.
        output_dir: Directory where analysis_summary.txt will be saved.

    Returns:
        JSON string with the saved file path and timestamp.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "analysis_summary.txt")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"CSV Data Analysis Report\nGenerated: {timestamp}\n{'=' * 60}\n\n"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + summary_text)

    return json.dumps({"saved_summary": out_path, "timestamp": timestamp})
