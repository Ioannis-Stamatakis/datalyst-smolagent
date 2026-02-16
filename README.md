# Datalyst Agent

> An autonomous data analysis agent powered by **smolagents** + **Google Gemini 2.5 Flash** that takes any CSV file, explores it, writes and executes its own analysis code, generates charts, and produces a structured report — all without human intervention.

---

## What It Does

Drop in a CSV file. The agent figures out the rest.

It follows a structured analysis protocol end-to-end:

1. **Loads** the CSV and inspects shape, dtypes, and missing values
2. **Classifies** every column (numeric / categorical / datetime / text)
3. **Computes** descriptive statistics for all numeric columns
4. **Detects outliers** per column using the IQR method
5. **Analyzes distributions** of categorical columns (value counts + frequencies)
6. **Computes** a Pearson correlation matrix
7. **Generates charts**: histograms with KDE, correlation heatmap, bar charts, missing value chart
8. **Writes** a structured analysis summary with all findings and chart paths

---

## Why It's Interesting

Most AI data tools ask the model to *describe* what to do. This one asks the model to *write and execute Python code* as its action — the core differentiator of [smolagents'](https://github.com/huggingface/smolagents) `CodeAgent`. The agent reasons, writes pandas/matplotlib code, runs it, reads the output, and adapts — in a live Thought → Code → Observation loop.

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent framework | [smolagents](https://github.com/huggingface/smolagents) (HuggingFace) |
| LLM | Google Gemini 2.5 Flash via [LiteLLM](https://github.com/BerriAI/litellm) |
| Data processing | pandas, numpy |
| Visualizations | matplotlib, seaborn |
| CLI | Python argparse + python-dotenv |

---

## Project Structure

```
datalyst-agent/
├── data/
│   ├── sales_data.py         # Generator: 300-row sales dataset (regions, revenue, reps)
│   ├── weather_data.py       # Generator: 365-row weather dataset (5 cities, seasonal)
│   └── population_data.py    # Generator: 150-row population dataset (6 continents)
├── tools/
│   ├── data_tools.py         # load_csv_file, get_column_schema
│   ├── stats_tools.py        # descriptive stats, outlier detection, correlations
│   ├── chart_tools.py        # histogram, heatmap, bar charts, missing values chart
│   └── summary_tools.py      # write_analysis_summary
├── agent.py                  # CodeAgent + Gemini model configuration
├── main.py                   # CLI entry point
├── requirements.txt
└── .env                      # API key (not committed)
```

---

## Getting Started

### 1. Clone & set up the environment

```bash
git clone https://github.com/yourusername/datalyst-agent.git
cd datalyst-agent

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your Gemini API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

Get a free key at [Google AI Studio](https://aistudio.google.com/app/apikey).

### 3. Run

**Analyze a bundled demo dataset:**
```bash
python main.py --demo sales
python main.py --demo weather
python main.py --demo population
```

**Analyze your own CSV:**
```bash
python main.py --csv path/to/your/data.csv
```

**Just generate the demo CSVs (no analysis):**
```bash
python main.py --generate-demos
```

**Custom output directory:**
```bash
python main.py --demo sales --output my_results/
```

---

## Example Output

Running `python main.py --demo sales` produces:

```
outputs/sales_data_analysis/
├── analysis_summary.txt       ← structured report with all findings
├── hist_units_sold.png
├── hist_unit_price.png
├── hist_revenue.png
├── hist_discount_pct.png
├── hist_customer_satisfaction.png
├── correlation_heatmap.png
├── bar_region.png
├── bar_product_category.png
├── bar_sales_rep.png
└── missing_values.png
```

The agent also streams its full reasoning to the terminal in real time:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Step 1 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 ─ Executing parsed code: ──────────────────────────────────────────────────────
  csv_info = load_csv_file(filepath='outputs/demo_data/sales_data.csv')
  print(csv_info)
 ───────────────────────────────────────────────────────────────────────────────
Execution logs:
{"shape": [300, 9], "columns": ["date", "region", ...], ...}
[Step 1: Duration 18.05 seconds | Input tokens: 3,501 | Output tokens: 758]
```

---

## Demo Datasets

All datasets are synthetically generated — no downloads needed.

| Dataset | Rows | Highlights |
|---|---|---|
| **Sales** | 300 | 4 regions, 5 product categories, 8 sales reps. Intentional outliers in `units_sold` and ~5% missing values in `discount_pct` to test detection. |
| **Weather** | 365 | 5 cities, sinusoidal seasonal temperatures, exponential precipitation distribution. |
| **Population** | 150 | 30 countries across 6 continents, right-skewed population distribution, GDP correlated with continent. |

---

## Architecture

```
main.py  ──►  build_agent()  ──►  CodeAgent
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
               Gemini 2.5 Flash   11 Tools         Sandbox
               (via LiteLLM)    (pandas/mpl)     (code exec)
                    │
               Thought → Code → Observation loop
```

The `CodeAgent` writes Python code at each step, executes it in a sandboxed interpreter, reads the output, and decides the next action. Tools are plain Python functions decorated with `@tool` — no JSON schema boilerplate required.

---

## Requirements

- Python 3.10+
- Google Gemini API key (free tier works, paid recommended for longer runs)

```
smolagents>=1.24.0
litellm>=1.50.0
pandas>=2.0.0
numpy>=1.26.0
matplotlib>=3.8.0
seaborn>=0.13.0
python-dotenv>=1.0.0
```
