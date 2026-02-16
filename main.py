import argparse
import os
import sys
from dotenv import load_dotenv


def generate_demo_csv(demo_name: str, output_dir: str) -> str:
    """Generate a demo CSV file and return its path."""
    os.makedirs(output_dir, exist_ok=True)

    if demo_name == "sales":
        from data.sales_data import generate_sales_data
        csv_path = os.path.join(output_dir, "sales_data.csv")
        generate_sales_data(csv_path)
    elif demo_name == "weather":
        from data.weather_data import generate_weather_data
        csv_path = os.path.join(output_dir, "weather_data.csv")
        generate_weather_data(csv_path)
    elif demo_name == "population":
        from data.population_data import generate_population_data
        csv_path = os.path.join(output_dir, "population_data.csv")
        generate_population_data(csv_path)
    else:
        print(f"Unknown demo: {demo_name}. Choose from: sales, weather, population")
        sys.exit(1)

    print(f"Generated {demo_name} CSV: {csv_path}")
    return csv_path


def run_analysis(csv_path: str, output_base: str) -> None:
    """Run the agent analysis on a CSV file."""
    from agent import build_agent

    csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.join(output_base, f"{csv_stem}_analysis")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nAnalyzing: {csv_path}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    agent = build_agent(csv_path=csv_path, output_dir=output_dir)
    agent.run(
        f"Analyze the CSV file at '{csv_path}' following the complete analysis protocol. "
        f"Save all outputs to '{output_dir}'."
    )

    print("\n" + "=" * 60)
    print(f"Analysis complete. Results in: {output_dir}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="CSV Data Analysis Agent using smolagents + Gemini"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--demo",
        choices=["sales", "weather", "population"],
        help="Generate and analyze a bundled demo CSV",
    )
    group.add_argument(
        "--csv",
        metavar="PATH",
        help="Path to a user-provided CSV file to analyze",
    )
    group.add_argument(
        "--generate-demos",
        action="store_true",
        help="Generate all demo CSVs without running analysis",
    )

    parser.add_argument(
        "--output",
        default="outputs",
        metavar="DIR",
        help="Base output directory (default: outputs/)",
    )

    args = parser.parse_args()

    if args.generate_demos:
        demos_dir = os.path.join(args.output, "demo_data")
        for name in ["sales", "weather", "population"]:
            generate_demo_csv(name, demos_dir)
        print("All demo CSVs generated.")
        return

    if args.demo:
        if not os.environ.get("GEMINI_API_KEY"):
            print("Error: GEMINI_API_KEY not set. Create a .env file with your key.")
            sys.exit(1)
        demos_dir = os.path.join(args.output, "demo_data")
        csv_path = generate_demo_csv(args.demo, demos_dir)
        run_analysis(csv_path, args.output)

    elif args.csv:
        if not os.environ.get("GEMINI_API_KEY"):
            print("Error: GEMINI_API_KEY not set. Create a .env file with your key.")
            sys.exit(1)
        if not os.path.isfile(args.csv):
            print(f"Error: File not found: {args.csv}")
            sys.exit(1)
        run_analysis(args.csv, args.output)


if __name__ == "__main__":
    main()
