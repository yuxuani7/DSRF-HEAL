import argparse
import sys
from pathlib import Path

from src.config import load_env_file, load_llm_configs
from src.control_reflect_runner import (
    run_control_reflect_all_datasets_with_run_dir,
    run_control_reflect_single_behavior_baseline,
)
from src.hallucination_validator import validate_output_run
from src.io_utils import ensure_dir
from src.results_summary import summarize_validation_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simple self-reflection control pipeline (1/2/3 reflection budgets)."
    )
    parser.add_argument("--llm", required=True, help="LLM name from configs/llms.yaml")
    parser.add_argument("--heal_root", default="HEAL", help="HEAL root directory")
    parser.add_argument("--out_dir", default="outputs", help="Output directory for jsonl")
    parser.add_argument(
        "--validations_root",
        default="validations",
        help="Output directory for validation jsonl",
    )
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Output directory for summary svg",
    )
    parser.add_argument("--config", default="configs/llms.yaml", help="LLM config yaml path")
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Retry count per request when API call fails",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=0,
        help="Parallel worker count. <=0 means auto (equal to total chunk count).",
    )
    parser.add_argument(
        "--max_reflect_rounds",
        type=int,
        default=3,
        help="Maximum simple reflection rounds (control pipeline default is 3).",
    )
    parser.add_argument(
        "--single_behavior_baseline",
        action="store_true",
        help="Only run first row in HEAL/behavior/baseline.csv and print result",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_reflect_rounds < 1:
        print("Invalid --max_reflect_rounds: must be >= 1")
        return 2

    project_root = Path(__file__).resolve().parent
    load_env_file(project_root / ".env")

    config_path = (project_root / args.config).resolve()
    heal_root = (project_root / args.heal_root).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    validations_root = (project_root / args.validations_root).resolve()
    results_root = (project_root / args.results_dir).resolve()

    llm_configs = load_llm_configs(config_path)
    if args.llm not in llm_configs:
        print(
            "Unknown llm `{0}`. Available: {1}".format(
                args.llm,
                ", ".join(sorted(llm_configs.keys())),
            )
        )
        return 2
    llm_config = llm_configs[args.llm]

    if args.single_behavior_baseline:
        return run_control_reflect_single_behavior_baseline(
            llm_config=llm_config,
            heal_root=heal_root,
            max_retries=args.max_retries,
            max_reflect_rounds=args.max_reflect_rounds,
        )

    run_exit_code, output_run_dir = run_control_reflect_all_datasets_with_run_dir(
        llm_config=llm_config,
        heal_root=heal_root,
        out_dir=out_dir,
        max_retries=args.max_retries,
        max_workers=args.max_workers,
        max_reflect_rounds=args.max_reflect_rounds,
    )
    if run_exit_code != 0:
        return run_exit_code

    run_validation_root = validations_root / output_run_dir.name
    run_results_root = results_root / output_run_dir.name
    ensure_dir(run_validation_root)
    ensure_dir(run_results_root)

    print("Control run output: {0}".format(output_run_dir))
    for budget in range(1, args.max_reflect_rounds + 1):
        budget_output_dir = output_run_dir / "reflect_{0}".format(budget)
        files, records, hallucinations = validate_output_run(
            output_run_dir=budget_output_dir,
            validations_root=run_validation_root,
        )
        budget_validation_dir = run_validation_root / budget_output_dir.name
        summary_path = summarize_validation_run(
            validation_run_dir=budget_validation_dir,
            results_dir=run_results_root,
        )
        print(
            "[reflect_{0}] validation: files={1}, records={2}, has_hallucination={3}".format(
                budget,
                files,
                records,
                hallucinations,
            )
        )
        print(
            "[reflect_{0}] validation output: {1}".format(
                budget,
                budget_validation_dir,
            )
        )
        print(
            "[reflect_{0}] summary: {1}".format(
                budget,
                summary_path,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
