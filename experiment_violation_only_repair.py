import argparse
import sys
from pathlib import Path

from src.config import load_env_file, load_llm_configs
from src.hallucination_validator import validate_output_run
from src.io_utils import ensure_dir
from src.repair_ablation_runner import (
    MODE_VIOLATION_ONLY,
    run_ablation_all_datasets_with_run_dir,
    run_ablation_single_behavior_baseline,
)
from src.results_summary import summarize_validation_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Violation-Only Repair ablation."
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
        "--max_repair_rounds",
        type=int,
        default=1,
        help="Maximum repair rounds after initial generation.",
    )
    parser.add_argument(
        "--max_total_llm_calls_per_sample",
        type=int,
        default=4,
        help="Hard budget of total LLM calls per sample (including initial call).",
    )
    parser.add_argument(
        "--single_behavior_baseline",
        action="store_true",
        help="Only run first row in HEAL/behavior/baseline.csv and print result",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_repair_rounds < 0:
        print("Invalid --max_repair_rounds: must be >= 0")
        return 2
    if args.max_total_llm_calls_per_sample < 1:
        print("Invalid --max_total_llm_calls_per_sample: must be >= 1")
        return 2

    project_root = Path(__file__).resolve().parent
    load_env_file(project_root / ".env")

    config_path = (project_root / args.config).resolve()
    heal_root = (project_root / args.heal_root).resolve()
    out_dir = (project_root / args.out_dir).resolve()
    validations_root = (project_root / args.validations_root).resolve()
    results_dir = (project_root / args.results_dir).resolve()

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
        return run_ablation_single_behavior_baseline(
            llm_config=llm_config,
            heal_root=heal_root,
            mode=MODE_VIOLATION_ONLY,
            max_retries=args.max_retries,
            max_total_llm_calls_per_sample=args.max_total_llm_calls_per_sample,
            max_repair_rounds=args.max_repair_rounds,
        )

    run_exit_code, output_run_dir = run_ablation_all_datasets_with_run_dir(
        llm_config=llm_config,
        heal_root=heal_root,
        out_dir=out_dir,
        mode=MODE_VIOLATION_ONLY,
        max_retries=args.max_retries,
        max_total_llm_calls_per_sample=args.max_total_llm_calls_per_sample,
        max_repair_rounds=args.max_repair_rounds,
        max_workers=args.max_workers,
    )
    if run_exit_code != 0:
        return run_exit_code

    ensure_dir(validations_root)
    files, records, hallucinations = validate_output_run(
        output_run_dir=output_run_dir,
        validations_root=validations_root,
    )
    validation_run_dir = validations_root / output_run_dir.name
    summary_path = summarize_validation_run(
        validation_run_dir=validation_run_dir,
        results_dir=results_dir,
    )

    print("Violation-Only Repair output: {0}".format(output_run_dir))
    print(
        "Validation done: files={0}, records={1}, has_hallucination={2}".format(
            files,
            records,
            hallucinations,
        )
    )
    print("Validation output: {0}".format(validation_run_dir))
    print("Summary image generated: {0}".format(summary_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
