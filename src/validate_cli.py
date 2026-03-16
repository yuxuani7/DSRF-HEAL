import argparse
from pathlib import Path

from src.hallucination_validator import validate_output_run
from src.io_utils import ensure_dir, resolve_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate hallucinations from one output run directory."
    )
    parser.add_argument(
        "--output_run_dir",
        required=True,
        help=(
            "Path/name/timestamp of one output run folder, "
            "e.g. outputs/20260303_230137__qwen or 20260303_230137"
        ),
    )
    parser.add_argument(
        "--validations_root",
        default="validations",
        help="Root directory where validation jsonl files are written",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    output_run_dir = resolve_run_dir(
        project_root=project_root,
        raw_value=args.output_run_dir,
        default_root="outputs",
    )
    validations_root = (project_root / args.validations_root).resolve()
    ensure_dir(validations_root)

    files, records, hallucinations = validate_output_run(
        output_run_dir=output_run_dir,
        validations_root=validations_root,
    )
    print(
        "Validation done: files={0}, records={1}, has_hallucination={2}".format(
            files,
            records,
            hallucinations,
        )
    )
    print(
        "Validation output: {0}".format(
            validations_root / output_run_dir.name,
        )
    )
    return 0
