import argparse
from pathlib import Path

from src.io_utils import resolve_run_dir
from src.results_summary import summarize_validation_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hallucination summary table image from one validations run folder."
    )
    parser.add_argument(
        "--validation_run_dir",
        required=True,
        help=(
            "Path/name/timestamp of one validations run folder, "
            "e.g. validations/20260303_230137__qwen or 20260303_230137"
        ),
    )
    parser.add_argument(
        "--results_dir",
        default="results",
        help="Output directory for summary table image",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    validation_run_dir = resolve_run_dir(
        project_root=project_root,
        raw_value=args.validation_run_dir,
        default_root="validations",
    )
    results_dir = (project_root / args.results_dir).resolve()

    output_path = summarize_validation_run(
        validation_run_dir=validation_run_dir,
        results_dir=results_dir,
    )
    print("Summary image generated: {0}".format(output_path))
    return 0
