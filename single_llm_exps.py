import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

from src.config import load_env_file, load_llm_configs
from src.hallucination_validator import validate_prompt_output
from src.heal_loader import load_behavior_baseline_first
from src.llm_client import LLMClient


EXPERIMENTS = {
    "direct_prompting": "experiment_direct_prompting.py",
    "diagnosis_guided_pipeline": "experiment_diagnosis_guided_pipeline.py",
    "naive_self_reflection": "experiment_naive_self_reflection.py",
    "constraint_only_repair": "experiment_constraint_only_repair.py",
    "violation_only_repair": "experiment_violation_only_repair.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all experiment scripts for one selected LLM."
    )
    parser.add_argument("--llm", required=True, help="LLM name from configs/llms.yaml")
    parser.add_argument("--heal_root", default="HEAL", help="HEAL root directory")
    parser.add_argument("--out_dir", default="outputs", help="Output directory for jsonl")
    parser.add_argument("--validations_root", default="validations", help="Validation output root")
    parser.add_argument("--results_dir", default="results", help="Summary output root")
    parser.add_argument("--config", default="configs/llms.yaml", help="LLM config yaml path")
    parser.add_argument("--max_retries", type=int, default=3, help="Retry count per API request")
    parser.add_argument(
        "--max_workers",
        type=int,
        default=0,
        help="Parallel workers for each experiment script (<=0 means auto).",
    )
    parser.add_argument(
        "--task_chunk_size",
        type=int,
        default=100,
        help="How many tasks per chunk for parallel processing in runners.",
    )
    parser.add_argument(
        "--max_refine_rounds",
        type=int,
        default=3,
        help="Used by diagnosis-guided pipeline script.",
    )
    parser.add_argument(
        "--max_reflect_rounds",
        type=int,
        default=3,
        help="Used by naive self-reflection script.",
    )
    parser.add_argument(
        "--max_repair_rounds",
        type=int,
        default=1,
        help="Used by constraint-only and violation-only scripts.",
    )
    parser.add_argument(
        "--max_total_llm_calls_per_sample",
        type=int,
        default=4,
        help="Used by constraint-only and violation-only scripts.",
    )
    parser.add_argument(
        "--single_behavior_baseline",
        action="store_true",
        help="Run single sample mode for every experiment script.",
    )
    parser.add_argument(
        "--experiments",
        default="all",
        help="Comma-separated subset from: {0}. Default: all".format(
            ",".join(EXPERIMENTS.keys())
        ),
    )
    parser.add_argument(
        "--skip_connectivity_check",
        action="store_true",
        help="Skip pre-run model connectivity check.",
    )
    parser.add_argument(
        "--check_only",
        action="store_true",
        help="Only perform connectivity check, do not run experiments.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately when one experiment command fails.",
    )
    return parser.parse_args()


def _resolve_experiment_names(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return list(EXPERIMENTS.keys())
    names = [item.strip() for item in raw.split(",") if item.strip()]
    invalid = [name for name in names if name not in EXPERIMENTS]
    if invalid:
        raise ValueError(
            "Unknown experiments: {0}. Valid: {1}".format(
                ",".join(invalid), ",".join(EXPERIMENTS.keys())
            )
        )
    return names


def _is_retryable_error(error_type: str, message: str) -> bool:
    et = str(error_type or "").strip().lower()
    msg = str(message or "").strip().lower()
    unretryable_tokens = (
        "authentication",
        "permission",
        "invalid_api_key",
        "invalidrequest",
        "notfound",
        "not found",
        "unauthorized",
        "forbidden",
    )
    retryable_tokens = (
        "apiconnection",
        "ratelimit",
        "timeout",
        "temporarily",
        "internalserver",
        "service unavailable",
        "429",
        "502",
        "503",
        "504",
        "connection error",
    )
    if any(token in et for token in unretryable_tokens):
        return False
    if any(token in msg for token in unretryable_tokens):
        return False
    if any(token in et for token in retryable_tokens):
        return True
    if any(token in msg for token in retryable_tokens):
        return True
    return True


def _check_single_model_connectivity(
    project_root: Path,
    config_path: Path,
    heal_root: Path,
    llm_name: str,
) -> Dict[str, str]:
    load_env_file(project_root / ".env")
    llm_configs = load_llm_configs(config_path)
    if llm_name not in llm_configs:
        raise ValueError(
            "Unknown llm `{0}`. Available: {1}".format(
                llm_name,
                ",".join(sorted(llm_configs.keys())),
            )
        )

    sample = load_behavior_baseline_first(heal_root)
    if sample is None:
        raise RuntimeError("No sample found at HEAL/behavior/baseline.csv")

    cfg = llm_configs[llm_name]
    print("[CHECK] llm={0}, task_id={1}".format(llm_name, sample.task_id), flush=True)
    total_start = time.perf_counter()
    attempts = 0
    while True:
        attempts += 1
        try:
            client = LLMClient(cfg)
            call_result = client.chat(messages=sample.messages)
            if call_result.error is not None:
                if _is_retryable_error(call_result.error.type, call_result.error.message):
                    print(
                        "[CHECK] retryable_error attempt={0} sleep=15s err={1}: {2}".format(
                            attempts,
                            call_result.error.type,
                            str(call_result.error.message)[:120],
                        ),
                        flush=True,
                    )
                    time.sleep(15)
                    continue
                return {
                    "status": "fail",
                    "has_hallucination": "unknown",
                    "latency_ms": str(int((time.perf_counter() - total_start) * 1000)),
                    "message": "{0}: {1}".format(call_result.error.type, call_result.error.message)[:300],
                }

            validation = validate_prompt_output(prompt=sample.prompt, output_text=call_result.text or "")
            return {
                "status": "ok",
                "has_hallucination": str(bool(validation.get("has_hallucination"))),
                "latency_ms": str(int((time.perf_counter() - total_start) * 1000)),
                "message": "text_len={0}".format(len(call_result.text or "")),
            }
        except Exception as exc:  # pylint: disable=broad-except
            if _is_retryable_error(type(exc).__name__, str(exc)):
                print(
                    "[CHECK] retryable_exception attempt={0} sleep=15s err={1}: {2}".format(
                        attempts,
                        type(exc).__name__,
                        str(exc)[:120],
                    ),
                    flush=True,
                )
                time.sleep(15)
                continue
            return {
                "status": "fail",
                "has_hallucination": "unknown",
                "latency_ms": str(int((time.perf_counter() - total_start) * 1000)),
                "message": "{0}: {1}".format(type(exc).__name__, str(exc))[:300],
            }


def _build_base_cmd(
    script: str,
    llm_name: str,
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        sys.executable,
        script,
        "--llm",
        llm_name,
        "--heal_root",
        args.heal_root,
        "--out_dir",
        args.out_dir,
        "--validations_root",
        args.validations_root,
        "--results_dir",
        args.results_dir,
        "--config",
        args.config,
        "--max_retries",
        str(args.max_retries),
        "--max_workers",
        str(args.max_workers),
    ]
    if args.single_behavior_baseline:
        cmd.append("--single_behavior_baseline")
    return cmd


def _build_cmd(script_key: str, llm_name: str, args: argparse.Namespace) -> List[str]:
    script = EXPERIMENTS[script_key]
    cmd = _build_base_cmd(script=script, llm_name=llm_name, args=args)
    if script_key == "diagnosis_guided_pipeline":
        cmd.extend(["--max_refine_rounds", str(args.max_refine_rounds)])
    elif script_key == "naive_self_reflection":
        cmd.extend(["--max_reflect_rounds", str(args.max_reflect_rounds)])
    elif script_key in ("constraint_only_repair", "violation_only_repair"):
        cmd.extend(
            [
                "--max_repair_rounds",
                str(args.max_repair_rounds),
                "--max_total_llm_calls_per_sample",
                str(args.max_total_llm_calls_per_sample),
            ]
        )
    return cmd


def main() -> int:
    args = parse_args()
    if args.task_chunk_size <= 0:
        print("Invalid --task_chunk_size: must be > 0")
        return 2

    project_root = Path(__file__).resolve().parent
    config_path = (project_root / args.config).resolve()
    heal_root = (project_root / args.heal_root).resolve()
    try:
        experiment_keys = _resolve_experiment_names(args.experiments)
    except ValueError as exc:
        print(str(exc))
        return 2

    if not args.skip_connectivity_check or args.check_only:
        print("=== Single-LLM Connectivity Check Start ===", flush=True)
        try:
            payload = _check_single_model_connectivity(
                project_root=project_root,
                config_path=config_path,
                heal_root=heal_root,
                llm_name=args.llm,
            )
        except Exception as exc:  # pylint: disable=broad-except
            print("check failed: {0}: {1}".format(type(exc).__name__, str(exc)), flush=True)
            return 1
        print(
            "{0}\tstatus={1}\thas_hall={2}\tlatency_ms={3}\tmsg={4}".format(
                args.llm,
                payload["status"],
                payload["has_hallucination"],
                payload["latency_ms"],
                payload["message"],
            ),
            flush=True,
        )
        print("=== Single-LLM Connectivity Check End ===", flush=True)
        if payload["status"] != "ok":
            return 1
        if args.check_only:
            return 0

    env = os.environ.copy()
    env["ICSS_TASK_CHUNK_SIZE"] = str(args.task_chunk_size)

    failures: List[Tuple[str, int]] = []
    total = len(experiment_keys)
    for idx, exp_key in enumerate(experiment_keys, start=1):
        cmd = _build_cmd(exp_key, args.llm, args)
        t0 = time.perf_counter()
        print(
            "[RUN {0}/{1}] llm={2} exp={3} start chunk_size={4}".format(
                idx, total, args.llm, exp_key, args.task_chunk_size
            ),
            flush=True,
        )
        print("           cmd={0}".format(" ".join(cmd)), flush=True)
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            env=env,
            check=False,
        )
        elapsed = int((time.perf_counter() - t0) * 1000)
        print(
            "[RUN {0}/{1}] llm={2} exp={3} done return_code={4} elapsed_ms={5}".format(
                idx, total, args.llm, exp_key, int(proc.returncode), elapsed
            ),
            flush=True,
        )
        if proc.returncode != 0:
            failures.append((exp_key, int(proc.returncode)))
            if args.stop_on_error:
                break

    print("=== Single-LLM Experiment Summary ===", flush=True)
    if not failures:
        print("llm={0}, all selected experiments finished with code 0".format(args.llm), flush=True)
        return 0

    print("llm={0}, failed experiments:".format(args.llm), flush=True)
    for exp_key, code in failures:
        print("exp={0}, return_code={1}".format(exp_key, code), flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
