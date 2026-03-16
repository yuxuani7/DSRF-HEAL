import argparse
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
CHECK_RETRY_SLEEP_SEC = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all experiment scripts for all models in configs/llms.yaml."
    )
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
    parser.add_argument(
        "--on_check_fail",
        choices=["skip_model", "stop"],
        default="skip_model",
        help="Behavior when model connectivity check fails.",
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


def _check_model_connectivity(project_root: Path, config_path: Path, heal_root: Path) -> Dict[str, Dict[str, str]]:
    load_env_file(project_root / ".env")
    llm_configs = load_llm_configs(config_path)
    sample = load_behavior_baseline_first(heal_root)
    if sample is None:
        raise RuntimeError("No sample found at HEAL/behavior/baseline.csv")

    model_names = sorted(llm_configs.keys())
    total = len(model_names)
    print(
        "[CHECK] single sample task_id={0}, models={1}".format(sample.task_id, total),
        flush=True,
    )

    results: Dict[str, Dict[str, str]] = {}
    for idx, llm_name in enumerate(model_names, start=1):
        cfg = llm_configs[llm_name]
        print(
            "[CHECK {0}/{1}] model={2} start".format(idx, total, llm_name),
            flush=True,
        )
        status = "ok"
        message = ""
        has_hall = "unknown"
        attempts = 0
        total_start = time.perf_counter()
        while True:
            attempts += 1
            attempt_start = time.perf_counter()
            try:
                client = LLMClient(cfg)
                call_result = client.chat(messages=sample.messages)
                if call_result.error is not None:
                    err_type = call_result.error.type
                    err_msg = call_result.error.message
                    if _is_retryable_error(err_type, err_msg):
                        print(
                            "[CHECK {0}/{1}] model={2} retryable_error attempt={3} sleep={4}s err={5}: {6}".format(
                                idx,
                                total,
                                llm_name,
                                attempts,
                                CHECK_RETRY_SLEEP_SEC,
                                err_type,
                                str(err_msg)[:120],
                            ),
                            flush=True,
                        )
                        time.sleep(CHECK_RETRY_SLEEP_SEC)
                        continue
                    status = "fail"
                    message = "{0}: {1}".format(err_type, err_msg)
                    break
                validation = validate_prompt_output(prompt=sample.prompt, output_text=call_result.text or "")
                has_hall = str(bool(validation.get("has_hallucination")))
                message = "text_len={0}".format(len(call_result.text or ""))
                break
            except Exception as exc:  # pylint: disable=broad-except
                err_type = type(exc).__name__
                err_msg = str(exc)
                if _is_retryable_error(err_type, err_msg):
                    elapsed = int((time.perf_counter() - attempt_start) * 1000)
                    print(
                        "[CHECK {0}/{1}] model={2} retryable_exception attempt={3} elapsed_ms={4} sleep={5}s err={6}: {7}".format(
                            idx,
                            total,
                            llm_name,
                            attempts,
                            elapsed,
                            CHECK_RETRY_SLEEP_SEC,
                            err_type,
                            err_msg[:120],
                        ),
                        flush=True,
                    )
                    time.sleep(CHECK_RETRY_SLEEP_SEC)
                    continue
                status = "fail"
                message = "{0}: {1}".format(err_type, err_msg)
                break

        latency_ms = int((time.perf_counter() - total_start) * 1000)
        results[llm_name] = {
            "status": status,
            "has_hallucination": has_hall,
            "latency_ms": str(latency_ms),
            "message": message[:300],
        }
        print(
            "[CHECK {0}/{1}] model={2} done status={3} latency_ms={4}".format(
                idx,
                total,
                llm_name,
                status,
                latency_ms,
            ),
            flush=True,
        )
    return results


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


def _run_experiment_for_model(
    model_name: str,
    experiment_keys: List[str],
    args: argparse.Namespace,
) -> List[Tuple[str, int]]:
    outcomes: List[Tuple[str, int]] = []
    total = len(experiment_keys)
    for idx, key in enumerate(experiment_keys, start=1):
        cmd = _build_cmd(script_key=key, llm_name=model_name, args=args)
        t0 = time.perf_counter()
        print("[RUN {0}/{1}] model={2} exp={3} start".format(idx, total, model_name, key), flush=True)
        print("           cmd={0}".format(" ".join(cmd)), flush=True)
        proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent), check=False)
        elapsed = int((time.perf_counter() - t0) * 1000)
        print(
            "[RUN {0}/{1}] model={2} exp={3} done return_code={4} elapsed_ms={5}".format(
                idx,
                total,
                model_name,
                key,
                int(proc.returncode),
                elapsed,
            ),
            flush=True,
        )
        outcomes.append((key, int(proc.returncode)))
        if proc.returncode != 0 and args.stop_on_error:
            break
    return outcomes


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = (project_root / args.config).resolve()
    heal_root = (project_root / args.heal_root).resolve()

    try:
        experiment_keys = _resolve_experiment_names(args.experiments)
    except ValueError as exc:
        print(str(exc))
        return 2

    if not args.skip_connectivity_check or args.check_only:
        print("=== Connectivity Check Start ===", flush=True)
        check_results = _check_model_connectivity(
            project_root=project_root,
            config_path=config_path,
            heal_root=heal_root,
        )
        for llm_name, payload in check_results.items():
            print(
                "{0}\tstatus={1}\thas_hall={2}\tlatency_ms={3}\tmsg={4}".format(
                    llm_name,
                    payload["status"],
                    payload["has_hallucination"],
                    payload["latency_ms"],
                    payload["message"],
                ),
                flush=True,
            )
        print("=== Connectivity Check End ===", flush=True)
        if args.check_only:
            return 0
    else:
        load_env_file(project_root / ".env")
        check_results = {}

    llm_configs = load_llm_configs(config_path)
    runnable_models: List[str] = []
    failed_models: List[str] = []
    for llm_name in sorted(llm_configs.keys()):
        if args.skip_connectivity_check:
            runnable_models.append(llm_name)
            continue
        status = (check_results.get(llm_name) or {}).get("status")
        if status == "ok":
            runnable_models.append(llm_name)
        else:
            failed_models.append(llm_name)

    if failed_models and args.on_check_fail == "stop":
        print("Connectivity check failed, stop by --on_check_fail=stop:", flush=True)
        print(",".join(failed_models), flush=True)
        return 1

    if failed_models and args.on_check_fail == "skip_model":
        print("Skip models with failed connectivity: {0}".format(",".join(failed_models)), flush=True)

    if not runnable_models:
        print("No runnable models after connectivity check.", flush=True)
        return 1

    overall_failures: List[Tuple[str, str, int]] = []
    for model_name in runnable_models:
        outcomes = _run_experiment_for_model(
            model_name=model_name,
            experiment_keys=experiment_keys,
            args=args,
        )
        for exp_key, code in outcomes:
            if code != 0:
                overall_failures.append((model_name, exp_key, code))
        if overall_failures and args.stop_on_error:
            break

    print("=== Full Experiments Summary ===", flush=True)
    print("runnable_models={0}".format(",".join(runnable_models)), flush=True)
    if failed_models:
        print("connectivity_failed_models={0}".format(",".join(failed_models)), flush=True)
    if not overall_failures:
        print("all experiment commands finished with code 0", flush=True)
        return 0
    print("failed commands:", flush=True)
    for model_name, exp_key, code in overall_failures:
        print("model={0}, exp={1}, return_code={2}".format(model_name, exp_key, code), flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
