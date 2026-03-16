import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import queue
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.hallucination_validator import validate_prompt_output
from src.heal_loader import load_behavior_baseline_first, load_heal_datasets
from src.io_utils import append_jsonl, ensure_dir, make_run_id, now_iso, output_file_path, output_run_dir
from src.llm_client import LLMClient
from src.progress import MultiDatasetProgress
from src.types import HEALSample, LLMCallError, LLMCallResult, LLMConfig, Message


def _resolve_chunk_size(default_value: int = 100) -> int:
    raw = str(os.getenv("ICSS_TASK_CHUNK_SIZE", "")).strip()
    if not raw:
        return default_value
    try:
        value = int(raw)
    except Exception:
        return default_value
    if value <= 0:
        return default_value
    return value


CHUNK_SIZE = _resolve_chunk_size(100)
RETRY_CYCLE_SLEEP_SEC = 15
MAX_RETRY_ERROR_LOG = 200


@dataclass
class RoundState:
    result: LLMCallResult
    validation: Dict[str, Any]
    retry_errors: List[Dict[str, Any]]
    params: Dict[str, Any]
    last_call_round: int


def _error_to_dict(error: Optional[LLMCallError]) -> Optional[Dict[str, Any]]:
    if error is None:
        return None
    return {
        "type": error.type,
        "message": error.message,
        "traceback": error.traceback,
    }


def _messages_with_user_prompt(messages: Iterable[Message], new_prompt: str) -> List[Message]:
    cloned = [dict(item) for item in messages]
    for idx in range(len(cloned) - 1, -1, -1):
        role = str(cloned[idx].get("role", "")).strip().lower()
        if role == "user":
            cloned[idx]["content"] = new_prompt
            return cloned
    cloned.append({"role": "user", "content": new_prompt})
    return cloned


def _call_with_retry(
    client: LLMClient,
    messages: Iterable[Message],
    max_retries: int,
) -> Tuple[LLMCallResult, List[Dict[str, Any]], Dict[str, Any]]:
    retry_errors: List[Dict[str, Any]] = []
    params_used: Dict[str, Any] = {}
    attempts = 0

    while True:
        for attempt in range(max_retries + 1):
            attempts += 1
            params_used = client.build_params()
            result = client.chat(messages=messages)
            if result.error is None:
                return result, retry_errors, params_used

            if len(retry_errors) < MAX_RETRY_ERROR_LOG:
                retry_errors.append(
                    {
                        "attempt": attempts,
                        "type": result.error.type,
                        "message": result.error.message,
                    }
                )

            if not _is_retryable_error(result.error.type, result.error.message):
                return result, retry_errors, params_used

            if attempt < max_retries:
                time.sleep(2 ** attempt)
        time.sleep(RETRY_CYCLE_SLEEP_SEC)


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


def _empty_validation_payload() -> Dict[str, Any]:
    return {
        "has_hallucination": False,
        "goals": {"node_goals_count": 0, "edge_goals_count": 0},
        "metrics": {
            key: {"count": 0, "denominator": 0, "ratio": 0.0, "percentage": 0.0, "score": "0/0"}
            for key in ("o1", "s", "o2", "r", "o3")
        },
        "hallucination_details": [],
        "parsing": {
            "prompt_parse_notes": [],
            "output_parse_notes": [],
            "constraint_coverage": {
                "object_count": 0,
                "relation_count": 0,
                "to_name_rule_count": 0,
            },
        },
    }


def _build_reflection_prompt(
    original_prompt: str,
    previous_output: str,
    validation: Dict[str, Any],
    reflect_round: int,
) -> str:
    details = validation.get("hallucination_details") or []
    hall_types = sorted(
        list({str(item.get("type", "")).strip().upper() for item in details if item.get("type")})
    )
    notice = (
        "\n\n[Simple Self-Reflection Round {0}]\n"
        "Your previous answer is judged as hallucinated and is not allowed.\n"
        "Previous answer:\n"
        "{1}\n\n"
        "Detected hallucination types: {2}\n"
        "Detected details:\n"
        "{3}\n\n"
        "Please regenerate the answer from scratch.\n"
        "You must strictly follow the original constraints and output JSON only with keys 'node goals' and 'edge goals'."
    ).format(
        reflect_round,
        previous_output,
        json.dumps(hall_types, ensure_ascii=False),
        json.dumps(details, ensure_ascii=False, indent=2),
    )
    return original_prompt + notice


def _build_record(
    run_id: str,
    dataset_name: str,
    sample: HEALSample,
    llm_config: LLMConfig,
    round_state: RoundState,
    reflection_budget: int,
    reflections_used: int,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp": now_iso(),
        "dataset": dataset_name,
        "task_id": sample.task_id,
        "input": {
            "messages": sample.messages,
            "prompt": sample.prompt,
        },
        "llm": {
            "llm_name": llm_config.name,
            "base_url": llm_config.base_url,
            "model": llm_config.model,
            "params": round_state.params,
        },
        "output": {
            "text": round_state.result.text,
            "finish_reason": round_state.result.finish_reason,
        },
        "metrics": {
            "latency_ms": round_state.result.latency_ms,
            "prompt_tokens": round_state.result.prompt_tokens,
            "completion_tokens": round_state.result.completion_tokens,
            "total_tokens": round_state.result.total_tokens,
            "raw_chunk_count": round_state.result.raw_chunk_count,
            "retry_count": len(round_state.retry_errors),
        },
        "error": _error_to_dict(round_state.result.error),
        "retry_errors": round_state.retry_errors,
        "meta": {
            **(sample.meta or {}),
            "simple_reflect": {
                "mode": "simple_reflect_control",
                "reflection_budget": reflection_budget,
                "reflections_used": reflections_used,
                "final_has_hallucination": bool(
                    round_state.validation.get("has_hallucination")
                ),
                "final_metrics": round_state.validation.get("metrics") or {},
                "final_hallucination_details": round_state.validation.get(
                    "hallucination_details"
                )
                or [],
            },
        },
    }


def _infer_sample_by_budget(
    client: LLMClient,
    sample: HEALSample,
    max_retries: int,
    max_reflect_rounds: int,
) -> Dict[int, Tuple[RoundState, int]]:
    original_prompt = sample.prompt
    base_messages = list(sample.messages)
    round_states: Dict[int, RoundState] = {}

    first_result, first_retry_errors, first_params = _call_with_retry(
        client=client,
        messages=base_messages,
        max_retries=max_retries,
    )
    first_validation = _empty_validation_payload()
    if first_result.error is None:
        first_validation = validate_prompt_output(
            prompt=original_prompt,
            output_text=first_result.text,
        )
    round_states[0] = RoundState(
        result=first_result,
        validation=first_validation,
        retry_errors=first_retry_errors,
        params=first_params,
        last_call_round=0,
    )

    for reflect_round in range(1, max_reflect_rounds + 1):
        prev_state = round_states[reflect_round - 1]
        if prev_state.result.error is not None or not prev_state.validation.get(
            "has_hallucination"
        ):
            round_states[reflect_round] = RoundState(
                result=prev_state.result,
                validation=prev_state.validation,
                retry_errors=prev_state.retry_errors,
                params=prev_state.params,
                last_call_round=prev_state.last_call_round,
            )
            continue

        reflected_prompt = _build_reflection_prompt(
            original_prompt=original_prompt,
            previous_output=prev_state.result.text,
            validation=prev_state.validation,
            reflect_round=reflect_round,
        )
        reflected_messages = _messages_with_user_prompt(base_messages, reflected_prompt)
        reflected_result, reflected_retry_errors, reflected_params = _call_with_retry(
            client=client,
            messages=reflected_messages,
            max_retries=max_retries,
        )
        reflected_validation = _empty_validation_payload()
        if reflected_result.error is None:
            reflected_validation = validate_prompt_output(
                prompt=original_prompt,
                output_text=reflected_result.text,
            )
        round_states[reflect_round] = RoundState(
            result=reflected_result,
            validation=reflected_validation,
            retry_errors=reflected_retry_errors,
            params=reflected_params,
            last_call_round=reflect_round,
        )

    by_budget: Dict[int, Tuple[RoundState, int]] = {}
    for budget in range(1, max_reflect_rounds + 1):
        final_state = round_states[budget]
        reflections_used = min(final_state.last_call_round, budget)
        by_budget[budget] = (final_state, reflections_used)
    return by_budget


def _resolve_max_workers(max_workers: Optional[int], task_count: int) -> int:
    if task_count <= 0:
        return 1
    if max_workers is None or max_workers <= 0:
        return task_count
    return max(1, min(max_workers, task_count))


def _drain_progress_queue(progress: MultiDatasetProgress, progress_queue: Any) -> None:
    while True:
        try:
            event = progress_queue.get_nowait()
        except queue.Empty:
            return
        progress.update(
            dataset_name=str(event.get("dataset_name", "")),
            task_id=str(event.get("task_id", "-")),
            success=bool(event.get("success", False)),
            retry_count=int(event.get("retry_count", 0)),
        )


def _worker_control_dataset_chunk(
    llm_config: LLMConfig,
    run_id: str,
    dataset_name: str,
    chunk_index: int,
    samples: List[HEALSample],
    temp_files_by_budget: Dict[int, str],
    max_retries: int,
    max_reflect_rounds: int,
    progress_queue: Any,
) -> Dict[str, Any]:
    client = LLMClient(llm_config)
    processed = 0
    failed = 0

    try:
        for sample in samples:
            budget_states = _infer_sample_by_budget(
                client=client,
                sample=sample,
                max_retries=max_retries,
                max_reflect_rounds=max_reflect_rounds,
            )
            for budget in range(1, max_reflect_rounds + 1):
                round_state, reflections_used = budget_states[budget]
                record = _build_record(
                    run_id=run_id,
                    dataset_name=dataset_name,
                    sample=sample,
                    llm_config=llm_config,
                    round_state=round_state,
                    reflection_budget=budget,
                    reflections_used=reflections_used,
                )
                append_jsonl(Path(temp_files_by_budget[budget]), record)

            processed += 1
            final_state, _ = budget_states[max_reflect_rounds]
            if final_state.result.error is not None:
                failed += 1
            progress_queue.put(
                {
                    "dataset_name": dataset_name,
                    "task_id": sample.task_id,
                    "success": final_state.result.error is None,
                    "retry_count": len(final_state.retry_errors),
                }
            )
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "dataset_name": dataset_name,
            "chunk_index": chunk_index,
            "processed": processed,
            "failed": failed,
            "temp_files_by_budget": temp_files_by_budget,
            "error": "{0}: {1}".format(type(exc).__name__, str(exc)),
        }

    return {
        "dataset_name": dataset_name,
        "chunk_index": chunk_index,
        "processed": processed,
        "failed": failed,
        "temp_files_by_budget": temp_files_by_budget,
        "error": None,
    }


def _merge_chunk_files(final_file: Path, chunk_files: List[Path]) -> None:
    ensure_dir(final_file.parent)
    if final_file.exists():
        final_file.unlink()
    with final_file.open("w", encoding="utf-8") as out_f:
        for part_file in chunk_files:
            with part_file.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)


def _write_worker_error_log(
    run_dir: Path,
    llm_config: LLMConfig,
    run_id: str,
    errors: List[Dict[str, Any]],
) -> None:
    if not errors:
        return
    error_file = run_dir / "control_worker_errors.jsonl"
    for item in errors:
        append_jsonl(
            error_file,
            {
                "run_id": run_id,
                "timestamp": now_iso(),
                "dataset": item.get("dataset_name"),
                "task_id": None,
                "input": None,
                "llm": {
                    "llm_name": llm_config.name,
                    "base_url": llm_config.base_url,
                    "model": llm_config.model,
                    "params": llm_config.default_params,
                },
                "output": None,
                "metrics": None,
                "error": {
                    "type": "ControlWorkerError",
                    "message": item.get("error"),
                    "traceback": None,
                },
                "meta": {"simple_reflect": {"mode": "simple_reflect_control"}},
            },
        )


def run_control_reflect_all_datasets_with_run_dir(
    llm_config: LLMConfig,
    heal_root: Path,
    out_dir: Path,
    max_retries: int,
    max_workers: Optional[int] = None,
    max_reflect_rounds: int = 3,
) -> Tuple[int, Path]:
    if max_reflect_rounds <= 0:
        raise ValueError("max_reflect_rounds must be > 0")

    ensure_dir(out_dir)
    start_time = datetime.now()
    run_id = "simpleReflect_{0}".format(make_run_id(start_time))
    run_dir = output_run_dir(out_dir=out_dir, run_id=run_id, llm_name=llm_config.name)
    ensure_dir(run_dir)

    for budget in range(1, max_reflect_rounds + 1):
        ensure_dir(run_dir / "reflect_{0}".format(budget))

    datasets, load_errors = load_heal_datasets(heal_root)
    progress_items = []
    for dataset in datasets:
        try:
            label = dataset.source_path.relative_to(heal_root).as_posix()
        except ValueError:
            label = dataset.source_path.as_posix()
        progress_items.append((dataset.name, label, len(dataset.samples)))

    progress = MultiDatasetProgress(progress_items)
    progress.start()

    active_datasets = [dataset for dataset in datasets if dataset.samples]
    worker_failures: List[Dict[str, Any]] = []
    chunk_outputs: Dict[int, Dict[str, List[Tuple[int, Path]]]] = {
        budget: {} for budget in range(1, max_reflect_rounds + 1)
    }
    expected_chunks: Dict[str, int] = {}

    if active_datasets:
        chunk_tasks: List[Tuple[str, int, List[HEALSample], Dict[int, str]]] = []
        for dataset in active_datasets:
            total = len(dataset.samples)
            chunk_count = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
            expected_chunks[dataset.name] = chunk_count
            for chunk_index in range(chunk_count):
                start = chunk_index * CHUNK_SIZE
                end = start + CHUNK_SIZE
                chunk_samples = dataset.samples[start:end]
                temp_files_by_budget: Dict[int, str] = {}
                for budget in range(1, max_reflect_rounds + 1):
                    budget_run_dir = run_dir / "reflect_{0}".format(budget)
                    final_file = output_file_path(budget_run_dir, dataset.name, llm_config.name)
                    part_file = (
                        run_dir
                        / ".tmp"
                        / "reflect_{0}".format(budget)
                        / final_file.stem
                        / "part_{0:05d}.jsonl".format(chunk_index)
                    )
                    temp_files_by_budget[budget] = str(part_file)
                chunk_tasks.append((dataset.name, chunk_index, chunk_samples, temp_files_by_budget))

        workers = _resolve_max_workers(max_workers, len(chunk_tasks))
        with mp.Manager() as manager:
            progress_queue = manager.Queue()
            with cf.ProcessPoolExecutor(max_workers=workers) as executor:
                futures: Dict[cf.Future, Tuple[str, int, Dict[int, str]]] = {}
                for dataset_name, chunk_index, chunk_samples, temp_files in chunk_tasks:
                    future = executor.submit(
                        _worker_control_dataset_chunk,
                        llm_config,
                        run_id,
                        dataset_name,
                        chunk_index,
                        chunk_samples,
                        temp_files,
                        max_retries,
                        max_reflect_rounds,
                        progress_queue,
                    )
                    futures[future] = (dataset_name, chunk_index, temp_files)

                while futures:
                    _drain_progress_queue(progress, progress_queue)
                    done_futures = [future for future in futures.keys() if future.done()]
                    for future in done_futures:
                        dataset_name, chunk_index, temp_files = futures.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:  # pylint: disable=broad-except
                            worker_failures.append(
                                {
                                    "dataset_name": dataset_name,
                                    "chunk_index": chunk_index,
                                    "error": "{0}: {1}".format(type(exc).__name__, str(exc)),
                                }
                            )
                            continue
                        if result.get("error"):
                            worker_failures.append(
                                {
                                    "dataset_name": dataset_name,
                                    "chunk_index": chunk_index,
                                    "error": str(result.get("error")),
                                }
                            )
                        else:
                            returned_temp = result.get("temp_files_by_budget") or temp_files
                            for budget in range(1, max_reflect_rounds + 1):
                                part_path = Path(str(returned_temp[budget]))
                                chunk_outputs[budget].setdefault(dataset_name, []).append(
                                    (chunk_index, part_path)
                                )
                    time.sleep(0.05)
                _drain_progress_queue(progress, progress_queue)

    progress.finish()

    if load_errors:
        load_error_file = run_dir / "dataset_load_errors.jsonl"
        for err in load_errors:
            append_jsonl(
                load_error_file,
                {
                    "run_id": run_id,
                    "timestamp": now_iso(),
                    "dataset": err.dataset_name,
                    "task_id": None,
                    "input": None,
                    "llm": {
                        "llm_name": llm_config.name,
                        "base_url": llm_config.base_url,
                        "model": llm_config.model,
                        "params": llm_config.default_params,
                    },
                    "output": None,
                    "metrics": None,
                    "error": {
                        "type": err.error_type,
                        "message": err.message,
                        "traceback": err.traceback,
                    },
                    "source_path": err.source_path,
                    "meta": {"simple_reflect": {"mode": "simple_reflect_control"}},
                },
            )

    if worker_failures:
        _write_worker_error_log(
            run_dir=run_dir,
            llm_config=llm_config,
            run_id=run_id,
            errors=worker_failures,
        )

    for budget in range(1, max_reflect_rounds + 1):
        budget_run_dir = run_dir / "reflect_{0}".format(budget)
        for dataset in active_datasets:
            dataset_name = dataset.name
            final_file = output_file_path(budget_run_dir, dataset_name, llm_config.name)
            sorted_parts = sorted(
                chunk_outputs[budget].get(dataset_name, []), key=lambda item: item[0]
            )
            expected = expected_chunks.get(dataset_name, 0)
            if len(sorted_parts) != expected:
                worker_failures.append(
                    {
                        "dataset_name": dataset_name,
                        "chunk_index": "budget_{0}".format(budget),
                        "error": "Chunk count mismatch: expected={0}, got={1}".format(
                            expected, len(sorted_parts)
                        ),
                    }
                )
            part_paths: List[Path] = []
            for part_index, part_path in sorted_parts:
                if not part_path.exists():
                    worker_failures.append(
                        {
                            "dataset_name": dataset_name,
                            "chunk_index": part_index,
                            "error": "Temp chunk file missing: {0}".format(part_path),
                        }
                    )
                    continue
                part_paths.append(part_path)
            if part_paths:
                _merge_chunk_files(final_file, part_paths)

    temp_root = run_dir / ".tmp"
    if temp_root.exists() and temp_root.is_dir():
        shutil.rmtree(temp_root, ignore_errors=True)

    return 0, run_dir


def run_control_reflect_single_behavior_baseline(
    llm_config: LLMConfig,
    heal_root: Path,
    max_retries: int,
    max_reflect_rounds: int = 3,
) -> int:
    sample = load_behavior_baseline_first(heal_root)
    if sample is None:
        print("No sample found at HEAL/behavior/baseline.csv")
        return 1

    client = LLMClient(llm_config)
    by_budget = _infer_sample_by_budget(
        client=client,
        sample=sample,
        max_retries=max_retries,
        max_reflect_rounds=max_reflect_rounds,
    )
    final_state, reflections_used = by_budget[max_reflect_rounds]
    if final_state.result.error is not None:
        print(
            "Request failed: {0}: {1}".format(
                final_state.result.error.type, final_state.result.error.message
            )
        )
        return 1
    print(final_state.result.text)
    print(
        "\n[control_reflect] budget={0}, reflections_used={1}, has_hallucination={2}".format(
            max_reflect_rounds,
            reflections_used,
            final_state.validation.get("has_hallucination"),
        )
    )
    return 0
