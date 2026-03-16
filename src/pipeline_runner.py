import concurrent.futures as cf
import multiprocessing as mp
import os
import queue
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.diagnose import diagnose_failure
from src.hallucination_validator import parse_prompt_constraints, validate_prompt_output
from src.heal_loader import load_behavior_baseline_first, load_heal_datasets
from src.io_utils import append_jsonl, ensure_dir, make_run_id, now_iso, output_file_path, output_run_dir
from src.llm_client import LLMClient
from src.pipeline_routes import (
    ROUTE_ABSTAIN,
    ROUTE_COMMON_REPAIR,
    ROUTE_NOOP,
    ROUTE_STRUCTURAL_REPAIR,
    route_refine_budget,
    select_repair_route,
)
from src.progress import MultiDatasetProgress
from src.repair_common import build_common_repair_prompt
from src.repair_structural_relation import (
    apply_target_repairs,
    build_final_answer_text,
    build_target_choice_prompt,
    extract_node_goals_from_output,
    extract_target_repairs,
)
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


PIPELINE_MODE = "taxonomy_aware_repair"
CHUNK_SIZE = _resolve_chunk_size(100)
RETRY_CYCLE_SLEEP_SEC = 15
MAX_RETRY_ERROR_LOG = 200


@dataclass
class PipelineOutcome:
    result: LLMCallResult
    validation: Dict[str, Any]
    params: Dict[str, Any]
    retry_errors: List[Dict[str, Any]]
    llm_call_count: int
    meta_pipeline: Dict[str, Any]


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
    overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[LLMCallResult, List[Dict[str, Any]], Dict[str, Any], int]:
    retry_errors: List[Dict[str, Any]] = []
    params_used: Dict[str, Any] = {}
    attempts = 0

    while True:
        for attempt in range(max_retries + 1):
            attempts += 1
            params_used = client.build_params(overrides)
            result = client.chat(messages=messages, overrides=overrides)
            if result.error is None:
                return result, retry_errors, params_used, attempts

            if len(retry_errors) < MAX_RETRY_ERROR_LOG:
                retry_errors.append(
                    {
                        "attempt": attempts,
                        "type": result.error.type,
                        "message": result.error.message,
                    }
                )

            if not _is_retryable_error(result.error.type, result.error.message):
                return result, retry_errors, params_used, attempts

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


def _tag_retry_errors(
    retry_errors: List[Dict[str, Any]],
    stage: str,
    round_idx: int,
) -> List[Dict[str, Any]]:
    tagged: List[Dict[str, Any]] = []
    for item in retry_errors:
        one = dict(item)
        one["stage"] = stage
        one["round"] = round_idx
        tagged.append(one)
    return tagged


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


def _synthetic_result(text: str, latency_ms: int = 0, raw_chunk_count: int = 1) -> LLMCallResult:
    return LLMCallResult(
        text=text,
        finish_reason="stop",
        latency_ms=latency_ms,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        raw_chunk_count=raw_chunk_count,
        error=None,
    )


def _attempt_common_repair(
    client: LLMClient,
    constraints: Any,
    base_messages: List[Message],
    original_prompt: str,
    current_validation: Dict[str, Any],
    diagnosis: Dict[str, Any],
    max_retries: int,
    round_idx: int,
) -> Tuple[LLMCallResult, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], int]:
    repair_prompt = build_common_repair_prompt(
        original_prompt=original_prompt,
        constraints=constraints,
        validation_result=current_validation,
        diagnosis=diagnosis,
    )
    repair_messages = _messages_with_user_prompt(base_messages, repair_prompt)
    repair_result, retry_errors, params, attempts = _call_with_retry(
        client=client,
        messages=repair_messages,
        max_retries=max_retries,
        overrides=None,
    )
    retry_log = _tag_retry_errors(retry_errors, stage="common_repair", round_idx=round_idx)
    return repair_result, params, {}, retry_log, attempts


def _attempt_structural_relation_repair(
    client: LLMClient,
    constraints: Any,
    current_result: LLMCallResult,
    current_validation: Dict[str, Any],
    max_retries: int,
    round_idx: int,
) -> Tuple[LLMCallResult, Dict[str, Any], Dict[str, Any], List[Dict[str, Any]], int]:
    target_prompt = build_target_choice_prompt(
        output_text=current_result.text,
        validation_result=current_validation,
        constraints=constraints,
    )
    target_messages = [{"role": "user", "content": target_prompt}]
    target_result, retry_errors, params, attempts = _call_with_retry(
        client=client,
        messages=target_messages,
        max_retries=max_retries,
        overrides={"stream": False},
    )
    retry_log = _tag_retry_errors(retry_errors, stage="structural_target", round_idx=round_idx)
    if target_result.error is not None:
        return target_result, params, {}, retry_log, attempts

    repairs = extract_target_repairs(target_result.text)
    node_goals = extract_node_goals_from_output(current_result.text)
    edge_goals = apply_target_repairs(
        output_text=current_result.text,
        validation_result=current_validation,
        constraints=constraints,
        repairs=repairs,
    )
    final_text = build_final_answer_text(node_goals=node_goals, edge_goals=edge_goals)
    synthetic = LLMCallResult(
        text=final_text,
        finish_reason=target_result.finish_reason or "stop",
        latency_ms=target_result.latency_ms,
        prompt_tokens=target_result.prompt_tokens,
        completion_tokens=target_result.completion_tokens,
        total_tokens=target_result.total_tokens,
        raw_chunk_count=target_result.raw_chunk_count,
        error=None,
    )
    return synthetic, params, {}, retry_log, attempts


def _build_pipeline_record(
    run_id: str,
    dataset_name: str,
    sample: HEALSample,
    llm_config: LLMConfig,
    params: Dict[str, Any],
    outcome: PipelineOutcome,
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
            "params": params,
        },
        "output": {
            "text": outcome.result.text,
            "finish_reason": outcome.result.finish_reason,
        },
        "metrics": {
            "latency_ms": outcome.result.latency_ms,
            "prompt_tokens": outcome.result.prompt_tokens,
            "completion_tokens": outcome.result.completion_tokens,
            "total_tokens": outcome.result.total_tokens,
            "raw_chunk_count": outcome.result.raw_chunk_count,
            "retry_count": len(outcome.retry_errors),
            "pipeline_llm_call_count": outcome.llm_call_count,
        },
        "error": _error_to_dict(outcome.result.error),
        "retry_errors": outcome.retry_errors,
        "meta": {
            **(sample.meta or {}),
            "pipeline": outcome.meta_pipeline,
        },
    }


def _pipeline_infer_one_sample(
    client: LLMClient,
    sample: HEALSample,
    max_retries: int,
    max_refine_rounds: int,
) -> PipelineOutcome:
    original_prompt = sample.prompt
    base_messages = list(sample.messages)
    constraints = parse_prompt_constraints(original_prompt)
    sample_meta = dict(sample.meta or {})
    sample_meta["dataset_name"] = sample.dataset_name

    initial_result, initial_retry_errors, initial_params, initial_attempts = _call_with_retry(
        client=client,
        messages=base_messages,
        max_retries=max_retries,
        overrides=None,
    )
    retry_log = _tag_retry_errors(initial_retry_errors, stage="initial", round_idx=0)

    initial_validation = _empty_validation_payload()
    working_validation = _empty_validation_payload()
    if initial_result.error is None:
        initial_validation = validate_prompt_output(
            prompt=original_prompt,
            output_text=initial_result.text,
        )
        working_validation = validate_prompt_output(
            prompt=original_prompt,
            output_text=initial_result.text,
        )

    if initial_result.error is not None:
        meta_pipeline = {
            "pipeline_mode": PIPELINE_MODE,
            "pipeline_route": "error",
            "diagnosed_categories": [],
            "repairability": "unavailable",
            "should_abstain": False,
            "pipeline_refine_rounds_used": 0,
            "pipeline_refine_rounds_max": 0,
            "pipeline_refine_rounds_global_max": max_refine_rounds,
            "initial_has_hallucination": False,
            "initial_metrics": initial_validation.get("metrics") or {},
            "pipeline_final_has_hallucination": False,
            "pipeline_final_metrics": initial_validation.get("metrics") or {},
            "pipeline_final_hallucination_details": [],
            "route_success": False,
            "llm_call_count": initial_attempts,
            "pipeline_route_history": [],
            "route_attempt_counts": {},
        }
        return PipelineOutcome(
            result=initial_result,
            validation=initial_validation,
            params=initial_params,
            retry_errors=retry_log,
            llm_call_count=initial_attempts,
            meta_pipeline=meta_pipeline,
        )

    initial_has_hall = bool(initial_validation.get("has_hallucination"))
    if not bool(working_validation.get("has_hallucination")):
        meta_pipeline = {
            "pipeline_mode": PIPELINE_MODE,
            "pipeline_route": ROUTE_NOOP,
            "diagnosed_categories": [],
            "repairability": "none",
            "should_abstain": False,
            "pipeline_refine_rounds_used": 0,
            "pipeline_refine_rounds_max": 0,
            "pipeline_refine_rounds_global_max": max_refine_rounds,
            "initial_has_hallucination": initial_has_hall,
            "initial_metrics": initial_validation.get("metrics") or {},
            "pipeline_final_has_hallucination": False,
            "pipeline_final_metrics": working_validation.get("metrics") or {},
            "pipeline_final_hallucination_details": [],
            "route_success": True,
            "llm_call_count": initial_attempts,
            "pipeline_route_history": [ROUTE_NOOP],
            "route_attempt_counts": {},
        }
        return PipelineOutcome(
            result=initial_result,
            validation=initial_validation,
            params=initial_params,
            retry_errors=retry_log,
            llm_call_count=initial_attempts,
            meta_pipeline=meta_pipeline,
        )

    final_result = initial_result
    final_validation = working_validation
    final_params = initial_params
    llm_call_count = initial_attempts
    rounds_used = 0
    route_attempt_counts: Dict[str, int] = {}
    route_history: List[str] = []
    primary_route = ROUTE_NOOP
    first_diagnosis: Dict[str, Any] = {}

    while bool(final_validation.get("has_hallucination")) and rounds_used < max_refine_rounds:
        diagnosis = diagnose_failure(
            validation_result=final_validation,
            prompt_constraints=constraints,
            output_text=final_result.text,
            sample_meta=sample_meta,
        )
        if not first_diagnosis:
            first_diagnosis = diagnosis

        route = select_repair_route(diagnosis)
        if primary_route == ROUTE_NOOP:
            primary_route = route
        route_history.append(route)

        if route == ROUTE_ABSTAIN:
            final_result = _synthetic_result(text='{"node goals": [], "edge goals": []}')
            final_validation = validate_prompt_output(prompt=original_prompt, output_text=final_result.text)
            break

        allowed_attempts = route_refine_budget(route, max_refine_rounds)
        current_attempts = int(route_attempt_counts.get(route, 0))
        if allowed_attempts <= current_attempts:
            break

        rounds_used += 1
        route_attempt_counts[route] = current_attempts + 1

        if route == ROUTE_COMMON_REPAIR:
            repair_result, params_used, _, repair_retry, repair_calls = _attempt_common_repair(
                client=client,
                constraints=constraints,
                base_messages=base_messages,
                original_prompt=original_prompt,
                current_validation=final_validation,
                diagnosis=diagnosis,
                max_retries=max_retries,
                round_idx=rounds_used,
            )
        elif route == ROUTE_STRUCTURAL_REPAIR:
            repair_result, params_used, _, repair_retry, repair_calls = _attempt_structural_relation_repair(
                client=client,
                constraints=constraints,
                current_result=final_result,
                current_validation=final_validation,
                max_retries=max_retries,
                round_idx=rounds_used,
            )
        else:
            break

        retry_log.extend(repair_retry)
        llm_call_count += repair_calls
        if repair_result.error is not None:
            final_result = repair_result
            if params_used:
                final_params = params_used
            break

        if params_used:
            final_params = params_used
        final_result = repair_result
        final_validation = validate_prompt_output(
            prompt=original_prompt,
            output_text=final_result.text,
        )

    final_has_hall = bool(final_validation.get("has_hallucination"))
    residual_diagnosis = {}
    if final_has_hall and final_result.error is None:
        residual_diagnosis = diagnose_failure(
            validation_result=final_validation,
            prompt_constraints=constraints,
            output_text=final_result.text,
            sample_meta=sample_meta,
        )

    meta_pipeline = {
        "pipeline_mode": PIPELINE_MODE,
        "pipeline_route": primary_route,
        "pipeline_route_history": route_history,
        "diagnosed_categories": first_diagnosis.get("diagnosed_categories") or [],
        "repairability": first_diagnosis.get("repairability"),
        "severity": first_diagnosis.get("severity"),
        "should_abstain": bool(first_diagnosis.get("should_abstain")),
        "feasibility_assessment": first_diagnosis.get("feasibility_assessment") or {},
        "substitution_risk_objects": first_diagnosis.get("substitution_risk_objects") or [],
        "missing_required_objects": first_diagnosis.get("missing_required_objects") or [],
        "pipeline_refine_rounds_used": rounds_used,
        "pipeline_refine_rounds_max": max_refine_rounds,
        "pipeline_refine_rounds_global_max": max_refine_rounds,
        "initial_has_hallucination": initial_has_hall,
        "initial_metrics": initial_validation.get("metrics") or {},
        "pipeline_final_has_hallucination": final_has_hall,
        "pipeline_final_metrics": final_validation.get("metrics") or {},
        "pipeline_final_hallucination_details": final_validation.get("hallucination_details") or [],
        "residual_diagnosed_categories": residual_diagnosis.get("diagnosed_categories") or [],
        "route_success": bool(initial_has_hall and not final_has_hall),
        "llm_call_count": llm_call_count,
        "route_attempt_counts": route_attempt_counts,
    }

    return PipelineOutcome(
        result=final_result,
        validation=final_validation,
        params=final_params,
        retry_errors=retry_log,
        llm_call_count=llm_call_count,
        meta_pipeline=meta_pipeline,
    )


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


def _worker_pipeline_dataset_chunk(
    llm_config: LLMConfig,
    run_id: str,
    dataset_name: str,
    chunk_index: int,
    samples: List[HEALSample],
    temp_file: str,
    max_retries: int,
    max_refine_rounds: int,
    progress_queue: Any,
) -> Dict[str, Any]:
    client = LLMClient(llm_config)
    out_path = Path(temp_file)
    processed = 0
    failed = 0

    try:
        for sample in samples:
            outcome = _pipeline_infer_one_sample(
                client=client,
                sample=sample,
                max_retries=max_retries,
                max_refine_rounds=max_refine_rounds,
            )
            record = _build_pipeline_record(
                run_id=run_id,
                dataset_name=dataset_name,
                sample=sample,
                llm_config=llm_config,
                params=outcome.params,
                outcome=outcome,
            )
            append_jsonl(out_path, record)
            processed += 1
            if outcome.result.error is not None:
                failed += 1
            progress_queue.put(
                {
                    "dataset_name": dataset_name,
                    "task_id": sample.task_id,
                    "success": outcome.result.error is None,
                    "retry_count": len(outcome.retry_errors),
                }
            )
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "dataset_name": dataset_name,
            "chunk_index": chunk_index,
            "temp_file": str(out_path),
            "processed": processed,
            "failed": failed,
            "error": "{0}: {1}".format(type(exc).__name__, str(exc)),
        }

    return {
        "dataset_name": dataset_name,
        "chunk_index": chunk_index,
        "temp_file": str(out_path),
        "processed": processed,
        "failed": failed,
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


def _run_pipeline_all_datasets_impl(
    llm_config: LLMConfig,
    heal_root: Path,
    out_dir: Path,
    max_retries: int,
    max_refine_rounds: int,
    max_workers: Optional[int] = None,
) -> Tuple[int, Path]:
    ensure_dir(out_dir)
    start_time = datetime.now()
    run_id = "taxonomyPipeline_{0}".format(make_run_id(start_time))
    run_dir = output_run_dir(out_dir=out_dir, run_id=run_id, llm_name=llm_config.name)
    ensure_dir(run_dir)

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
    chunk_outputs: Dict[str, List[Tuple[int, Path]]] = {}
    expected_chunks: Dict[str, int] = {}

    if active_datasets:
        chunk_tasks: List[Tuple[str, int, List[HEALSample], Path]] = []
        for dataset in active_datasets:
            final_file = output_file_path(run_dir, dataset.name, llm_config.name)
            temp_dir = run_dir / ".tmp" / final_file.stem
            total = len(dataset.samples)
            chunk_count = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
            expected_chunks[dataset.name] = chunk_count
            for chunk_index in range(chunk_count):
                start = chunk_index * CHUNK_SIZE
                end = start + CHUNK_SIZE
                chunk_samples = dataset.samples[start:end]
                part_file = temp_dir / "part_{0:05d}.jsonl".format(chunk_index)
                chunk_tasks.append((dataset.name, chunk_index, chunk_samples, part_file))

        workers = _resolve_max_workers(max_workers, len(chunk_tasks))
        with mp.Manager() as manager:
            progress_queue = manager.Queue()
            with cf.ProcessPoolExecutor(max_workers=workers) as executor:
                futures: Dict[cf.Future, Tuple[str, int, Path]] = {}
                for dataset_name, chunk_index, chunk_samples, part_file in chunk_tasks:
                    future = executor.submit(
                        _worker_pipeline_dataset_chunk,
                        llm_config,
                        run_id,
                        dataset_name,
                        chunk_index,
                        chunk_samples,
                        str(part_file),
                        max_retries,
                        max_refine_rounds,
                        progress_queue,
                    )
                    futures[future] = (dataset_name, chunk_index, part_file)

                while futures:
                    _drain_progress_queue(progress, progress_queue)
                    done_futures = [future for future in futures.keys() if future.done()]
                    for future in done_futures:
                        dataset_name, chunk_index, part_file = futures.pop(future)
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
                            chunk_outputs.setdefault(dataset_name, []).append(
                                (chunk_index, Path(str(result.get("temp_file") or part_file)))
                            )
                    time.sleep(0.05)

                _drain_progress_queue(progress, progress_queue)

        for dataset in active_datasets:
            dataset_name = dataset.name
            final_file = output_file_path(run_dir, dataset_name, llm_config.name)
            sorted_parts = sorted(chunk_outputs.get(dataset_name, []), key=lambda item: item[0])
            expected = expected_chunks.get(dataset_name, 0)

            if len(sorted_parts) != expected:
                worker_failures.append(
                    {
                        "dataset_name": dataset_name,
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

            temp_dir = run_dir / ".tmp" / final_file.stem
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

        temp_root = run_dir / ".tmp"
        if temp_root.exists() and temp_root.is_dir():
            shutil.rmtree(temp_root, ignore_errors=True)

    progress.finish()

    if load_errors or worker_failures:
        error_file = output_file_path(run_dir, "dataset_load_errors", llm_config.name)
        for err in load_errors:
            append_jsonl(
                error_file,
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
                    "meta": {"pipeline": {"pipeline_mode": PIPELINE_MODE}},
                },
            )
        for worker_err in worker_failures:
            append_jsonl(
                error_file,
                {
                    "run_id": run_id,
                    "timestamp": now_iso(),
                    "dataset": worker_err.get("dataset_name"),
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
                        "type": "WorkerRuntimeError",
                        "message": worker_err.get("error"),
                        "traceback": None,
                    },
                    "source_path": None,
                    "meta": {"pipeline": {"pipeline_mode": PIPELINE_MODE}},
                },
            )
    return 0, run_dir


def run_pipeline_all_datasets(
    llm_config: LLMConfig,
    heal_root: Path,
    out_dir: Path,
    max_retries: int,
    max_refine_rounds: int,
    max_workers: Optional[int] = None,
) -> int:
    exit_code, _ = _run_pipeline_all_datasets_impl(
        llm_config=llm_config,
        heal_root=heal_root,
        out_dir=out_dir,
        max_retries=max_retries,
        max_refine_rounds=max_refine_rounds,
        max_workers=max_workers,
    )
    return exit_code


def run_pipeline_all_datasets_with_run_dir(
    llm_config: LLMConfig,
    heal_root: Path,
    out_dir: Path,
    max_retries: int,
    max_refine_rounds: int,
    max_workers: Optional[int] = None,
) -> Tuple[int, Path]:
    return _run_pipeline_all_datasets_impl(
        llm_config=llm_config,
        heal_root=heal_root,
        out_dir=out_dir,
        max_retries=max_retries,
        max_refine_rounds=max_refine_rounds,
        max_workers=max_workers,
    )


def run_pipeline_single_behavior_baseline(
    llm_config: LLMConfig,
    heal_root: Path,
    max_retries: int,
    max_refine_rounds: int,
) -> int:
    sample = load_behavior_baseline_first(heal_root)
    if sample is None:
        print("No sample found at HEAL/behavior/baseline.csv")
        return 1

    client = LLMClient(llm_config)
    outcome = _pipeline_infer_one_sample(
        client=client,
        sample=sample,
        max_retries=max_retries,
        max_refine_rounds=max_refine_rounds,
    )
    if outcome.result.error is not None:
        print("Request failed: {0}: {1}".format(outcome.result.error.type, outcome.result.error.message))
        return 1

    print(outcome.result.text)
    print(
        "\n[taxonomy_pipeline] route={0}, rounds_used={1}, has_hallucination={2}".format(
            outcome.meta_pipeline.get("pipeline_route"),
            outcome.meta_pipeline.get("pipeline_refine_rounds_used"),
            outcome.validation.get("has_hallucination"),
        )
    )
    return 0
