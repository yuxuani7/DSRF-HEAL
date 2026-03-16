import concurrent.futures as cf
import multiprocessing as mp
import os
import queue
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.heal_loader import load_behavior_baseline_first, load_heal_datasets
from src.io_utils import (
    append_jsonl,
    ensure_dir,
    make_run_id,
    now_iso,
    output_file_path,
    output_run_dir,
)
from src.llm_client import LLMClient
from src.progress import MultiDatasetProgress
from src.types import HEALSample, LLMCallError, LLMCallResult, LLMConfig

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


DATASET_CHUNK_SIZE = _resolve_chunk_size(100)
RETRY_CYCLE_SLEEP_SEC = 15
MAX_RETRY_ERROR_LOG = 200


def _error_to_dict(error: Optional[LLMCallError]) -> Optional[Dict[str, Any]]:
    if error is None:
        return None
    return {
        "type": error.type,
        "message": error.message,
        "traceback": error.traceback,
    }


def _build_record(
    run_id: str,
    dataset_name: str,
    sample: HEALSample,
    llm_config: LLMConfig,
    params: Dict[str, Any],
    result: LLMCallResult,
    retry_count: int,
    retry_errors: List[Dict[str, Any]],
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
            "text": result.text,
            "finish_reason": result.finish_reason,
        },
        "metrics": {
            "latency_ms": result.latency_ms,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "raw_chunk_count": result.raw_chunk_count,
            "retry_count": retry_count,
        },
        "error": _error_to_dict(result.error),
        "retry_errors": retry_errors,
        "meta": sample.meta,
    }


def _call_with_retry(
    client: LLMClient,
    sample: HEALSample,
    max_retries: int,
) -> Tuple[LLMCallResult, int, List[Dict[str, Any]], Dict[str, Any]]:
    retry_errors: List[Dict[str, Any]] = []
    params_used: Dict[str, Any] = {}
    total_attempts = 0

    while True:
        for attempt in range(max_retries + 1):
            total_attempts += 1
            params_used = client.build_params()
            result = client.chat(messages=sample.messages)
            if result.error is None:
                return result, max(total_attempts - 1, 0), retry_errors, params_used

            if len(retry_errors) < MAX_RETRY_ERROR_LOG:
                retry_errors.append(
                    {
                        "attempt": total_attempts,
                        "type": result.error.type,
                        "message": result.error.message,
                    }
                )

            if not _is_retryable_error(result.error.type, result.error.message):
                return result, max(total_attempts - 1, 0), retry_errors, params_used

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


def _resolve_max_workers(max_workers: Optional[int], dataset_count: int) -> int:
    if dataset_count <= 0:
        return 1
    if max_workers is None or max_workers <= 0:
        return dataset_count
    return max(1, min(max_workers, dataset_count))


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


def _worker_run_dataset_chunk(
    llm_config: LLMConfig,
    run_id: str,
    dataset_name: str,
    chunk_index: int,
    samples: List[HEALSample],
    temp_file: str,
    max_retries: int,
    progress_queue: Any,
) -> Dict[str, Any]:
    client = LLMClient(llm_config)
    out_path = Path(temp_file)
    processed = 0
    failed = 0

    try:
        for sample in samples:
            result, retry_count, retry_errors, params_used = _call_with_retry(
                client=client,
                sample=sample,
                max_retries=max_retries,
            )
            record = _build_record(
                run_id=run_id,
                dataset_name=dataset_name,
                sample=sample,
                llm_config=llm_config,
                params=params_used,
                result=result,
                retry_count=retry_count,
                retry_errors=retry_errors,
            )
            append_jsonl(out_path, record)
            processed += 1
            if result.error is not None:
                failed += 1
            progress_queue.put(
                {
                    "dataset_name": dataset_name,
                    "task_id": sample.task_id,
                    "success": result.error is None,
                    "retry_count": retry_count,
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


def _merge_chunk_files(
    final_file: Path,
    chunk_files: List[Path],
) -> None:
    ensure_dir(final_file.parent)
    if final_file.exists():
        final_file.unlink()
    with final_file.open("w", encoding="utf-8") as out_f:
        for part_file in chunk_files:
            with part_file.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)


def run_single_behavior_baseline(
    llm_config: LLMConfig,
    heal_root: Path,
    max_retries: int,
) -> int:
    sample = load_behavior_baseline_first(heal_root)
    if sample is None:
        print("No sample found at HEAL/behavior/baseline.csv")
        return 1

    client = LLMClient(llm_config)
    result, _, _, _ = _call_with_retry(client=client, sample=sample, max_retries=max_retries)
    if result.error is not None:
        print("Request failed: {0}: {1}".format(result.error.type, result.error.message))
        return 1

    print(result.text)
    return 0


def _run_all_datasets_impl(
    llm_config: LLMConfig,
    heal_root: Path,
    out_dir: Path,
    max_retries: int,
    max_workers: Optional[int] = None,
) -> Tuple[int, Path]:
    ensure_dir(out_dir)
    start_time = datetime.now()
    run_id = make_run_id(start_time)
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
            chunk_count = (total + DATASET_CHUNK_SIZE - 1) // DATASET_CHUNK_SIZE
            expected_chunks[dataset.name] = chunk_count
            for chunk_index in range(chunk_count):
                start = chunk_index * DATASET_CHUNK_SIZE
                end = start + DATASET_CHUNK_SIZE
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
                        _worker_run_dataset_chunk,
                        llm_config,
                        run_id,
                        dataset_name,
                        chunk_index,
                        chunk_samples,
                        str(part_file),
                        max_retries,
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
                            chunk_outputs.setdefault(dataset_name, []).append((chunk_index, Path(str(result.get("temp_file") or part_file))))
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
                },
            )
    return 0, run_dir


def run_all_datasets(
    llm_config: LLMConfig,
    heal_root: Path,
    out_dir: Path,
    max_retries: int,
    max_workers: Optional[int] = None,
) -> int:
    exit_code, _ = _run_all_datasets_impl(
        llm_config=llm_config,
        heal_root=heal_root,
        out_dir=out_dir,
        max_retries=max_retries,
        max_workers=max_workers,
    )
    return exit_code


def run_all_datasets_with_run_dir(
    llm_config: LLMConfig,
    heal_root: Path,
    out_dir: Path,
    max_retries: int,
    max_workers: Optional[int] = None,
) -> Tuple[int, Path]:
    return _run_all_datasets_impl(
        llm_config=llm_config,
        heal_root=heal_root,
        out_dir=out_dir,
        max_retries=max_retries,
        max_workers=max_workers,
    )
