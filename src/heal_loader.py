import csv
import json
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.types import HEALDataset, HEALSample, LoadError, Message


SUPPORTED_SUFFIXES = {".csv", ".jsonl", ".json", ".txt"}
PROMPT_KEYS = (
    "modified_prompts",
    "baseline_prompts",
    "prompt",
    "input",
    "question",
    "query",
    "instruction",
    "text",
)
TASK_ID_KEYS = ("task_id", "taskid", "id", "uid")
MESSAGES_KEYS = ("messages", "chat_messages")


def _dataset_name_from_path(heal_root: Path, file_path: Path) -> str:
    rel = file_path.relative_to(heal_root).with_suffix("")
    return rel.as_posix().replace("/", "_")


def _default_task_id(dataset_name: str, index: int) -> str:
    return "{0}_{1:06d}".format(dataset_name, index)


def _pick_first_string(record: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_messages(record: Dict[str, Any], prompt: str) -> List[Message]:
    for key in MESSAGES_KEYS:
        raw_messages = record.get(key)
        if raw_messages is None:
            continue
        if isinstance(raw_messages, list):
            return _normalize_messages(raw_messages)
        if isinstance(raw_messages, str):
            try:
                parsed = json.loads(raw_messages)
                if isinstance(parsed, list):
                    return _normalize_messages(parsed)
            except json.JSONDecodeError:
                continue
    return [{"role": "user", "content": prompt}]


def _normalize_messages(messages: List[Any]) -> List[Message]:
    normalized: List[Message] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user"))
        content = item.get("content", "")
        if content is None:
            content = ""
        normalized.append({"role": role, "content": str(content)})
    if not normalized:
        normalized.append({"role": "user", "content": ""})
    return normalized


def _to_sample(dataset_name: str, index: int, record: Dict[str, Any]) -> HEALSample:
    prompt = _pick_first_string(record, PROMPT_KEYS)
    if not prompt:
        prompt = json.dumps(record, ensure_ascii=False)

    task_id = _pick_first_string(record, TASK_ID_KEYS)
    if not task_id:
        task_id = _default_task_id(dataset_name, index)

    messages = _extract_messages(record, prompt)

    return HEALSample(
        dataset_name=dataset_name,
        task_id=task_id,
        prompt=prompt,
        messages=messages,
        meta=record,
    )


def _load_csv(file_path: Path, dataset_name: str) -> List[HEALSample]:
    samples: List[HEALSample] = []
    with file_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            record = dict(row or {})
            samples.append(_to_sample(dataset_name, idx, record))
    return samples


def _load_jsonl(file_path: Path, dataset_name: str) -> List[HEALSample]:
    samples: List[HEALSample] = []
    with file_path.open("r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            data = json.loads(line)
            record = data if isinstance(data, dict) else {"raw": data}
            samples.append(_to_sample(dataset_name, idx, record))
    return samples


def _load_json(file_path: Path, dataset_name: str) -> List[HEALSample]:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[HEALSample] = []
    if isinstance(data, list):
        for idx, item in enumerate(data):
            record = item if isinstance(item, dict) else {"raw": item}
            samples.append(_to_sample(dataset_name, idx, record))
        return samples
    if isinstance(data, dict):
        # Common pattern: {"data": [...]}.
        if isinstance(data.get("data"), list):
            for idx, item in enumerate(data["data"]):
                record = item if isinstance(item, dict) else {"raw": item}
                samples.append(_to_sample(dataset_name, idx, record))
            return samples
        samples.append(_to_sample(dataset_name, 0, data))
        return samples
    samples.append(_to_sample(dataset_name, 0, {"raw": data}))
    return samples


def _load_txt(file_path: Path, dataset_name: str) -> List[HEALSample]:
    samples: List[HEALSample] = []
    with file_path.open("r", encoding="utf-8") as f:
        for idx, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            record = {"prompt": line}
            samples.append(_to_sample(dataset_name, idx, record))
    return samples


def _iter_dataset_files(heal_root: Path) -> List[Path]:
    files: List[Path] = []
    for path in heal_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(heal_root).parts):
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return sorted(files)


def load_heal_datasets(heal_root: Path) -> Tuple[List[HEALDataset], List[LoadError]]:
    if not heal_root.exists():
        raise FileNotFoundError("HEAL root not found: {0}".format(heal_root))

    datasets: List[HEALDataset] = []
    errors: List[LoadError] = []

    for file_path in _iter_dataset_files(heal_root):
        dataset_name = _dataset_name_from_path(heal_root, file_path)
        try:
            suffix = file_path.suffix.lower()
            if suffix == ".csv":
                samples = _load_csv(file_path, dataset_name)
            elif suffix == ".jsonl":
                samples = _load_jsonl(file_path, dataset_name)
            elif suffix == ".json":
                samples = _load_json(file_path, dataset_name)
            elif suffix == ".txt":
                samples = _load_txt(file_path, dataset_name)
            else:
                continue
            datasets.append(
                HEALDataset(name=dataset_name, source_path=file_path, samples=samples)
            )
        except Exception as exc:  # pylint: disable=broad-except
            errors.append(
                LoadError(
                    dataset_name=dataset_name,
                    source_path=str(file_path),
                    error_type=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                )
            )
    return datasets, errors


def load_behavior_baseline_first(heal_root: Path) -> Optional[HEALSample]:
    baseline_path = heal_root / "behavior" / "baseline.csv"
    if not baseline_path.exists():
        return None

    dataset_name = _dataset_name_from_path(heal_root, baseline_path)
    samples = _load_csv(baseline_path, dataset_name)
    if not samples:
        return None
    return samples[0]

