import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_run_id(start_time: datetime) -> str:
    return start_time.strftime("%Y%m%d_%H%M%S")


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def slugify(name: str) -> str:
    normalized = name.replace("/", "_").replace("\\", "_").strip().lower()
    normalized = re.sub(r"[^a-z0-9_.-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_") or "dataset"


def output_run_dir(out_dir: Path, run_id: str, llm_name: str) -> Path:
    safe_llm = slugify(llm_name)
    return out_dir / "{run_id}__{llm}".format(run_id=run_id, llm=safe_llm)


def output_file_path(run_dir: Path, dataset_name: str, llm_name: str) -> Path:
    safe_dataset = slugify(dataset_name)
    safe_llm = slugify(llm_name)
    filename = "{dataset}__{llm}.jsonl".format(
        dataset=safe_dataset,
        llm=safe_llm,
    )
    return run_dir / filename


def _to_json_safe(value: Any) -> Any:
    if value is Ellipsis:
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_to_json_safe(item) for item in sorted(value, key=lambda item: str(item))]
    if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
        try:
            return _to_json_safe(value.model_dump())
        except Exception:
            return str(value)
    if hasattr(value, "__dict__"):
        try:
            return _to_json_safe(vars(value))
        except Exception:
            return str(value)
    return str(value)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        safe_record = _to_json_safe(record)
        f.write(json.dumps(safe_record, ensure_ascii=False) + "\n")


TIMESTAMP_RUN_ID_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def resolve_run_dir(project_root: Path, raw_value: str, default_root: str) -> Path:
    token = str(raw_value or "").strip()
    if not token:
        raise ValueError("Run dir argument is empty.")

    input_path = Path(token).expanduser()
    candidates: List[Path] = []
    if input_path.is_absolute():
        candidates.append(input_path)
    else:
        candidates.append(project_root / input_path)
        candidates.append(project_root / default_root / input_path)

    for candidate in candidates:
        if candidate.exists():
            if not candidate.is_dir():
                raise ValueError("Path is not a directory: {0}".format(candidate))
            return candidate.resolve()

    search_root = (project_root / default_root).resolve()
    if TIMESTAMP_RUN_ID_PATTERN.match(token):
        if not search_root.exists() or not search_root.is_dir():
            raise FileNotFoundError(
                "Search root not found: {0}".format(search_root)
            )
        pattern = re.compile(r"(?:^|_){0}(?:__|$)".format(re.escape(token)))
        matches = sorted(
            [
                child
                for child in search_root.iterdir()
                if child.is_dir() and pattern.search(child.name)
            ],
            key=lambda path: path.name,
        )
        if len(matches) == 1:
            return matches[0].resolve()
        if len(matches) > 1:
            raise ValueError(
                "Timestamp {0} matched multiple run dirs in {1}: {2}".format(
                    token,
                    search_root,
                    ", ".join(path.name for path in matches),
                )
            )
        raise FileNotFoundError(
            "No run dir matched timestamp {0} under {1}".format(token, search_root)
        )

    raise FileNotFoundError(
        "Run dir not found: {0}. Tried: {1}".format(
            token,
            ", ".join(str(path.resolve()) for path in candidates),
        )
    )
