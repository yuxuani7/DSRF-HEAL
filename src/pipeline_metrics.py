import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from src.io_utils import ensure_dir, now_iso


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _avg(total: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return total / float(count)


def _load_jsonl_records(folder: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for file_path in sorted(folder.glob("*.jsonl")):
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = line.strip()
                if not row:
                    continue
                try:
                    obj = json.loads(row)
                except Exception:
                    continue
                records.append(obj)
    return records


def analyze_taxonomy_pipeline_run(
    output_run_dir: Path,
    validation_run_dir: Path,
    results_dir: Path,
) -> Path:
    output_records = _load_jsonl_records(output_run_dir)
    validation_records = _load_jsonl_records(validation_run_dir)

    route_count = Counter()
    route_success = Counter()
    route_attempt = Counter()
    route_attempt_usage = Counter()
    route_token_total = defaultdict(float)
    route_token_prompt = defaultdict(float)
    route_token_completion = defaultdict(float)
    route_token_samples = Counter()
    category_count = Counter()
    repairability_count = Counter()
    residual_diag_count = Counter()
    total_samples = 0
    initial_hall = 0
    final_hall = 0
    improved = 0

    for rec in output_records:
        if rec.get("task_id") is None:
            continue
        pipe = (rec.get("meta") or {}).get("pipeline") or {}
        route = str(pipe.get("pipeline_route") or "unknown")
        route_count[route] += 1
        route_attempt[route] += 1
        total_samples += 1
        route_history = pipe.get("pipeline_route_history") or []
        if isinstance(route_history, list):
            for route_item in route_history:
                route_attempt_usage[str(route_item)] += 1

        initial_has = bool(pipe.get("initial_has_hallucination"))
        final_has = bool(pipe.get("pipeline_final_has_hallucination"))
        if initial_has:
            initial_hall += 1
        if final_has:
            final_hall += 1
        if initial_has and not final_has:
            improved += 1

        if not final_has:
            route_success[route] += 1

        categories = pipe.get("diagnosed_categories") or []
        for cat in categories:
            category_count[str(cat)] += 1
        rep = str(pipe.get("repairability") or "unknown")
        repairability_count[rep] += 1
        if final_has:
            for cat in categories:
                residual_diag_count[str(cat)] += 1

        metrics = rec.get("metrics") or {}
        total_tokens = metrics.get("total_tokens")
        prompt_tokens = metrics.get("prompt_tokens")
        completion_tokens = metrics.get("completion_tokens")
        if total_tokens is not None:
            route_token_total[route] += _safe_float(total_tokens)
        if prompt_tokens is not None:
            route_token_prompt[route] += _safe_float(prompt_tokens)
        if completion_tokens is not None:
            route_token_completion[route] += _safe_float(completion_tokens)
        if total_tokens is not None or prompt_tokens is not None or completion_tokens is not None:
            route_token_samples[route] += 1

    validation_residual_map = {
        "O1": "entity_grounding",
        "O2": "entity_grounding",
        "S": "state_property",
        "R": "structural_relation",
        "O3": "structural_relation_target_mismatch",
    }
    residual_validation_category_count = Counter()
    for rec in validation_records:
        if not bool(rec.get("has_hallucination")):
            continue
        details = rec.get("hallucination_details") or []
        for item in details:
            hall_type = str(item.get("type") or "").strip().upper()
            mapped = validation_residual_map.get(hall_type)
            if mapped:
                residual_validation_category_count[mapped] += 1

    route_usage = {}
    route_success_rate = {}
    route_avg_token_cost = {}
    for route, count in route_count.items():
        route_usage[route] = {
            "count": count,
            "ratio": _avg(float(count), total_samples),
        }
        success = route_success[route]
        route_success_rate[route] = {
            "success_count": success,
            "attempt_count": route_attempt[route],
            "success_rate": _avg(float(success), route_attempt[route]),
        }
        token_samples = route_token_samples[route]
        route_avg_token_cost[route] = {
            "samples_with_token_usage": token_samples,
            "avg_total_tokens": _avg(route_token_total[route], token_samples),
            "avg_prompt_tokens": _avg(route_token_prompt[route], token_samples),
            "avg_completion_tokens": _avg(route_token_completion[route], token_samples),
        }

    summary = {
        "generated_at": now_iso(),
        "run_dir": output_run_dir.name,
        "total_samples": total_samples,
        "before_after_hallucination": {
            "initial_hallucination_samples": initial_hall,
            "final_hallucination_samples": final_hall,
            "initial_hallucination_rate": _avg(float(initial_hall), total_samples),
            "final_hallucination_rate": _avg(float(final_hall), total_samples),
            "improved_samples": improved,
        },
        "route_usage": route_usage,
        "route_attempt_usage": {
            key: {
                "count": count,
                "ratio_per_sample": _avg(float(count), total_samples),
            }
            for key, count in sorted(route_attempt_usage.items())
        },
        "route_success": route_success_rate,
        "route_avg_token_cost": route_avg_token_cost,
        "diagnosed_category_distribution": {
            key: {
                "count": count,
                "ratio": _avg(float(count), total_samples),
            }
            for key, count in sorted(category_count.items())
        },
        "repairability_distribution": {
            key: {
                "count": count,
                "ratio": _avg(float(count), total_samples),
            }
            for key, count in sorted(repairability_count.items())
        },
        "residual_category_distribution_from_diagnosis": dict(sorted(residual_diag_count.items())),
        "residual_category_distribution_from_validation": dict(
            sorted(residual_validation_category_count.items())
        ),
    }

    ensure_dir(results_dir)
    output_path = results_dir / "{0}__taxonomy_metrics.json".format(output_run_dir.name)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
