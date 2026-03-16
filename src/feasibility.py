import re
from typing import Any, Dict, List


def _canonical(value: Any) -> str:
    return str(value or "").strip().casefold()


def _base_object_name(value: str) -> str:
    text = _canonical(value)
    if not text:
        return ""
    head = text.split(".", 1)[0]
    head = re.sub(r"_[0-9]+$", "", head)
    return head


def _task_tokens(task_description: str) -> List[str]:
    return [token.casefold() for token in re.findall(r"[A-Za-z_]+", task_description or "")]


def assess_feasibility(
    task_description: str,
    constraints: Any,
    validation_result: Dict[str, Any],
    diagnosis: Dict[str, Any],
    llm_client: Any = None,
    llm_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    del llm_client, llm_config, constraints  # rule-first implementation

    missing_required_objects = list(diagnosis.get("missing_required_objects") or [])
    substitution_risk_objects = list(diagnosis.get("substitution_risk_objects") or [])

    tokens = set(_task_tokens(task_description))
    core_missing = []
    for obj_name in missing_required_objects:
        base_name = _base_object_name(obj_name)
        if base_name and base_name in tokens:
            core_missing.append(obj_name)

    metrics = validation_result.get("metrics") or {}
    o1 = int(((metrics.get("o1") or {}).get("count") or 0))
    o2 = int(((metrics.get("o2") or {}).get("count") or 0))
    node_den = int(((metrics.get("o1") or {}).get("denominator") or 0))
    edge_den = int(((metrics.get("o2") or {}).get("denominator") or 0))
    total_den = max(node_den + edge_den, 1)
    entity_ratio = float(o1 + o2) / float(total_den)

    should_abstain = False
    reason = ""
    if len(core_missing) >= 2 and entity_ratio >= 0.6:
        should_abstain = True
        reason = "Multiple key task objects appear missing from scene and entity errors dominate."
    elif len(core_missing) >= 1 and missing_required_objects and substitution_risk_objects and entity_ratio >= 0.85:
        should_abstain = True
        reason = "Core task object appears missing and the current answer is dominated by unsafe substitutions."

    return {
        "is_task_feasible": not should_abstain,
        "missing_required_objects": core_missing or missing_required_objects,
        "should_abstain": should_abstain,
        "reason": reason,
    }
