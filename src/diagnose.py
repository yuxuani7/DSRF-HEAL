import re
from typing import Any, Dict, List, Set

from src.feasibility import assess_feasibility
from src.pipeline_routes import select_repair_route


def extract_task_description(sample_meta: Dict[str, Any]) -> str:
    meta = sample_meta or {}
    for key in (
        "original_natural_language_description",
        "distractor_injected_task",
        "task_description",
        "goal_description",
        "instruction",
    ):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _canonical(value: Any) -> str:
    return str(value or "").strip().casefold()


def _base_object_name(value: str) -> str:
    text = _canonical(value)
    if not text:
        return ""
    base = text.split(".", 1)[0]
    base = re.sub(r"_[0-9]+$", "", base)
    return base


def _collect_missing_objects(validation_result: Dict[str, Any], constraints: Any) -> List[str]:
    missing: Set[str] = set()
    details = validation_result.get("hallucination_details") or []
    scene_objects = set(constraints.objects or [])

    for item in details:
        hall_type = str(item.get("type", "")).strip().upper()
        parsed = item.get("parsed_goal") or {}
        if hall_type == "O1":
            obj = _canonical(parsed.get("object_name"))
            if obj and obj not in scene_objects:
                missing.add(obj)
        elif hall_type == "O2":
            from_name = _canonical(parsed.get("from_name"))
            to_name = _canonical(parsed.get("to_name"))
            if from_name and from_name not in scene_objects:
                missing.add(from_name)
            if to_name and to_name not in scene_objects:
                missing.add(to_name)
    return sorted(list(missing))


def _collect_substitution_risk_objects(validation_result: Dict[str, Any]) -> List[str]:
    risk: Set[str] = set()
    details = validation_result.get("hallucination_details") or []
    for item in details:
        hall_type = str(item.get("type", "")).strip().upper()
        if hall_type != "O3":
            continue
        parsed = item.get("parsed_goal") or {}
        to_name = _canonical(parsed.get("to_name"))
        if to_name:
            risk.add(to_name)
    return sorted(list(risk))


def diagnose_failure(
    validation_result: Dict[str, Any],
    prompt_constraints: Any,
    output_text: str,
    sample_meta: Dict[str, Any],
) -> Dict[str, Any]:
    del output_text

    metrics = validation_result.get("metrics") or {}
    o1 = int(((metrics.get("o1") or {}).get("count") or 0))
    s = int(((metrics.get("s") or {}).get("count") or 0))
    o2 = int(((metrics.get("o2") or {}).get("count") or 0))
    r = int(((metrics.get("r") or {}).get("count") or 0))
    o3 = int(((metrics.get("o3") or {}).get("count") or 0))

    categories: List[str] = []
    if (o1 + o2) > 0:
        categories.append("entity_grounding")
    if s > 0:
        categories.append("state_property")
    if r > 0:
        categories.append("structural_relation")
    if o3 > 0:
        categories.append("structural_relation_target_mismatch")

    missing_required_objects = _collect_missing_objects(validation_result, prompt_constraints)
    substitution_risk_objects = _collect_substitution_risk_objects(validation_result)
    structural_relation_risk = (r + o3) > 0

    severity = "low"
    total_count = o1 + s + o2 + r + o3
    if structural_relation_risk or total_count >= 3:
        severity = "medium"
    if (o1 + o2) >= 2 and structural_relation_risk:
        severity = "high"

    task_description = extract_task_description(sample_meta)
    base = {
        "diagnosed_categories": categories,
        "repairability": "repairable",
        "route": "noop",
        "severity": severity,
        "should_abstain": False,
        "missing_required_objects": missing_required_objects,
        "structural_relation_risk": structural_relation_risk,
        "substitution_risk_objects": substitution_risk_objects,
        "repair_hints": [],
        "task_description": task_description,
        "dataset_name": str((sample_meta or {}).get("dataset_name") or ""),
    }

    feasibility = assess_feasibility(
        task_description=task_description,
        constraints=prompt_constraints,
        validation_result=validation_result,
        diagnosis=base,
    )
    if feasibility.get("should_abstain"):
        base["diagnosed_categories"] = list(base["diagnosed_categories"]) + ["feasibility"]
        base["repairability"] = "abstention_required"
        base["should_abstain"] = True
        base["severity"] = "high"
        base["repair_hints"] = [
            "Do not substitute missing key objects.",
            "Return empty goals when task is infeasible.",
        ]
    elif "structural_relation_target_mismatch" in base["diagnosed_categories"]:
        base["repairability"] = "structure_repairable"
        base["repair_hints"] = [
            "Repair edge relation-target pairs with strict candidate constraints."
        ]
    elif base["diagnosed_categories"]:
        base["repairability"] = "constraint_repairable"
        base["repair_hints"] = [
            "Use strict object/state/relation constraints.",
            "Avoid introducing unseen objects.",
        ]

    base["route"] = select_repair_route(base)
    base["feasibility_assessment"] = feasibility
    return base
