import json
from typing import Any, Dict, List


def _build_forbidden_states(validation_result: Dict[str, Any]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for item in validation_result.get("hallucination_details") or []:
        if str(item.get("type", "")).strip().upper() != "S":
            continue
        parsed = item.get("parsed_goal") or {}
        obj = str(parsed.get("object_name") or "").strip()
        state = str(parsed.get("state_name") or "").strip()
        if not obj or not state:
            continue
        if obj not in mapping:
            mapping[obj] = []
        if state not in mapping[obj]:
            mapping[obj].append(state)
    return mapping


def build_common_repair_prompt(
    original_prompt: str,
    constraints: Any,
    validation_result: Dict[str, Any],
    diagnosis: Dict[str, Any],
) -> str:
    violations = validation_result.get("hallucination_details") or []
    forbidden_objects = diagnosis.get("missing_required_objects") or []
    forbidden_states = _build_forbidden_states(validation_result)

    strict_rules: Dict[str, Any] = {
        "allowed_objects": sorted(list(constraints.objects)),
        "allowed_states_by_object": {
            key: sorted(list(value)) for key, value in sorted(constraints.states_by_object.items())
        },
        "allowed_relations": sorted(list(constraints.relations)),
        "do_not_substitute_missing_objects": True,
        "forbidden_objects": forbidden_objects,
        "forbidden_states": forbidden_states,
    }
    if constraints.to_name_by_relation:
        strict_rules["allowed_to_name_by_relation"] = {
            key: sorted(list(value))
            for key, value in sorted(constraints.to_name_by_relation.items())
        }

    guidance = (
        "\n\n[Taxonomy Pipeline | Common Repair]\n"
        "Your previous answer violated constraints.\n"
        "Do not repeat these violations:\n"
        "{violations}\n\n"
        "Diagnosis summary:\n"
        "{diagnosis}\n\n"
        "Strict constraints JSON:\n"
        "{strict_rules}\n\n"
        "If the task is infeasible under current scene constraints, return empty goals:\n"
        "{{\"node goals\": [], \"edge goals\": []}}\n\n"
        "Now regenerate final answer in JSON only with exactly keys 'node goals' and 'edge goals'."
    ).format(
        violations=json.dumps(violations, ensure_ascii=False, indent=2),
        diagnosis=json.dumps(diagnosis, ensure_ascii=False, indent=2),
        strict_rules=json.dumps(strict_rules, ensure_ascii=False, indent=2),
    )
    return original_prompt + guidance
