import json
from typing import Any, Dict, List, Optional

from src.hallucination_validator import parse_output_payload


def _canonical(value: Any) -> str:
    return str(value or "").strip().casefold()


def _parse_json_dict(text: str) -> Dict[str, Any]:
    clean = (text or "").strip()
    if not clean:
        return {}
    candidates = [clean]
    first = clean.find("{")
    last = clean.rfind("}")
    if first != -1 and last != -1 and last > first:
        candidates.append(clean[first : last + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return {}


def _extract_raw_edge_goals(output_text: str) -> List[Any]:
    output_obj, _ = parse_output_payload(output_text or "")
    for key in ("edge goals", "edge_goals", "Edge goals"):
        raw = output_obj.get(key)
        if isinstance(raw, list):
            return raw
    return []


def extract_node_goals_from_output(output_text: str) -> List[Any]:
    output_obj, _ = parse_output_payload(output_text or "")
    for key in ("node goals", "node_goals", "Node goals"):
        raw = output_obj.get(key)
        if isinstance(raw, list):
            return raw
    return []


def _parse_edge_goal(raw_goal: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_goal, list):
        return None
    negated = False
    target = raw_goal
    if len(raw_goal) == 2 and _canonical(raw_goal[0]) == "not" and isinstance(raw_goal[1], list):
        negated = True
        target = raw_goal[1]
    if len(target) < 3:
        return None
    return {
        "relation": str(target[0] or ""),
        "from_name": str(target[1] or ""),
        "to_name": str(target[2] or ""),
        "negated": negated,
        "raw_goal": raw_goal,
    }


def build_target_choice_prompt(
    output_text: str,
    validation_result: Dict[str, Any],
    constraints: Any,
) -> str:
    raw_edge_goals = _extract_raw_edge_goals(output_text)
    illegal_indexes = set()
    for item in validation_result.get("hallucination_details") or []:
        hall_type = str(item.get("type") or "").strip().upper()
        if hall_type != "O3":
            continue
        try:
            illegal_indexes.add(int(item.get("goal_index")))
        except Exception:
            continue

    repair_items: List[Dict[str, Any]] = []
    for idx in sorted(illegal_indexes):
        if idx < 0 or idx >= len(raw_edge_goals):
            continue
        parsed = _parse_edge_goal(raw_edge_goals[idx])
        if parsed is None:
            continue
        relation_key = _canonical(parsed["relation"])
        allowed_targets = sorted(list(constraints.to_name_by_relation.get(relation_key) or []))
        if not allowed_targets:
            continue
        repair_items.append(
            {
                "goal_index": idx,
                "from_name": parsed["from_name"],
                "relation": parsed["relation"],
                "current_to_name": parsed["to_name"],
                "negated": parsed["negated"],
                "allowed_targets": allowed_targets,
            }
        )

    payload = {
        "repair_items": repair_items,
    }
    return (
        "[Taxonomy Pipeline | Structural Relation Repair]\n"
        "Fix only relation-target mismatch edges.\n"
        "For each item, choose exactly one target index from allowed_targets.\n"
        "If none is suitable, choose -1 to drop that edge.\n"
        "Return JSON only with key 'repairs'.\n"
        "Each repair item must be: "
        "{\"goal_index\": int, \"to_name_choice\": int}\n\n"
        "Input JSON:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def extract_target_repairs(text: str) -> List[Dict[str, int]]:
    obj = _parse_json_dict(text)
    raw = obj.get("repairs")
    if not isinstance(raw, list):
        return []
    repairs: List[Dict[str, int]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            goal_index = int(item.get("goal_index"))
            to_name_choice = int(item.get("to_name_choice"))
        except Exception:
            continue
        repairs.append(
            {
                "goal_index": goal_index,
                "to_name_choice": to_name_choice,
            }
        )
    return repairs


def apply_target_repairs(
    output_text: str,
    validation_result: Dict[str, Any],
    constraints: Any,
    repairs: List[Dict[str, int]],
) -> List[Any]:
    raw_edge_goals = _extract_raw_edge_goals(output_text)
    repair_map = {int(item["goal_index"]): int(item["to_name_choice"]) for item in repairs}
    result_edges: List[Any] = []

    for idx, raw_goal in enumerate(raw_edge_goals):
        parsed = _parse_edge_goal(raw_goal)
        if parsed is None:
            continue
        if idx not in repair_map:
            result_edges.append(raw_goal)
            continue

        relation_key = _canonical(parsed["relation"])
        allowed_targets = sorted(list(constraints.to_name_by_relation.get(relation_key) or []))
        chosen = repair_map[idx]
        if chosen < 0 or chosen >= len(allowed_targets):
            continue

        new_goal = [
            parsed["relation"],
            parsed["from_name"],
            allowed_targets[chosen],
        ]
        if parsed["negated"]:
            result_edges.append(["not", new_goal])
        else:
            result_edges.append(new_goal)

    return result_edges


def build_final_answer_text(
    node_goals: Optional[List[Any]],
    edge_goals: Optional[List[Any]],
) -> str:
    payload = {
        "node goals": node_goals or [],
        "edge goals": edge_goals or [],
    }
    return json.dumps(payload, ensure_ascii=False)
