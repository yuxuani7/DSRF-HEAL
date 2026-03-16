import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.io_utils import append_jsonl, ensure_dir, now_iso
from src.progress import ValidationProgress


OBJECTS_MARKERS = [
    "Relevant objects in the scene are:",
]

RELATIONS_DICT_MARKERS = [
    "All possible relationships are the keys of the following dictionary, and the corresponding values are their descriptions:",
]

TO_NAME_MARKERS = [
    "Each relation has a fixed set of objects to be its 'to_name' target. Here is a dictionary where keys are 'relation' and corresponding values is its possible set of 'to_name' objects:",
]

INITIAL_STATES_MARKERS = [
    "All initial states in the scene are:",
]

SECTION_END_MARKERS = [
    "All initial states in the scene are:",
    "All possible relationships are the keys of the following dictionary",
    "Symbolic goals format:",
    "Task Name and Goal Instructions:",
    "Now using json format",
]


@dataclass
class PromptConstraints:
    objects: Set[str]
    states_by_object: Dict[str, Set[str]]
    all_states: Set[str]
    relations: Set[str]
    to_name_by_relation: Dict[str, Set[str]]
    parse_notes: List[str]


@dataclass
class ParsedGoal:
    goal_type: str
    index: int
    raw_goal: Any
    object_name: Optional[str] = None
    state_name: Optional[str] = None
    relation_name: Optional[str] = None
    from_name: Optional[str] = None
    to_name: Optional[str] = None
    negated: bool = False


def validate_output_run(
    output_run_dir: Path,
    validations_root: Path,
) -> Tuple[int, int, int]:
    if not output_run_dir.exists():
        raise FileNotFoundError("Output run dir not found: {0}".format(output_run_dir))
    if not output_run_dir.is_dir():
        raise ValueError("Output run path is not a directory: {0}".format(output_run_dir))

    source_files = sorted(output_run_dir.glob("*.jsonl"))
    if not source_files:
        raise ValueError("No jsonl files found in: {0}".format(output_run_dir))

    target_run_dir = validations_root / output_run_dir.name
    ensure_dir(target_run_dir)

    total_files = 0
    total_records = 0
    total_hallucinations = 0
    progress_items = []
    for source_file in source_files:
        progress_items.append(
            (
                source_file.name,
                source_file.name,
                _count_non_empty_lines(source_file),
            )
        )
    progress = ValidationProgress(progress_items)
    progress.start()

    for source_file in source_files:
        total_files += 1
        target_file = target_run_dir / source_file.name
        if target_file.exists():
            target_file.unlink()

        with source_file.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                total_records += 1
                record = _validate_single_record(
                    raw_record_line=line,
                    source_run_dir=output_run_dir,
                    source_file=source_file,
                    line_no=line_no,
                )
                append_jsonl(target_file, record)
                if record.get("has_hallucination"):
                    total_hallucinations += 1
                source_info = record.get("source") or {}
                source_task_id = source_info.get("source_task_id") or "-"
                progress.update(
                    file_key=source_file.name,
                    task_id=str(source_task_id),
                    has_hallucination=bool(record.get("has_hallucination")),
                    parse_error=bool(record.get("source_record_parse_error")),
                )
    progress.finish()
    return total_files, total_records, total_hallucinations


def _count_non_empty_lines(file_path: Path) -> int:
    count = 0
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _validate_single_record(
    raw_record_line: str,
    source_run_dir: Path,
    source_file: Path,
    line_no: int,
) -> Dict[str, Any]:
    try:
        src_record = json.loads(raw_record_line)
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "validation_timestamp": now_iso(),
            "source": {
                "output_run_dir": source_run_dir.name,
                "output_file": source_file.name,
                "source_line_number": line_no,
                "source_task_id": None,
                "source_dataset": None,
                "source_run_id": None,
            },
            "has_hallucination": False,
            "is_empty_output": True,
            "empty_node_goals": True,
            "empty_edge_goals": True,
            "is_abstained": False,
            "goals": {
                "node_goals_count": 0,
                "edge_goals_count": 0,
            },
            "metrics": _empty_metrics(),
            "hallucination_details": [],
            "parsing": {
                "prompt_parse_notes": [],
                "output_parse_notes": [
                    "Source record json parse failed: {0}: {1}".format(
                        type(exc).__name__,
                        str(exc),
                    )
                ],
            },
            "source_record_parse_error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }

    prompt = str(((src_record.get("input") or {}).get("prompt") or ""))
    output_text = str(((src_record.get("output") or {}).get("text") or ""))
    validation_payload = validate_prompt_output(prompt=prompt, output_text=output_text)

    record = {
        "validation_timestamp": now_iso(),
        "source": {
            "output_run_dir": source_run_dir.name,
            "output_file": source_file.name,
            "source_line_number": line_no,
            "source_task_id": src_record.get("task_id"),
            "source_dataset": src_record.get("dataset"),
            "source_run_id": src_record.get("run_id"),
        },
    }
    record.update(validation_payload)
    return record


def validate_prompt_output(prompt: str, output_text: str) -> Dict[str, Any]:
    constraints = parse_prompt_constraints(prompt)
    output_obj, output_parse_notes = parse_output_payload(output_text)
    node_goals, edge_goals = extract_goals(output_obj)
    node_count = len(node_goals)
    edge_count = len(edge_goals)
    empty_node_goals = node_count == 0
    empty_edge_goals = edge_count == 0
    is_empty_output = empty_node_goals and empty_edge_goals

    metrics, details = evaluate_hallucinations(
        node_goals=node_goals,
        edge_goals=edge_goals,
        constraints=constraints,
    )
    has_hallucination = (
        metrics["o1"]["count"] > 0
        or metrics["s"]["count"] > 0
        or metrics["o2"]["count"] > 0
        or metrics["o3"]["count"] > 0
        or metrics["r"]["count"] > 0
    )
    return {
        "has_hallucination": has_hallucination,
        "is_empty_output": is_empty_output,
        "empty_node_goals": empty_node_goals,
        "empty_edge_goals": empty_edge_goals,
        "is_abstained": is_empty_output,
        "goals": {
            "node_goals_count": node_count,
            "edge_goals_count": edge_count,
        },
        "metrics": metrics,
        "hallucination_details": details,
        "parsing": {
            "prompt_parse_notes": constraints.parse_notes,
            "output_parse_notes": output_parse_notes,
            "constraint_coverage": {
                "object_count": len(constraints.objects),
                "relation_count": len(constraints.relations),
                "to_name_rule_count": len(constraints.to_name_by_relation),
            },
        },
    }


def parse_prompt_constraints(prompt: str) -> PromptConstraints:
    notes: List[str] = []
    objects, states_by_object, object_notes = _parse_objects_and_states(prompt)
    notes.extend(object_notes)

    relations, relation_notes = _parse_relations(prompt)
    notes.extend(relation_notes)

    to_name_by_relation, to_name_notes = _parse_relation_to_name_rules(prompt)
    notes.extend(to_name_notes)

    all_states: Set[str] = set()
    for states in states_by_object.values():
        all_states.update(states)

    return PromptConstraints(
        objects=objects,
        states_by_object=states_by_object,
        all_states=all_states,
        relations=relations,
        to_name_by_relation=to_name_by_relation,
        parse_notes=notes,
    )


def parse_output_payload(output_text: str) -> Tuple[Dict[str, Any], List[str]]:
    notes: List[str] = []
    if not output_text.strip():
        notes.append("Output text is empty.")
        return {}, notes

    candidates: List[str] = []
    cleaned = output_text.strip()
    candidates.append(cleaned)
    fenced = _strip_code_fence(cleaned)
    if fenced and fenced not in candidates:
        candidates.append(fenced)

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        brace_slice = cleaned[first_brace : last_brace + 1]
        if brace_slice not in candidates:
            candidates.append(brace_slice)

    for candidate in candidates:
        parsed = _parse_json_or_literal(candidate)
        if isinstance(parsed, dict):
            return parsed, notes

    notes.append("Cannot parse full output into dict, fallback to key extraction.")
    fallback = {}
    node_value = _extract_value_by_key(cleaned, ["node goals", "node_goals"])
    edge_value = _extract_value_by_key(cleaned, ["edge goals", "edge_goals"])
    if node_value is not None:
        fallback["node goals"] = node_value
    if edge_value is not None:
        fallback["edge goals"] = edge_value
    if fallback:
        return fallback, notes

    notes.append("Fallback key extraction failed for node/edge goals.")
    return {}, notes


def extract_goals(output_obj: Dict[str, Any]) -> Tuple[List[ParsedGoal], List[ParsedGoal]]:
    node_raw = None
    edge_raw = None
    for key in ("node goals", "node_goals", "Node goals"):
        if key in output_obj:
            node_raw = output_obj.get(key)
            break
    for key in ("edge goals", "edge_goals", "Edge goals"):
        if key in output_obj:
            edge_raw = output_obj.get(key)
            break

    node_goals = _parse_node_goals(node_raw)
    edge_goals = _parse_edge_goals(edge_raw)
    return node_goals, edge_goals


def evaluate_hallucinations(
    node_goals: List[ParsedGoal],
    edge_goals: List[ParsedGoal],
    constraints: PromptConstraints,
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    details: List[Dict[str, Any]] = []
    counts = {"o1": 0, "s": 0, "o2": 0, "r": 0, "o3": 0}

    for goal in node_goals:
        obj = _canonical_token(goal.object_name)
        state = _canonical_token(goal.state_name)
        obj_exists = not constraints.objects or obj in constraints.objects
        if not obj_exists:
            counts["o1"] += 1
            details.append(
                _detail_item(
                    hallucination_type="O1",
                    goal=goal,
                    reason="Node goal object not found in relevant objects: {0}".format(
                        goal.object_name
                    ),
                )
            )

        state_ok = True
        if state:
            if obj in constraints.states_by_object:
                state_ok = state in constraints.states_by_object[obj]
            elif constraints.all_states:
                state_ok = state in constraints.all_states
        if not state_ok:
            counts["s"] += 1
            details.append(
                _detail_item(
                    hallucination_type="S",
                    goal=goal,
                    reason="Node goal state not allowed: object={0}, state={1}".format(
                        goal.object_name,
                        goal.state_name,
                    ),
                )
            )

    for goal in edge_goals:
        relation = _canonical_token(goal.relation_name)
        from_name = _canonical_token(goal.from_name)
        to_name = _canonical_token(goal.to_name)

        relation_ok = True
        if constraints.relations:
            relation_ok = relation in constraints.relations
        if not relation_ok:
            counts["r"] += 1
            details.append(
                _detail_item(
                    hallucination_type="R",
                    goal=goal,
                    reason="Edge goal relation not allowed: {0}".format(goal.relation_name),
                )
            )

        object_ok = True
        if constraints.objects:
            if from_name not in constraints.objects or to_name not in constraints.objects:
                object_ok = False
        if not object_ok:
            counts["o2"] += 1
            details.append(
                _detail_item(
                    hallucination_type="O2",
                    goal=goal,
                    reason="Edge goal object not found: from={0}, to={1}".format(
                        goal.from_name,
                        goal.to_name,
                    ),
                )
            )

        o3_triggered = False
        # O3 is only for existing objects with invalid to_name for the relation.
        # If objects are missing, that should be O2 only.
        if object_ok and relation in constraints.to_name_by_relation:
            allowed_to = constraints.to_name_by_relation[relation]
            if to_name and allowed_to and to_name not in allowed_to:
                o3_triggered = True
        if o3_triggered:
            counts["o3"] += 1
            details.append(
                _detail_item(
                    hallucination_type="O3",
                    goal=goal,
                    reason="Edge goal to_name not allowed for relation: relation={0}, to={1}".format(
                        goal.relation_name,
                        goal.to_name,
                    ),
                )
            )

    node_den = len(node_goals)
    edge_den = len(edge_goals)
    metrics = {
        "o1": _ratio_payload(counts["o1"], node_den),
        "s": _ratio_payload(counts["s"], node_den),
        "o2": _ratio_payload(counts["o2"], edge_den),
        "r": _ratio_payload(counts["r"], edge_den),
        "o3": _ratio_payload(counts["o3"], edge_den),
    }
    return metrics, details


def _ratio_payload(count: int, denominator: int) -> Dict[str, Any]:
    ratio = 0.0
    if denominator > 0:
        ratio = float(count) / float(denominator)
    return {
        "count": count,
        "denominator": denominator,
        "ratio": ratio,
        "percentage": ratio * 100.0,
        "score": "{0}/{1}".format(count, denominator),
    }


def _empty_metrics() -> Dict[str, Dict[str, Any]]:
    return {
        "o1": _ratio_payload(0, 0),
        "s": _ratio_payload(0, 0),
        "o2": _ratio_payload(0, 0),
        "r": _ratio_payload(0, 0),
        "o3": _ratio_payload(0, 0),
    }


def _detail_item(hallucination_type: str, goal: ParsedGoal, reason: str) -> Dict[str, Any]:
    return {
        "type": hallucination_type,
        "goal_type": goal.goal_type,
        "goal_index": goal.index,
        "reason": reason,
        "goal": goal.raw_goal,
        "parsed_goal": {
            "object_name": goal.object_name,
            "state_name": goal.state_name,
            "relation_name": goal.relation_name,
            "from_name": goal.from_name,
            "to_name": goal.to_name,
            "negated": goal.negated,
        },
    }


def _parse_objects_and_states(prompt: str) -> Tuple[Set[str], Dict[str, Set[str]], List[str]]:
    notes: List[str] = []
    section = _extract_section(prompt, OBJECTS_MARKERS, SECTION_END_MARKERS)
    if not section:
        return set(), {}, ["Cannot find objects section marker."]

    objects: Set[str] = set()
    states_by_object: Dict[str, Set[str]] = {}
    lines = [line.strip() for line in section.splitlines() if line.strip()]

    vh_pattern = re.compile(
        r"^([^,:]+?)\s*,\s*initial states:\s*(\[[^\]]*\])\s*,\s*possible states:\s*(\[[^\]]*\])\s*$"
    )
    bh_pattern = re.compile(r"^([^:]+?)\s*:\s*(\[[^\]]*\])\s*$")

    for line in lines:
        m_vh = vh_pattern.match(line)
        if m_vh:
            object_name = m_vh.group(1).strip()
            states_literal = m_vh.group(3).strip()
            states_list = _safe_literal_to_list(states_literal)
            obj_key = _canonical_token(object_name)
            objects.add(obj_key)
            states_by_object[obj_key] = {
                _canonical_token(s) for s in states_list if str(s).strip()
            }
            continue

        m_bh = bh_pattern.match(line)
        if m_bh:
            object_name = m_bh.group(1).strip()
            states_literal = m_bh.group(2).strip()
            states_list = _safe_literal_to_list(states_literal)
            obj_key = _canonical_token(object_name)
            objects.add(obj_key)
            states_by_object[obj_key] = {
                _canonical_token(s) for s in states_list if str(s).strip()
            }
            continue

    if not objects:
        notes.append("Objects section found but no object entries parsed.")
    return objects, states_by_object, notes


def _parse_relations(prompt: str) -> Tuple[Set[str], List[str]]:
    notes: List[str] = []
    relations: Set[str] = set()

    rel_dict, ok = _extract_dict_after_markers(prompt, RELATIONS_DICT_MARKERS)
    if ok and isinstance(rel_dict, dict):
        for key in rel_dict.keys():
            relations.add(_canonical_token(str(key)))
    elif ok:
        notes.append("Relationships marker found but parsed value is not a dict.")

    line_match = re.search(
        r"All relations should only be within the following set:\s*([^\n.]+)",
        prompt,
        flags=re.IGNORECASE,
    )
    if line_match:
        token_text = line_match.group(1)
        for token in _extract_relation_tokens_from_text(token_text):
            relations.add(token)

    edge_set_match = re.search(
        r"Edge goal states[^\n]*?comes from the set\s*\{([^}]*)\}",
        prompt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if edge_set_match:
        for token in _extract_relation_tokens_from_text(edge_set_match.group(1)):
            relations.add(token)

    init_section = _extract_section(
        prompt,
        INITIAL_STATES_MARKERS,
        ["Symbolic goals format:", "Task Name and Goal Instructions:"],
    )
    if init_section:
        for line in init_section.splitlines():
            line = line.strip()
            if not line or not line.startswith("["):
                continue
            parsed = _parse_json_or_literal(line)
            if isinstance(parsed, (list, tuple)) and len(parsed) >= 3:
                rel = parsed[0]
                if isinstance(rel, str):
                    relations.add(_canonical_token(rel))

    if not relations:
        notes.append("No relationship constraints parsed from prompt.")
    return relations, notes


def _extract_relation_tokens_from_text(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in text.split(","):
        candidate = raw.strip()
        if not candidate:
            continue
        candidate = candidate.strip(" \t\r\n{}[]()'\"`.")
        if not candidate:
            continue
        # Strict token policy: relation names are single identifiers.
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", candidate):
            continue
        tokens.append(_canonical_token(candidate))
    return tokens


def _parse_relation_to_name_rules(prompt: str) -> Tuple[Dict[str, Set[str]], List[str]]:
    notes: List[str] = []
    to_name_by_relation: Dict[str, Set[str]] = {}

    parsed_dict, ok = _extract_dict_after_markers(prompt, TO_NAME_MARKERS)
    if not ok:
        notes.append("No relation-to-to_name constraint marker found.")
        return to_name_by_relation, notes
    if not isinstance(parsed_dict, dict):
        notes.append("to_name marker found but parsed value is not a dict.")
        return to_name_by_relation, notes

    for key, value in parsed_dict.items():
        rel_key = _canonical_token(str(key))
        names: Set[str] = set()
        if isinstance(value, (set, list, tuple)):
            for item in value:
                token = _canonical_token(str(item))
                if token:
                    names.add(token)
        to_name_by_relation[rel_key] = names
    return to_name_by_relation, notes


def _extract_dict_after_markers(prompt: str, markers: List[str]) -> Tuple[Optional[Any], bool]:
    for marker in markers:
        idx = prompt.find(marker)
        if idx == -1:
            continue
        brace_idx = prompt.find("{", idx)
        if brace_idx == -1:
            return None, True
        block = _extract_balanced_block(prompt, brace_idx, "{", "}")
        if not block:
            return None, True
        parsed = _parse_json_or_literal(block)
        return parsed, True
    return None, False


def _extract_section(prompt: str, start_markers: List[str], end_markers: List[str]) -> str:
    start_idx = -1
    start_marker_len = 0
    for marker in start_markers:
        idx = prompt.find(marker)
        if idx != -1:
            start_idx = idx
            start_marker_len = len(marker)
            break
    if start_idx == -1:
        return ""

    section_start = start_idx + start_marker_len
    end_idx = len(prompt)
    for end_marker in end_markers:
        idx = prompt.find(end_marker, section_start)
        if idx != -1 and idx < end_idx:
            end_idx = idx
    return prompt[section_start:end_idx]


def _extract_balanced_block(
    text: str,
    start_idx: int,
    open_char: str,
    close_char: str,
) -> str:
    if start_idx < 0 or start_idx >= len(text):
        return ""
    if text[start_idx] != open_char:
        return ""

    depth = 0
    quote_char = ""
    escaped = False
    for idx in range(start_idx, len(text)):
        ch = text[idx]
        if quote_char:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == quote_char:
                quote_char = ""
            continue

        if ch == "'" or ch == '"':
            quote_char = ch
            continue
        if ch == open_char:
            depth += 1
        elif ch == close_char:
            depth -= 1
            if depth == 0:
                return text[start_idx : idx + 1]
    return ""


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if not lines:
        return stripped
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_value_by_key(text: str, keys: List[str]) -> Optional[Any]:
    for key in keys:
        pattern = re.compile(r"['\"]{0}['\"]\s*:\s*".format(re.escape(key)), re.IGNORECASE)
        match = pattern.search(text)
        if not match:
            continue
        idx = match.end()
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            continue
        ch = text[idx]
        if ch == "[":
            block = _extract_balanced_block(text, idx, "[", "]")
            if block:
                parsed = _parse_json_or_literal(block)
                if parsed is not None:
                    return parsed
        elif ch == "{":
            block = _extract_balanced_block(text, idx, "{", "}")
            if block:
                parsed = _parse_json_or_literal(block)
                if parsed is not None:
                    return parsed
    return None


def _parse_json_or_literal(text: str) -> Optional[Any]:
    s = text.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:  # pylint: disable=broad-except
        pass
    try:
        return ast.literal_eval(s)
    except Exception:  # pylint: disable=broad-except
        return None


def _safe_literal_to_list(text: str) -> List[Any]:
    parsed = _parse_json_or_literal(text)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, tuple):
        return list(parsed)
    return []


def _parse_node_goals(node_raw: Any) -> List[ParsedGoal]:
    if not isinstance(node_raw, list):
        return []
    goals: List[ParsedGoal] = []
    for idx, item in enumerate(node_raw):
        goal = _parse_single_node_goal(item, idx, negated=False)
        if goal is not None:
            goals.append(goal)
    return goals


def _parse_single_node_goal(item: Any, index: int, negated: bool) -> Optional[ParsedGoal]:
    if isinstance(item, dict):
        name = item.get("name")
        state = item.get("state")
        if isinstance(name, str) and isinstance(state, str):
            return ParsedGoal(
                goal_type="node",
                index=index,
                raw_goal=item,
                object_name=name,
                state_name=state,
                negated=negated,
            )
        return None

    if isinstance(item, (list, tuple)):
        values = list(item)
        if len(values) == 2 and isinstance(values[0], str) and values[0].strip().lower() == "not":
            return _parse_single_node_goal(values[1], index, negated=True)
        if len(values) == 2 and isinstance(values[0], str) and isinstance(values[1], str):
            return ParsedGoal(
                goal_type="node",
                index=index,
                raw_goal=item,
                object_name=values[1],
                state_name=values[0],
                negated=negated,
            )
        if len(values) == 1:
            return _parse_single_node_goal(values[0], index, negated=negated)
    return None


def _parse_edge_goals(edge_raw: Any) -> List[ParsedGoal]:
    if not isinstance(edge_raw, list):
        return []
    goals: List[ParsedGoal] = []
    for idx, item in enumerate(edge_raw):
        goal = _parse_single_edge_goal(item, idx, negated=False)
        if goal is not None:
            goals.append(goal)
    return goals


def _parse_single_edge_goal(item: Any, index: int, negated: bool) -> Optional[ParsedGoal]:
    if isinstance(item, dict):
        relation = item.get("relation")
        from_name = item.get("from_name")
        to_name = item.get("to_name")
        if isinstance(relation, str) and isinstance(from_name, str) and isinstance(to_name, str):
            return ParsedGoal(
                goal_type="edge",
                index=index,
                raw_goal=item,
                relation_name=relation,
                from_name=from_name,
                to_name=to_name,
                negated=negated,
            )
        return None

    if isinstance(item, (list, tuple)):
        values = list(item)
        if len(values) == 2 and isinstance(values[0], str) and values[0].strip().lower() == "not":
            return _parse_single_edge_goal(values[1], index, negated=True)
        if (
            len(values) == 3
            and isinstance(values[0], str)
            and isinstance(values[1], str)
            and isinstance(values[2], str)
        ):
            return ParsedGoal(
                goal_type="edge",
                index=index,
                raw_goal=item,
                relation_name=values[0],
                from_name=values[1],
                to_name=values[2],
                negated=negated,
            )
        if len(values) == 1:
            return _parse_single_edge_goal(values[0], index, negated=negated)
    return None


def _canonical_token(value: Any) -> str:
    """
    Case-insensitive matching with strict equality on all other characters.
    """
    return str(value or "").strip().casefold()
