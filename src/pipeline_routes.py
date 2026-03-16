from typing import Any, Dict


ROUTE_COMMON_REPAIR = "common_repair"
ROUTE_STRUCTURAL_REPAIR = "structural_relation_repair"
ROUTE_ABSTAIN = "abstain"
ROUTE_NOOP = "noop"


def select_repair_route(diagnosis: Dict[str, Any]) -> str:
    if bool(diagnosis.get("should_abstain")):
        return ROUTE_ABSTAIN

    categories = diagnosis.get("diagnosed_categories") or []
    if "structural_relation_target_mismatch" in categories:
        return ROUTE_STRUCTURAL_REPAIR

    if categories:
        return ROUTE_COMMON_REPAIR
    return ROUTE_NOOP


def route_refine_budget(route: str, max_refine_rounds: int) -> int:
    if max_refine_rounds < 0:
        max_refine_rounds = 0
    if route == ROUTE_COMMON_REPAIR:
        return min(max_refine_rounds, 2)
    if route == ROUTE_STRUCTURAL_REPAIR:
        return min(max_refine_rounds, 1)
    return 0
