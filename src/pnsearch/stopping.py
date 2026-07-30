from __future__ import annotations

from pnsearch.config import Settings
from pnsearch.schema import SearchState


def should_stop(state: SearchState, settings: Settings) -> tuple[bool, str]:
    if state.api_calls >= settings.max_api_calls:
        return True, "api_budget_exhausted"
    if len(state.history) >= settings.max_rounds:
        return True, "max_rounds_reached"
    if len(state.history) < 2:
        return False, ""
    last_two = state.history[-2:]
    low_gain = all(item.new_select_count <= settings.stop_new_select_threshold for item in last_two)
    noisy = all(item.reject_ratio >= settings.stop_reject_ratio for item in last_two)
    repetitive = all(item.duplicate_ratio >= settings.stop_duplicate_ratio for item in last_two)
    conditions = sum((low_gain, noisy, repetitive))
    if conditions >= 2:
        reasons = []
        if low_gain:
            reasons.append("low_marginal_gain")
        if noisy:
            reasons.append("high_reject_ratio")
        if repetitive:
            reasons.append("high_duplicate_ratio")
        return True, "+".join(reasons)
    return False, ""

