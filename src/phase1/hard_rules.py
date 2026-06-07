from typing import Dict, Any, Tuple
from .date_utils import compute_chrono_months

def check_hard_eliminations(candidate: Dict[str, Any], chrono_months: float) -> Tuple[bool, str]:
    """
    Evaluates Phase 1A hard rules.
    Returns (is_eliminated, reason_string).
    """
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])

    # Rule 1: Zero-Duration Expert
    for skill in skills:
        if skill.get("proficiency") in ("expert", "advanced"):
            duration = skill.get("duration_months")
            if duration is not None and duration == 0:
                return True, "Rule 1: Zero-Duration Expert"

    # Rule 2: Single Role Exceeds Career Timeline (+12mo)
    for role in career:
        role_duration = role.get("duration_months", 0)
        if role_duration > chrono_months + 12:
            return True, f"Rule 2: Single Role ({role_duration}mo) > Timeline ({chrono_months:.1f}mo) + 12"

    # Rule 3: Extreme Chronological Overlap (>1.5x)
    if chrono_months > 0:
        total_career_months = sum(r.get("duration_months", 0) for r in career)
        if total_career_months / chrono_months > 1.5:
            return True, f"Rule 3: Overlap Ratio ({total_career_months/chrono_months:.2f}x) > 1.5x"

    return False, ""
