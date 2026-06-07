import datetime
from typing import Dict, Any, List

def parse_date(d_str: str) -> datetime.date:
    """Safely parse dates into python date objects."""
    if not d_str:
        return None
    try:
        if len(d_str) == 7:
            return datetime.datetime.strptime(d_str, "%Y-%m").date()
        elif len(d_str) == 10:
            return datetime.datetime.strptime(d_str, "%Y-%m-%d").date()
        elif len(d_str) == 4:
            return datetime.datetime.strptime(d_str, "%Y").date()
    except Exception:
        return None
    return None

def compute_chrono_months(career: List[Dict[str, Any]], reference_date: datetime.date = datetime.date(2026, 6, 1)) -> float:
    """Compute exact chronological passage of time from career dates."""
    start_dates = []
    end_dates = []
    
    for role in career:
        sd = parse_date(role.get("start_date"))
        ed = parse_date(role.get("end_date"))
        if sd: start_dates.append(sd)
        if ed: end_dates.append(ed)
        if not ed and role.get("start_date"):
            # Assume current role goes to present (reference date)
            end_dates.append(reference_date)

    if start_dates and end_dates:
        min_start = min(start_dates)
        max_end = max(end_dates)
        return max(0.0, (max_end - min_start).days / 30.436875)
    return 0.0
