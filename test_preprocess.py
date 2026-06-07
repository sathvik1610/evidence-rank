"""
test_preprocess.py — Phase Review test for Phase 1f (Honeypot & Ghost Detection)
Run: python test_preprocess.py
"""
import sys
import json
from datetime import date
sys.path.insert(0, ".")
from preprocess import _check_impossible_flag, _compute_honeypot_score, _is_ghost

print("=== Test 1: Ghost Detection (4-condition AND-gate) ===")
reference_date = date(2026, 6, 1)

ghost_cand = {
    "redrob_signals": {
        "last_active_date": "2025-01-01",  # > 365 days ago
        "recruiter_response_rate": 0.01,   # < 0.05
        "open_to_work_flag": False,        # == False
        "applications_submitted_30d": 0    # == 0
    }
}
assert _is_ghost(ghost_cand, reference_date) == True
print("  PASS: Perfect ghost detected")

# Break ONE condition — should no longer be a ghost
active_ghost = dict(ghost_cand)
active_ghost["redrob_signals"] = dict(ghost_cand["redrob_signals"])
active_ghost["redrob_signals"]["applications_submitted_30d"] = 1
assert _is_ghost(active_ghost, reference_date) == False
print("  PASS: 1 condition broken -> not a ghost")

print("\n=== Test 2: Impossible Tech Release (Tier 1) ===")
# Langchain released Oct 2022. Claiming 60 months (5 years) in 2026 implies started in 2021.
time_traveler = {
    "career_history": [],
    "skills": [{"name": "LangChain", "duration_months": 60}],
    "profile": {"years_of_experience": 5}
}
assert _check_impossible_flag(time_traveler) == True
print("  PASS: LangChain 60 months correctly flagged as impossible")

# Valid duration
honest_eng = {
    "career_history": [],
    "skills": [{"name": "LangChain", "duration_months": 12}],
    "profile": {"years_of_experience": 5}
}
assert _check_impossible_flag(honest_eng) == False
print("  PASS: LangChain 12 months allowed")

print("\n=== Test 3: Honeypot Scoring (Tier 2) ===")
# Maxed redrob signals + 2 simultaneous current roles
honeypot = {
    "career_history": [
        {"title": "Role 1", "is_current": True},
        {"title": "Role 2", "is_current": True}
    ],
    "skills": [],
    "redrob_signals": {
        "recruiter_response_rate": 1.0,
        "interview_completion_rate": 1.0,
        "offer_acceptance_rate": 1.0,
        "profile_completeness_score": 100
    },
    "profile": {"years_of_experience": 5}
}
score = _compute_honeypot_score(honeypot)
assert score >= 0.60, f"Expected high honeypot score, got {score}"
print(f"  PASS: High honeypot score computed: {score}")

print("\n=== Test 4: Sentinel Safety (Missing Dates) ===")
empty_cand = {}
assert _is_ghost(empty_cand, reference_date) == False
assert _check_impossible_flag(empty_cand) == False
assert _compute_honeypot_score(empty_cand) == 0.0
print("  PASS: Handled empty candidate safely")

print("\nAll preprocess logic tests passed successfully.")
