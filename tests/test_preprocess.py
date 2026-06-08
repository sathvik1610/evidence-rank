import sys
sys.path.insert(0, ".")
from datetime import date
import pytest
import pandas as pd
import constants
from src.weights import W
from preprocess import (
    _check_impossible_flag,
    _compute_honeypot_score,
    _is_ghost,
    run_phase_1f_honeypots,
)

def test_ghost_detection():
    """Ghost detection: 4-condition AND-gate."""
    reference_date = date(2026, 6, 1)
    
    ghost_cand = {
        "redrob_signals": {
            "last_active_date": "2025-01-01",  # > 365 days ago
            "recruiter_response_rate": 0.01,   # < 0.05
            "open_to_work_flag": False,        # == False
            "applications_submitted_30d": 0    # == 0
        }
    }
    assert _is_ghost(ghost_cand, reference_date) is True
    
    # Break ONE condition — should no longer be a ghost
    active_ghost = dict(ghost_cand)
    active_ghost["redrob_signals"] = dict(ghost_cand["redrob_signals"])
    active_ghost["redrob_signals"]["applications_submitted_30d"] = 1
    assert _is_ghost(active_ghost, reference_date) is False

def test_impossible_tech_release():
    """Impossible Tech Release (Tier 1)."""
    # Langchain released Oct 2022. Claiming 60 months (5 years) in 2026 implies started in 2021.
    time_traveler = {
        "career_history": [],
        "skills": [{"name": "LangChain", "duration_months": 60}],
        "profile": {"years_of_experience": 5}
    }
    assert _check_impossible_flag(time_traveler) is True
    
    # Valid duration
    honest_eng = {
        "career_history": [],
        "skills": [{"name": "LangChain", "duration_months": 12}],
        "profile": {"years_of_experience": 5}
    }
    assert _check_impossible_flag(honest_eng) is False

def test_honeypot_scoring():
    """Honeypot Scoring (Tier 2)."""
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

def test_uniform_descriptions_honeypot_triggers_for_two_roles():
    """Two identical role descriptions should trigger S-6 copy-paste signal."""
    cand = {
        "career_history": [
            {"description": "Built the same generic AI platform for client delivery."},
            {"description": "Built the same generic AI platform for client delivery."},
        ],
        "skills": [],
        "redrob_signals": {},
        "profile": {"years_of_experience": 4},
    }
    assert _compute_honeypot_score(cand) >= W["honeypot.s6_uniform_descriptions"]

def test_wrong_domain_escape_requires_description_evidence(tmp_path, monkeypatch):
    """BM25 in skills must not rescue pure CV profiles; BM25 in descriptions can."""
    out_path = tmp_path / "candidate_flags.parquet"
    monkeypatch.setattr(constants, "CANDIDATE_FLAGS_PARQUET", str(out_path))

    candidates = [
        {
            "candidate_id": "CAND_9000001",
            "profile": {"years_of_experience": 6},
            "career_history": [
                {
                    "title": "Computer Vision Engineer",
                    "company": "Ola",
                    "industry": "Mobility",
                    "duration_months": 36,
                    "description": "Built image classification and object detection pipelines.",
                }
            ],
            "skills": [{"name": "Computer Vision"}, {"name": "BM25"}],
            "redrob_signals": {},
        },
        {
            "candidate_id": "CAND_9000002",
            "profile": {"years_of_experience": 6},
            "career_history": [
                {
                    "title": "Computer Vision Engineer",
                    "company": "Ola",
                    "industry": "Mobility",
                    "duration_months": 36,
                    "description": "Built computer vision pipelines and later owned BM25 retrieval for image search.",
                }
            ],
            "skills": [{"name": "Computer Vision"}],
            "redrob_signals": {},
        },
    ]

    run_phase_1f_honeypots(candidates)
    rows = pd.read_parquet(out_path).set_index("candidate_id")
    assert rows.loc["CAND_9000001", "wrong_domain"] == True
    assert rows.loc["CAND_9000002", "wrong_domain"] == False

def test_current_cv_primary_domain_needs_deep_ir_escape(tmp_path, monkeypatch):
    """A current CV role should not escape on one older weak recsys mention."""
    out_path = tmp_path / "candidate_flags.parquet"
    monkeypatch.setattr(constants, "CANDIDATE_FLAGS_PARQUET", str(out_path))

    candidates = [
        {
            "candidate_id": "CAND_9000003",
            "profile": {"years_of_experience": 6},
            "career_history": [
                {
                    "title": "Computer Vision Engineer",
                    "company": "Ola",
                    "industry": "Mobility",
                    "duration_months": 18,
                    "is_current": True,
                    "description": "Built image classification and object detection pipelines.",
                },
                {
                    "title": "Junior ML Engineer",
                    "company": "Tech Mahindra",
                    "industry": "IT Services",
                    "duration_months": 24,
                    "description": "Built lightweight recommendation-style features for engagement.",
                },
            ],
            "skills": [{"name": "Computer Vision"}, {"name": "Recommendation Systems"}],
            "redrob_signals": {},
        },
        {
            "candidate_id": "CAND_9000004",
            "profile": {"years_of_experience": 6},
            "career_history": [
                {
                    "title": "Computer Vision Engineer",
                    "company": "Ola",
                    "industry": "Mobility",
                    "duration_months": 18,
                    "is_current": True,
                    "description": "Built BM25 retrieval and ranking for visual search relevance.",
                }
            ],
            "skills": [{"name": "Computer Vision"}],
            "redrob_signals": {},
        },
    ]

    run_phase_1f_honeypots(candidates)
    rows = pd.read_parquet(out_path).set_index("candidate_id")
    assert rows.loc["CAND_9000003", "wrong_domain"] == True
    assert rows.loc["CAND_9000004", "wrong_domain"] == False

def test_sentinel_safety_empty():
    """Sentinel Safety (Missing Dates / Empty profile)."""
    reference_date = date(2026, 6, 1)
    empty_cand = {}
    assert _is_ghost(empty_cand, reference_date) is False
    assert _check_impossible_flag(empty_cand) is False
    assert _compute_honeypot_score(empty_cand) == 0.0
