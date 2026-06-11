import sys
sys.path.insert(0, ".")
from datetime import date
import pytest
from preprocess import (
    _check_impossible_flag,
    _compute_honeypot_score,
    _has_repeated_role_descriptions,
    _is_ghost,
    _target_skill_duration_contradictions,
    build_exact_recall_ranked_list,
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

def test_external_tech_release_dates_are_not_hard_kills():
    """Hard flags should rely on JSONL-internal contradictions, not external launch dates."""
    candidate = {
        "career_history": [],
        "skills": [{"name": "Pinecone", "duration_months": 88, "proficiency": "expert"}],
        "profile": {"years_of_experience": 8.0},
    }
    assert _check_impossible_flag(candidate) is False

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

def test_sentinel_safety_empty():
    """Sentinel Safety (Missing Dates / Empty profile)."""
    reference_date = date(2026, 6, 1)
    empty_cand = {}
    assert _is_ghost(empty_cand, reference_date) is False
    assert _check_impossible_flag(empty_cand) is False
    assert _compute_honeypot_score(empty_cand) == 0.0

def test_exact_recall_rescues_career_evidence():
    """Exact recall lane should rescue strong profile evidence before scoring."""
    strong = {
        "candidate_id": "STRONG_RECALL",
        "profile": {"current_title": "ML Engineer", "years_of_experience": 6.0},
        "career_history": [{
            "title": "Senior ML Engineer",
            "description": (
                "Built production hybrid search and candidate ranking systems "
                "with FAISS, BM25, vector search, and NDCG evaluation."
            ),
        }],
        "skills": [{"name": "Python"}],
    }
    weak = {
        "candidate_id": "WEAK_RECALL",
        "profile": {"current_title": "Marketing Manager", "years_of_experience": 6.0},
        "career_history": [{"title": "Marketing Manager", "description": "Managed campaigns."}],
        "skills": [{"name": "Python"}],
    }

    ranked = build_exact_recall_ranked_list([weak, strong])
    assert ranked == ["STRONG_RECALL"]

def test_exact_recall_skips_ghost_ids():
    """Ghost IDs should not be rescued into the widened retrieval pool."""
    candidate = {
        "candidate_id": "GHOST_RECALL",
        "profile": {"current_title": "ML Engineer", "years_of_experience": 6.0},
        "career_history": [{
            "title": "ML Engineer",
            "description": "Built production dense retrieval and ranking systems with FAISS.",
        }],
        "skills": [{"name": "FAISS"}],
    }

    ranked = build_exact_recall_ranked_list([candidate], ghost_ids={"GHOST_RECALL"})
    assert ranked == []

def test_target_skill_duration_contradictions_are_target_aware():
    candidate = {
        "profile": {"years_of_experience": 6.0},
        "skills": [
            {"name": "Pinecone", "duration_months": 88, "proficiency": "expert"},
            {"name": "Information Retrieval", "duration_months": 84, "proficiency": "advanced"},
            {"name": "Microsoft Excel", "duration_months": 120, "proficiency": "expert"},
        ],
    }

    count, max_overclaim = _target_skill_duration_contradictions(candidate)
    assert count == 2
    assert max_overclaim == 10.0

def test_target_skill_duration_ignores_non_target_or_junior_claims():
    candidate = {
        "profile": {"years_of_experience": 6.0},
        "skills": [
            {"name": "Microsoft Excel", "duration_months": 120, "proficiency": "expert"},
            {"name": "Pinecone", "duration_months": 88, "proficiency": "beginner"},
        ],
    }

    assert _target_skill_duration_contradictions(candidate) == (0, 0.0)

def test_repeated_role_descriptions_across_companies_are_impossible():
    repeated = (
        "Trained and shipped multiple ranking models for our product discovery feed. "
        "Designed features across content metadata, user behavior signals, and item "
        "engagement history. Owned offline-online correlation analysis for A/B tests."
    )
    candidate = {
        "career_history": [
            {"company": "Company A", "description": repeated},
            {"company": "Company B", "description": repeated},
            {"company": "Company C", "description": repeated},
        ],
        "skills": [],
        "profile": {"years_of_experience": 6.0},
    }

    assert _has_repeated_role_descriptions(candidate) is True
    assert _check_impossible_flag(candidate) is True


def test_many_expert_zero_duration_skills_are_impossible():
    candidate = {
        "profile": {"years_of_experience": 6.0},
        "career_history": [
            {
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 72,
                "is_current": True,
                "description": "Built production ML systems.",
            }
        ],
        "skills": [
            {"name": f"Skill {idx}", "proficiency": "expert", "duration_months": 0}
            for idx in range(8)
        ],
        "redrob_signals": {},
    }

    assert _check_impossible_flag(candidate) is True
