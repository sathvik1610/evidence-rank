import sys
sys.path.insert(0, ".")
import pytest
from src.explainer import (
    get_90day_milestone, get_largest_concern, _profile_prefix, generate_reasoning
)

def test_get_90day_milestone():
    assert "Weeks 1-3" in get_90day_milestone("retrieval_search")
    assert "Weeks 4-8" in get_90day_milestone("vector_db_hybrid")
    assert "Weeks 9-12" in get_90day_milestone("eval_framework")
    assert "early engineering milestones" in get_90day_milestone("unknown_domain")

def test_get_largest_concern():
    assert "highly suspicious" in get_largest_concern({"impossible_flag": True})
    assert "Research-heavy" in get_largest_concern({"research_only": True})
    assert "NLP/Search" in get_largest_concern({"wrong_domain": True})
    assert "LLM wrappers" in get_largest_concern({"langchain_only_flag": True})
    assert "consulting" in get_largest_concern({"consulting_only": True})
    assert "hands-on coding" in get_largest_concern({"code_stopped": True})
    assert "Frequent title changes" in get_largest_concern({"title_velocity_flag": True})
    assert get_largest_concern({}) == ""

def test_profile_prefix():
    cand = {
        "profile_current_title": "Senior Engineer",
        "profile_current_company": "Acme Inc.",
        "profile_years_of_experience": 5.5
    }
    prefix = _profile_prefix(cand)
    assert "Senior Engineer at Acme Inc. with 5.5 years of experience" in prefix
    
    cand_unknown = {
        "profile_current_title": "UNKNOWN",
        "profile_current_company": "UNKNOWN",
        "profile_years_of_experience": -1
    }
    assert _profile_prefix(cand_unknown) == ""

def test_generate_reasoning_strong():
    cand = {
        "profile_current_title": "Search Engineer",
        "profile_current_company": "FindMe",
        "profile_years_of_experience": 6.0,
        "rank": 10,
        "retrieval_search": 3.5,
        "snippets_json": '{"retrieval_search": "Designed hybrid dense-sparse vector search using Qdrant."}'
    }
    reason = generate_reasoning(cand)
    assert "Search Engineer at FindMe with 6.0 years of experience" in reason
    assert "Strong evidence in Retrieval Systems" in reason
    assert "Designed hybrid dense-sparse vector search using Qdrant." in reason

def test_generate_reasoning_weak():
    cand = {
        "profile_current_title": "AI Dev",
        "profile_years_of_experience": 2.0,
        "rank": 40,
        "retrieval_search": 2.0
    }
    reason = generate_reasoning(cand)
    assert "Mentions Retrieval Systems but lacks deep production evidence" in reason

def test_generate_reasoning_low_rank():
    cand = {
        "profile_current_title": "Junior QA",
        "profile_years_of_experience": 1.0,
        "rank": 80,
        "retrieval_search": 0.5,
        "core_score": 35.0
    }
    reason = generate_reasoning(cand)
    assert "Failed to meet the technical depth" in reason
