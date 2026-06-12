import sys
sys.path.insert(0, ".")
import pytest
from src.explainer import (
    get_90day_milestone, get_largest_concern, _profile_prefix, _trim_snippet,
    generate_reasoning
)

def test_get_90day_milestone():
    assert "Weeks 1-3" in get_90day_milestone("retrieval_search")
    assert "Weeks 4-8" in get_90day_milestone("vector_db_hybrid")
    assert "Weeks 9-12" in get_90day_milestone("eval_framework")
    assert "early engineering milestones" in get_90day_milestone("unknown_domain")

def test_get_largest_concern():
    assert "highly suspicious" in get_largest_concern({"impossible_flag": True})
    assert "not shortlist-fit" in get_largest_concern({"python_coding": 0.0})
    assert "120-day notice" in get_largest_concern({"beh_notice_period_days": 120})
    assert "Location/no-relocation" in get_largest_concern({
        "beh_country": "India",
        "beh_location": "Kolkata",
        "beh_willing_to_relocate": False,
    })
    assert "Research-heavy" in get_largest_concern({"research_only": True})
    assert "NLP/Search" in get_largest_concern({"wrong_domain": True})
    assert "LLM wrappers" in get_largest_concern({"langchain_only_flag": True})
    assert "consulting" in get_largest_concern({"consulting_only": True})
    assert "hands-on coding" in get_largest_concern({"code_stopped": True})
    assert "Short average tenure" in get_largest_concern({"title_velocity_flag": True})
    assert "title progression" in get_largest_concern({"title_bump_flag": True})
    assert "title-chasing risk" in get_largest_concern({"title_chaser_flag": True})
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
    assert "Excellent JD fit" in reason
    assert "and direct retrieval/search evidence" in reason
    assert "Designed hybrid dense-sparse vector search using Qdrant" in reason

def test_generate_reasoning_weak():
    cand = {
        "profile_current_title": "AI Dev",
        "profile_years_of_experience": 2.0,
        "rank": 40,
        "retrieval_search": 2.0
    }
    reason = generate_reasoning(cand)
    assert "search" in reason.lower() or "retrieval" in reason.lower()
    assert "production depth" in reason or "scale evidence" in reason or "system details" in reason


def test_generate_reasoning_mid_rank_uses_concrete_snippet():
    cand = {
        "profile_current_title": "Ranking Engineer",
        "rank": 52,
        "retrieval_search": 2.0,
        "eval_framework": 2.0,
        "snippets_json": '{"retrieval_search": "Owned semantic search with FAISS and human relevance judgments."}'
    }
    reason = generate_reasoning(cand)
    assert "Owned semantic search with FAISS" in reason
    assert "production depth" not in reason

def test_generate_reasoning_top_rank_moderate_is_positive():
    cand = {
        "candidate_id": "CAND_0000001",
        "profile_current_title": "AI Engineer",
        "profile_current_company": "Acme",
        "profile_years_of_experience": 6.0,
        "rank": 8,
        "retrieval_search": 2.0,
        "vector_db_hybrid": 1.0,
        "beh_notice_period_days": 30,
        "beh_recruiter_response_rate": 0.82,
    }
    reason = generate_reasoning(cand)
    assert "lacks deep production evidence" not in reason
    assert "30-day notice" in reason
    assert "82% recruiter response" in reason
    assert "Hiring fit is helped by" in reason


def test_generate_reasoning_missing_vector_blocks_full_plan_match():
    cand = {
        "candidate_id": "CAND_GAP",
        "profile_current_title": "Senior AI Engineer",
        "profile_current_company": "Meta",
        "profile_years_of_experience": 7.9,
        "rank": 4,
        "retrieval_search": 2.0,
        "sys_experience_score": 1.0,
        "vector_db_hybrid": 0.0,
        "ltr_reranking": 3.0,
        "eval_framework": 2.0,
        "python_coding": 1.0,
        "runtime_full_plan_signal": 0.85,
        "snippets_json": '{"ltr_reranking": "Owned learning-to-rank and relevance labeling work."}',
    }
    reason = generate_reasoning(cand)
    lower = reason.lower()
    assert "This is a full-plan match" not in reason
    assert "not a clean full-plan match" in lower or "missing must-have evidence" in lower
    assert "vector/hybrid" in reason


def test_generate_reasoning_duration_contradiction_always_caveats():
    cand = {
        "candidate_id": "CAND_DURATION",
        "profile_current_title": "Lead AI Engineer",
        "profile_current_company": "Razorpay",
        "profile_years_of_experience": 6.7,
        "rank": 6,
        "retrieval_search": 3.0,
        "sys_experience_score": 1.0,
        "vector_db_hybrid": 3.0,
        "ltr_reranking": 3.0,
        "eval_framework": 2.0,
        "python_coding": 2.0,
        "target_skill_duration_contradiction": 1,
        "snippets_json": '{"retrieval_search": "Built BM25 and dense retrieval for recruiter search."}',
    }
    reason = generate_reasoning(cand)
    assert "Skill-duration metadata has 1 overclaim signal" in reason
    assert "duration claims are not used" in reason


def test_generate_reasoning_large_ce_core_delta_caveats():
    cand = {
        "candidate_id": "CAND_DELTA",
        "profile_current_title": "Senior Data Scientist",
        "profile_current_company": "Amazon",
        "profile_years_of_experience": 7.6,
        "rank": 67,
        "core_score": 84.34,
        "ce_score": 50.0,
        "retrieval_search": 3.0,
        "sys_experience_score": 1.0,
        "vector_db_hybrid": 3.0,
        "ltr_reranking": 3.0,
        "eval_framework": 2.0,
        "python_coding": 1.0,
        "snippets_json": '{"ltr_reranking": "Trained and shipped multiple ranking models."}',
    }
    reason = generate_reasoning(cand)
    assert "Cross-encoder and handcrafted score strongly disagree" in reason


def test_generate_reasoning_low_rank_uses_concrete_snippet_when_available():
    cand = {
        "candidate_id": "CAND_LOW",
        "profile_current_title": "Search Engineer",
        "profile_current_company": "Rephrase.ai",
        "profile_years_of_experience": 4.8,
        "rank": 88,
        "retrieval_search": 2.0,
        "sys_experience_score": 1.0,
        "vector_db_hybrid": 2.0,
        "eval_framework": 2.0,
        "python_coding": 1.0,
        "runtime_full_plan_signal": 0.85,
        "snippets_json": '{"retrieval_search": "Owned semantic search with FAISS and relevance labels."}',
    }
    reason = generate_reasoning(cand)
    assert "Owned semantic search with FAISS" in reason
    assert "technically relevant for the JD" not in reason

def test_generate_reasoning_low_rank():
    cand = {
        "profile_current_title": "Junior QA",
        "profile_years_of_experience": 1.0,
        "rank": 80,
        "retrieval_search": 0.5,
        "core_score": 35.0
    }
    reason = generate_reasoning(cand)
    assert "Missing must-have evidence for core retrieval/ranking" in reason
    assert "Rank is limited" not in reason


def test_generate_reasoning_uses_natural_caveat_language():
    cand = {
        "profile_current_title": "Search Engineer",
        "profile_current_company": "FindMe",
        "profile_years_of_experience": 6.0,
        "rank": 40,
        "retrieval_search": 3.5,
        "beh_location": "Kolkata",
        "beh_willing_to_relocate": False,
        "snippets_json": '{"retrieval_search": "Designed hybrid dense-sparse vector search using Qdrant."}'
    }
    reason = generate_reasoning(cand)
    assert "The main caveat is logistics" in reason
    assert "preferred/welcome cities" in reason
    assert "Rank is limited" not in reason


def test_generate_reasoning_ltr_lead_is_not_redundant():
    cand = {
        "profile_current_title": "Ranking Engineer",
        "profile_current_company": "FindMe",
        "profile_years_of_experience": 6.0,
        "rank": 8,
        "ltr_reranking": 3.5,
        "snippets_json": '{"ltr_reranking": "reranking work"}'
    }
    reason = generate_reasoning(cand)
    assert "direct learning-to-rank/reranking evidence" in reason
    assert "including reranking work" not in reason


def test_trim_snippet_drops_partial_hyphenated_suffix():
    snippet = "Built a retrieval system with LLM-bas"
    trimmed = _trim_snippet(snippet)
    assert "LLM-bas" not in trimmed
    assert trimmed == "Built a retrieval system"


def test_trim_snippet_keeps_complete_hyphenated_phrase():
    snippet = "Owned the end-to-end ranking pipeline for a marketplace product"
    trimmed = _trim_snippet(snippet, limit=40)
    assert "end-to-end" in trimmed


def test_trim_snippet_removes_weak_connector_edges():
    snippet = "and shipped a production recommendation system at a marketplace product, going"
    assert _trim_snippet(snippet) == "shipped a production recommendation system at a marketplace product"


def test_trim_snippet_removes_heading_and_title_prefixes():
    assert (
        _trim_snippet("Search & Ranking Shipped the personalization infrastructure")
        == "Shipped the personalization infrastructure"
    )
    assert (
        _trim_snippet("Applied Scientist Shipped the personalization infrastructure")
        == "Shipped the personalization infrastructure"
    )


def test_trim_snippet_recovers_from_mid_word_windows():
    assert (
        _trim_snippet("iddle layer — the ranking and retrieval systems that decide what to show. Strong preferenc")
        == "ranking and retrieval systems that decide what to show"
    )
    assert (
        _trim_snippet("engagement history. Owned the offline-online correlation analysis that determined which")
        == "Owned the offline-online correlation analysis"
    )
    assert (
        _trim_snippet("sential parts: index refresh, query understanding, ranking calibration")
        == "index refresh, query understanding, ranking calibration"
    )
    assert (
        _trim_snippet("engineering early, optimizing offline metrics that didn't move online numbers, building be")
        == "optimizing offline metrics that didn't move online numbers"
    )
    assert (
        _trim_snippet("sentence-transformers, FAISS, the works, I've spent enough t")
        == "sentence-transformers, FAISS"
    )
    assert (
        _trim_snippet("learning-to-rank model over 9 months, Designed the relevance")
        == "learning-to-rank model over 9 months"
    )
    assert (
        _trim_snippet("(via sentence-transformer embeddings) for cold starts and a gradient-boosted")
        == "sentence-transformer embeddings for cold starts"
    )
    assert (
        _trim_snippet("Owned the offline-online correlation analysis that determined which")
        == "Owned the offline-online correlation analysis"
    )
    assert (
        _trim_snippet("9 months, Designed the relevance labeling pipeline mix of click-through data")
        == "Designed a relevance labeling pipeline using click-through data"
    )
    assert (
        _trim_snippet("(via sentence-transformer embeddings) for cold starts and a gradient-boosted")
        == "sentence-transformer embeddings for cold starts"
    )
    assert (
        _trim_snippet("Recently, I shipped our first RAG-based feature this year and now own the eval")
        == "shipped a RAG-based feature and owned evaluation work"
    )
    assert (
        _trim_snippet("BM25 setup, validated through human relevance judgments, Owned the ranking la")
        == "BM25 retrieval validated through human relevance judgments and ranking-layer ownership"
    )
    assert (
        _trim_snippet("BM25 setup, validated through human relevance judgments, AI Engineer Trained")
        == "BM25 retrieval validated through human relevance judgments"
    )
    assert (
        _trim_snippet("Engineer Implemented a RAG-based customer support chatbot integrated with our")
        == "Implemented a RAG-based customer support chatbot integrated"
    )
    assert (
        _trim_snippet("Implemented a RAG-based customer support chatbot integrated with our exist")
        == "Implemented a RAG-based customer support chatbot"
    )
    assert (
        _trim_snippet("Owned the offline-online correlation analysis that determined")
        == "Owned the offline-online correlation analysis"
    )
