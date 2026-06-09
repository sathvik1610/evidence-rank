import json
from pathlib import Path
import pytest
import sys
sys.path.insert(0, ".")
from src.features import (
    extract_features, compute_product_ratio, score_skill_bucket,
    score_career_quality, score_fit_gaps, _build_career_text, extract_behavioral
)

@pytest.fixture
def sample_candidates():
    sample_path = Path("sample_candidates.json")
    if not sample_path.exists():
        sample_path = Path("Resources/sample_candidates.json")
    with sample_path.open(encoding="utf-8") as f:
        return json.load(f)

def test_smoke_extract_features(sample_candidates):
    """Smoke test: extract_features on all sample candidates."""
    for cand in sample_candidates:
        flags = {
            "product_ratio": compute_product_ratio(cand),
            "consulting_only": False,
            "research_only": False,
            "wrong_domain": False,
        }
        f = extract_features(cand, flags)
        assert "candidate_id" in f
        assert "retrieval_search" in f
        assert "snippets_json" in f
        # Must be valid JSON
        json.loads(f["snippets_json"])

def test_score_range_validation(sample_candidates):
    """Score range validation on the first 20 candidates."""
    for cand in sample_candidates[:20]:
        flags = {
            "product_ratio": compute_product_ratio(cand),
            "consulting_only": False,
            "research_only": False,
            "wrong_domain": False,
        }
        f = extract_features(cand, flags)
        pbs = f["product_builder_score"]
        rs  = f["retrieval_search"]
        ss  = f["seniority_score"]
        nd  = f["ninety_day_alignment"]
        
        assert 0.0 <= pbs <= 1.0, f"product_builder_score {pbs} out of bounds"
        assert 0.0 <= rs <= 3.5, f"retrieval_search {rs} out of bounds"
        assert ss in {0.75, 0.90, 0.95, 1.00}, f"seniority_score {ss} invalid"
        assert 0.0 <= nd <= 1.0, f"ninety_day_alignment {nd} out of bounds"

def test_realistic_strong_candidate():
    """Realistic candidate with strong IR/search background."""
    strong_cand = {
        "candidate_id": "TEST_STRONG",
        "profile": {"current_title": "ML Engineer", "years_of_experience": 6.0, "location": "Bangalore", "country": "India"},
        "career_history": [
            {
                "title": "Senior ML Engineer",
                "company": "Swiggy",
                "industry": "Food Tech",
                "duration_months": 36,
                "is_current": True,
                "description": "Built and shipped production retrieval system using FAISS and BM25 hybrid search. Deployed dense retrieval serving 10M+ queries per day. Implemented NDCG evaluation framework for offline A/B testing. Real users depend on this system daily."
            },
            {
                "title": "ML Engineer",
                "company": "Razorpay",
                "industry": "Fintech",
                "duration_months": 30,
                "is_current": False,
                "description": "Designed recommendation system and ranking pipeline. Used XGBoost learning-to-rank. Shipped to production users at scale. Built vector search infrastructure."
            }
        ],
        "skills": [
            {"name": "FAISS", "proficiency": "expert", "duration_months": 24, "endorsements": 5},
            {"name": "Python", "proficiency": "expert", "duration_months": 72, "endorsements": 12},
            {"name": "NDCG", "proficiency": "advanced", "duration_months": 18, "endorsements": 3},
        ],
        "redrob_signals": {
            "last_active_date": "2026-05-15", "open_to_work_flag": True,
            "recruiter_response_rate": 0.75, "github_activity_score": 72,
            "notice_period_days": 30, "applications_submitted_30d": 2
        },
        "education": []
    }
    flags_strong = {"product_ratio": 1.0, "consulting_only": False, "research_only": False, "wrong_domain": False}
    f_strong = extract_features(strong_cand, flags_strong)

    assert f_strong["retrieval_search"] >= 2.0
    assert f_strong["eval_framework"] >= 1.0
    assert f_strong["python_coding"] >= 1.0
    assert f_strong["product_builder_score"] > 0.5
    assert f_strong["sys_experience_score"] == 1.0
    assert f_strong["seniority_score"] == 1.0
    assert f_strong["depth_signal"] == 1.0

def test_jd_intent_evaluation_phrases_from_summary_and_roles():
    cand = {
        "candidate_id": "TEST_EVAL_INTENT",
        "profile": {
            "current_title": "Senior NLP Engineer",
            "years_of_experience": 7.0,
            "summary": (
                "Owned the offline-online evaluation harness with NDCG/MRR/recall "
                "calibrated to live A/B metrics for ranking quality."
            ),
        },
        "career_history": [
            {
                "title": "Senior NLP Engineer",
                "company": "ProductCo",
                "duration_months": 36,
                "description": (
                    "Built an end-to-end ranking pipeline with BGE embeddings, Pinecone retrieval, "
                    "XGBoost learning-to-rank, and behavioral-signal integration. The hardest part "
                    "was building offline metrics that predicted what the recommendation would do "
                    "to live engagement, validated through simulated A/B tests."
                ),
            }
        ],
        "skills": [],
        "redrob_signals": {},
        "education": [],
    }
    flags = {"product_ratio": 1.0, "consulting_only": False, "research_only": False, "wrong_domain": False}
    features = extract_features(cand, flags)

    assert features["retrieval_search"] >= 2.0
    assert features["vector_db_hybrid"] >= 2.0
    assert features["ltr_reranking"] >= 2.0
    assert features["eval_framework"] >= 2.0
    assert features["ninety_day_alignment"] > 0.8


def test_skill_snippet_prefers_concrete_role_evidence_over_generic_summary():
    candidate = {
        "profile": {
            "headline": "Strong background in NLP, recommendation systems, and applied AI.",
            "summary": "Strong background in NLP, recommendation systems, and applied AI; comfortable across the stack.",
        },
        "career_history": [
            {
                "title": "Search Engineer",
                "description": (
                    "Built BM25 + dense retrieval with BGE embeddings and FAISS, "
                    "then evaluated ranking quality using NDCG@10 and recall@K."
                ),
                "duration_months": 24,
            }
        ],
        "skills": [],
        "redrob_signals": {},
    }
    _, snippets = score_skill_bucket(candidate, _build_career_text(candidate))
    assert "Strong background" not in snippets["retrieval_search"]
    assert "BM25" in snippets["retrieval_search"] or "dense retrieval" in snippets["retrieval_search"]

def test_consulting_only_candidate():
    """Pure consulting candidate."""
    consulting_cand = {
        "candidate_id": "TEST_CONSULT",
        "profile": {"current_title": "Senior Consultant", "years_of_experience": 8.0, "location": "Mumbai", "country": "India"},
        "career_history": [
            {"title": "Senior Consultant", "company": "TCS", "industry": "IT Services",
             "duration_months": 48, "is_current": True, "description": "Worked on client AI projects using Python and machine learning."},
            {"title": "Analyst", "company": "Infosys", "industry": "IT Services",
             "duration_months": 48, "is_current": False, "description": "Built dashboards for enterprise clients."},
        ],
        "skills": [{"name": "Python", "proficiency": "intermediate", "duration_months": 60}],
        "redrob_signals": {"last_active_date": "2026-06-01", "open_to_work_flag": True},
        "education": []
    }
    product_ratio_consult = compute_product_ratio(consulting_cand)
    flags_consult = {"product_ratio": product_ratio_consult, "consulting_only": True, "research_only": False, "wrong_domain": False}
    f_consult = extract_features(consulting_cand, flags_consult)

    assert product_ratio_consult == 0.0
    assert f_consult["consulting_flag"] is True
    assert f_consult["product_builder_score"] < 0.2

def test_langchain_only_candidate():
    """LangChain-only candidate."""
    lc_cand = {
        "candidate_id": "TEST_LC",
        "profile": {"current_title": "AI Engineer", "years_of_experience": 1.5, "location": "Pune", "country": "India"},
        "career_history": [
            {"title": "AI Engineer", "company": "Startup", "industry": "Tech",
             "duration_months": 18, "is_current": True,
             "description": "Used LangChain and OpenAI API to build chatbot demos. Wrapped GPT-4 for document Q&A. Built LangChain pipelines."}
        ],
        "skills": [
            {"name": "LangChain", "proficiency": "advanced", "duration_months": 6},
            {"name": "OpenAI", "proficiency": "intermediate", "duration_months": 5},
        ],
        "redrob_signals": {},
        "education": []
    }
    flags_lc = {"product_ratio": 0.8, "consulting_only": False, "research_only": False, "wrong_domain": False}
    f_lc = extract_features(lc_cand, flags_lc)
    assert f_lc["langchain_only_flag"] is True
    assert f_lc["seniority_score"] < 1.0

def test_adversarial_empty_candidate():
    """Empty candidate (all fields missing)."""
    empty_cand = {"candidate_id": "EMPTY", "profile": {}, "career_history": [], "skills": [], "redrob_signals": {}, "education": []}
    flags_empty = {"product_ratio": 0.5, "consulting_only": False, "research_only": False, "wrong_domain": False}
    
    f_empty = extract_features(empty_cand, flags_empty)
    assert f_empty["product_builder_score"] >= 0.0
    assert f_empty["seniority_score"] == 1.0
    assert f_empty["langchain_only_flag"] is False
    assert f_empty["beh_offer_acceptance_rate"] == -1

def test_adversarial_extreme_yoe():
    """Adversarial candidate with extreme YoE (30 years)."""
    old_cand = {
        "candidate_id": "OLD_GUY",
        "profile": {"current_title": "CTO", "years_of_experience": 30.0, "location": "Delhi", "country": "India"},
        "career_history": [{"title": "CTO", "company": "MegaCorp", "industry": "Tech", "duration_months": 120, "is_current": True, "description": "Led AI strategy."}],
        "skills": [], "redrob_signals": {}, "education": []
    }
    flags_old = {"product_ratio": 0.5, "consulting_only": False, "research_only": False, "wrong_domain": False}
    f_old = extract_features(old_cand, flags_old)
    assert f_old["seniority_score"] == 0.90
    assert f_old["code_stopped"] is True
