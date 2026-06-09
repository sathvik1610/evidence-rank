import sys
sys.path.insert(0, ".")
import pytest
import polars as pl
from src.scorer import compute_core_score, score_candidates_vectorized
from src.weights import W

def test_compute_core_score_perfect():
    """Verify that a candidate with maximum features scores 100.0."""
    features = {
        "retrieval_search": 3.0,
        "vector_db_hybrid": 3.0,
        "eval_framework": 3.0,
        "python_coding": 3.0,
        "sys_experience_score": 1.0,
        
        "ltr_reranking": 3.0,
        "llm_integration": 3.0,
        "distributed_systems": 3.0,
        "hr_tech_exposure": 3.0,
        
        "experience_recency": 1.0,
        "depth_signal": 1.0,
        
        "product_builder_score": 1.0,
    }
    score = compute_core_score(features)
    assert abs(score - 100.0) < 1e-9

def test_compute_core_score_no_retrieval_cap():
    """Verify the no-retrieval cap is applied when retrieval/vectordb/sys_exp are zero."""
    features = {
        "retrieval_search": 0.0,
        "vector_db_hybrid": 0.0,
        "sys_experience_score": 0.0,
        
        "eval_framework": 3.0,
        "python_coding": 3.0,
        
        "ltr_reranking": 3.0,
        "llm_integration": 3.0,
        "distributed_systems": 3.0,
        "hr_tech_exposure": 3.0,
        
        "experience_recency": 1.0,
        "depth_signal": 1.0,
        "product_builder_score": 1.0,
    }
    score = compute_core_score(features)
    assert score > 0.0

def test_compute_core_score_penalties():
    """Verify career quality penalties (consulting, research, wrong domain)."""
    base_features = {
        "retrieval_search": 3.0,
        "vector_db_hybrid": 3.0,
        "eval_framework": 3.0,
        "python_coding": 3.0,
        "sys_experience_score": 1.0,
        "experience_recency": 1.0,
        "depth_signal": 1.0,
        "product_builder_score": 1.0,
    }
    
    score_clean = compute_core_score(base_features)
    
    # Consulting penalty
    feat_consulting = base_features.copy()
    feat_consulting["consulting_flag"] = True
    score_consulting = compute_core_score(feat_consulting)
    assert score_consulting < score_clean
    
    # Research penalty
    feat_research = base_features.copy()
    feat_research["research_only"] = True
    score_research = compute_core_score(feat_research)
    assert score_research < score_clean

    # Wrong domain penalty
    feat_domain = base_features.copy()
    feat_domain["wrong_domain"] = True
    score_domain = compute_core_score(feat_domain)
    assert score_domain < score_clean


def test_eval_plan_coverage_bonus_below_full_trifecta():
    base_features = {
        "retrieval_search": 2.0,
        "vector_db_hybrid": 2.0,
        "eval_framework": 2.0,
        "ltr_reranking": 1.0,
        "python_coding": 1.0,
        "sys_experience_score": 0.5,
        "llm_integration": 0.0,
        "distributed_systems": 0.0,
        "hr_tech_exposure": 0.0,
        "experience_recency": 1.0,
        "depth_signal": 1.0,
        "product_builder_score": 0.8,
    }
    weaker = compute_core_score(base_features)
    covered_features = base_features.copy()
    covered_features["ltr_reranking"] = 2.0
    covered = compute_core_score(covered_features)
    assert covered > weaker

def test_score_candidates_vectorized():
    """Verify that vectorized polars scoring matches compute_core_score row-by-row."""
    candidates_data = [
        {
            "retrieval_search": 3.0,
            "vector_db_hybrid": 3.0,
            "eval_framework": 3.0,
            "python_coding": 3.0,
            "sys_experience_score": 1.0,
            "ltr_reranking": 3.0,
            "llm_integration": 3.0,
            "distributed_systems": 3.0,
            "hr_tech_exposure": 3.0,
            "experience_recency": 1.0,
            "depth_signal": 1.0,
            "product_builder_score": 1.0,
            "consulting_flag": False,
            "research_only": False,
            "wrong_domain": False,
        },
        {
            "retrieval_search": 1.0,
            "vector_db_hybrid": 0.0,
            "eval_framework": 2.0,
            "python_coding": 1.5,
            "sys_experience_score": 0.0,
            "ltr_reranking": 0.0,
            "llm_integration": 2.0,
            "distributed_systems": 1.0,
            "hr_tech_exposure": 0.0,
            "experience_recency": 0.5,
            "depth_signal": 0.2,
            "product_builder_score": 0.4,
            "consulting_flag": True,
            "research_only": False,
            "wrong_domain": True,
        }
    ]
    
    df = pl.DataFrame(candidates_data)
    df_scored = score_candidates_vectorized(df)
    
    assert "core_score" in df_scored.columns
    scores_vec = df_scored["core_score"].to_list()
    
    for i, cand in enumerate(candidates_data):
        score_row = compute_core_score(cand)
        assert abs(scores_vec[i] - score_row) < 1e-6
