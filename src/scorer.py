"""
src/scorer.py — Phase 4: Core Scoring

Implements the 4-part weighted formula mapping exactly to the JD contract:
- 55% Must-Have Score (IR, Vector DB, Eval, Python, System Experience)
- 10% Nice-to-Have Score (LTR, LLM, Dist Sys, HR Tech)
- 15% Career Quality Score (Recency, Depth)
- 20% Product Builder Score (Deployment, Shipper signals, Ownership)

Operates entirely on the flat feature dictionary produced by Phase 1c (features.py).
"""

from typing import Dict, Any

def compute_core_score(features: Dict[str, Any]) -> float:
    """
    Computes the 0-100 Core Score for a single candidate based on Phase 4 design.
    """
    # --- Must-Have Score (55%) ---
    retrieval_ev = features.get("retrieval_search", 0.0) / 3.0
    vectordb_ev = features.get("vector_db_hybrid", 0.0) / 3.0
    eval_ev = features.get("eval_framework", 0.0) / 3.0
    python_ev = features.get("python_coding", 0.0) / 3.0
    sys_experience_score = features.get("sys_experience_score", 0.0)

    must_have_raw = (
        0.25 * retrieval_ev +
        0.20 * vectordb_ev +
        0.20 * sys_experience_score +
        0.10 * eval_ev +
        0.05 * python_ev
    )

    # Softened Hard cap: cap at 0.5 if no retrieval/recsys evidence exists
    has_any_retrieval_or_recsys = (
        features.get("retrieval_search", 0.0) > 0 or
        features.get("vector_db_hybrid", 0.0) > 0 or
        sys_experience_score > 0.0
    )
    if not has_any_retrieval_or_recsys:
        must_have_raw = min(must_have_raw, 0.5)

    must_have_score = min(must_have_raw / 0.80, 1.0)


    # --- Nice-to-Have Score (10%) ---
    ltr_ev = features.get("ltr_reranking", 0.0) / 3.0
    llm_ev = features.get("llm_integration", 0.0) / 3.0
    dist_ev = features.get("distributed_systems", 0.0) / 3.0
    hr_ev = features.get("hr_tech_exposure", 0.0) / 3.0

    nice_to_have_score = min((
        0.04 * ltr_ev +
        0.03 * llm_ev +
        0.02 * dist_ev +
        0.01 * hr_ev
    ) / 0.10, 1.0)


    # --- Career Quality Score (15%) ---
    experience_recency = features.get("experience_recency", 0.5)
    depth_signal = features.get("depth_signal", 0.0)

    career_quality_raw = (
        0.08 * sys_experience_score +
        0.04 * experience_recency +
        0.03 * depth_signal
    )

    # Multipliers (these are boolean flags in features)
    if features.get("consulting_flag", False):
        career_quality_raw *= 0.4
    if features.get("research_only", False):
        career_quality_raw *= 0.5
    if features.get("wrong_domain", False):
        career_quality_raw *= 0.3

    career_quality_score = min(career_quality_raw / 0.15, 1.0)


    # --- Product Builder Score (20%) ---
    # Pre-computed and normalized to [0,1] in Phase 3
    product_builder_score = features.get("product_builder_score", 0.0)


    # --- Combined Weighted Score ---
    core_score = (
        0.55 * must_have_score +
        0.10 * nice_to_have_score +
        0.15 * career_quality_score +
        0.20 * product_builder_score
    ) * 100.0

    return float(core_score)


def score_candidates_vectorized(df) -> object:
    """
    Vectorized Polars execution for the core_score.
    Returns a DataFrame with a new `core_score` column.
    """
    import polars as pl
    
    # Must Have Raw
    must_have_raw = (
        0.25 * (pl.col("retrieval_search") / 3.0) +
        0.20 * (pl.col("vector_db_hybrid") / 3.0) +
        0.20 * pl.col("sys_experience_score") +
        0.10 * (pl.col("eval_framework") / 3.0) +
        0.05 * (pl.col("python_coding") / 3.0)
    )
    
    # Cap if no retrieval/recsys evidence
    has_retrieval_or_recsys = (
        (pl.col("retrieval_search") > 0) |
        (pl.col("vector_db_hybrid") > 0) |
        (pl.col("sys_experience_score") > 0)
    )
    
    capped_must_have_raw = pl.when(has_retrieval_or_recsys).then(must_have_raw).otherwise(
        pl.when(must_have_raw > 0.5).then(0.5).otherwise(must_have_raw)
    )
    
    # Clip must_have to 1.0 max
    must_have_score = (capped_must_have_raw / 0.80).clip(upper_bound=1.0)
    
    # Nice to have
    nice_to_have_score = (
        (
            0.04 * (pl.col("ltr_reranking") / 3.0) +
            0.03 * (pl.col("llm_integration") / 3.0) +
            0.02 * (pl.col("distributed_systems") / 3.0) +
            0.01 * (pl.col("hr_tech_exposure") / 3.0)
        ) / 0.10
    ).clip(upper_bound=1.0)
    
    # Career Quality
    career_quality_raw = (
        0.08 * pl.col("sys_experience_score") +
        0.04 * pl.col("experience_recency") +
        0.03 * pl.col("depth_signal")
    )
    
    # Penalty modifiers
    cq_mod = career_quality_raw
    cq_mod = pl.when(pl.col("consulting_flag") == True).then(cq_mod * 0.4).otherwise(cq_mod)
    
    # Handle optional flags safely using getattr to check existence, or just pl.col if they exist
    if "research_only" in df.columns:
        cq_mod = pl.when(pl.col("research_only") == True).then(cq_mod * 0.5).otherwise(cq_mod)
    if "wrong_domain" in df.columns:
        cq_mod = pl.when(pl.col("wrong_domain") == True).then(cq_mod * 0.3).otherwise(cq_mod)
        
    career_quality_score = (cq_mod / 0.15).clip(upper_bound=1.0)
    
    # Product Builder
    product_builder_score = pl.col("product_builder_score")
    
    # Final composite
    core_score = (
        0.55 * must_have_score +
        0.10 * nice_to_have_score +
        0.15 * career_quality_score +
        0.20 * product_builder_score
    ) * 100.0
    
    return df.with_columns(core_score.alias("core_score"))
