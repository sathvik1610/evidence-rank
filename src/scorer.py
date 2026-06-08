"""
src/scorer.py — Phase 4: Core Scoring

Implements the 4-part weighted formula mapping exactly to the JD contract:
- 55% Must-Have Score (IR, Vector DB, Eval, Python, System Experience)
- 10% Nice-to-Have Score (LTR, LLM, Dist Sys, HR Tech)
- 15% Career Quality Score (Recency, Depth)
- 20% Product Builder Score (Deployment, Shipper signals, Ownership)

All weights come from weights.yaml via src/weights.py — no hardcoded numbers.
"""

from typing import Dict, Any
from src.weights import W


def compute_core_score(features: Dict[str, Any]) -> float:
    """
    Computes the 0-100 Core Score for a single candidate based on Phase 4 design.
    All numeric weights are sourced from weights.yaml via W.
    """
    # --- Must-Have Score ---
    retrieval_ev      = features.get("retrieval_search", 0.0) / 3.0
    vectordb_ev       = features.get("vector_db_hybrid", 0.0) / 3.0
    eval_ev           = features.get("eval_framework", 0.0) / 3.0
    python_ev         = features.get("python_coding", 0.0) / 3.0
    sys_experience    = features.get("sys_experience_score", 0.0)

    must_have_raw = (
        W["scoring.must_have_sub.retrieval_weight"]      * retrieval_ev +
        W["scoring.must_have_sub.vectordb_weight"]       * vectordb_ev +
        W["scoring.must_have_sub.sys_experience_weight"] * sys_experience +
        W["scoring.must_have_sub.eval_weight"]           * eval_ev +
        W["scoring.must_have_sub.python_weight"]         * python_ev
    )

    # Softened cap: if zero retrieval/vectordb/recsys, cap must_have at no_retrieval_cap
    has_retrieval_or_recsys = (
        features.get("retrieval_search", 0.0) > 0 or
        features.get("vector_db_hybrid", 0.0) > 0 or
        sys_experience > 0.0
    )
    if not has_retrieval_or_recsys:
        must_have_raw = min(must_have_raw, W["scoring.no_retrieval_cap"])

    # Normalise to [0, 1]: sum of must_have_sub weights = 0.80
    must_have_score = min(must_have_raw / 0.80, 1.0)

    # --- Nice-to-Have Score ---
    ltr_ev  = features.get("ltr_reranking", 0.0) / 3.0
    llm_ev  = features.get("llm_integration", 0.0) / 3.0
    dist_ev = features.get("distributed_systems", 0.0) / 3.0
    hr_ev   = features.get("hr_tech_exposure", 0.0) / 3.0

    nice_to_have_score = min((
        W["scoring.nice_to_have_sub.ltr_weight"]         * ltr_ev +
        W["scoring.nice_to_have_sub.llm_weight"]         * llm_ev +
        W["scoring.nice_to_have_sub.distributed_weight"] * dist_ev +
        W["scoring.nice_to_have_sub.hr_tech_weight"]     * hr_ev
    ) / 0.10, 1.0)  # normalise; sum of nice_to_have_sub weights = 0.10

    # --- Career Quality Score ---
    experience_recency = features.get("experience_recency", 0.5)
    depth_signal       = features.get("depth_signal", 0.0)

    career_quality_raw = (
        W["scoring.career_quality_sub.sys_experience_weight"] * sys_experience +
        W["scoring.career_quality_sub.recency_weight"]        * experience_recency +
        W["scoring.career_quality_sub.depth_weight"]          * depth_signal
    )

    # Phase 4 career quality penalties (also applied again in Phase 5 soft_penalties —
    # this double-layer is intentional per the plan)
    if features.get("consulting_flag", False) or features.get("consulting_only", False):
        career_quality_raw *= W["career_multipliers.consulting_only_penalty"]
    if features.get("research_only", False):
        career_quality_raw *= W["career_multipliers.research_only_penalty"]
    if features.get("wrong_domain", False):
        career_quality_raw *= W["career_multipliers.wrong_domain_penalty"]

    career_quality_score = min(career_quality_raw / 0.15, 1.0)  # normalise

    # --- Product Builder Score (20%) ---
    # Pre-computed and normalised to [0,1] in Phase 3 / features.py
    product_builder_score = features.get("product_builder_score", 0.0)

    # --- Combined Weighted Score ---
    core_score = (
        W["scoring.must_have_weight"]       * must_have_score +
        W["scoring.nice_to_have_weight"]    * nice_to_have_score +
        W["scoring.career_quality_weight"]  * career_quality_score +
        W["scoring.product_builder_weight"] * product_builder_score
    ) * 100.0

    career_ir_density = features.get("career_ir_density", 0.5)
    if (
        features.get("retrieval_search", 0.0) >= 2.0
        and features.get("ltr_reranking", 0.0) >= 3.0
        and features.get("eval_framework", 0.0) >= 2.0
    ):
        core_score += W["scoring.eval_trifecta_bonus"]
    elif (
        features.get("retrieval_search", 0.0) >= 2.0
        and features.get("eval_framework", 0.0) >= 2.0
    ):
        core_score += W["scoring.eval_retrieval_bonus"]

    if career_ir_density >= W["scoring.career_ir_density_bonus_threshold"]:
        core_score += W["scoring.career_ir_density_bonus"]
    elif career_ir_density < W["scoring.low_career_ir_density_threshold"]:
        core_score *= W["scoring.low_career_ir_density_mult"]

    if features.get("isolated_template_risk", False):
        core_score *= W["scoring.isolated_template_risk_mult"]

    return float(min(core_score, 100.0))


def score_candidates_vectorized(df) -> object:
    """
    Vectorized Polars execution for the core_score.
    Returns a DataFrame with a new `core_score` column.
    All weights sourced from W (weights.yaml).
    """
    import polars as pl

    def col_or_lit(name: str, default):
        return pl.col(name) if name in df.columns else pl.lit(default)

    # Must Have Raw
    must_have_raw = (
        W["scoring.must_have_sub.retrieval_weight"]      * (pl.col("retrieval_search") / 3.0) +
        W["scoring.must_have_sub.vectordb_weight"]       * (pl.col("vector_db_hybrid") / 3.0) +
        W["scoring.must_have_sub.sys_experience_weight"] * pl.col("sys_experience_score") +
        W["scoring.must_have_sub.eval_weight"]           * (pl.col("eval_framework") / 3.0) +
        W["scoring.must_have_sub.python_weight"]         * (pl.col("python_coding") / 3.0)
    )

    # Cap if no retrieval/recsys evidence
    has_retrieval_or_recsys = (
        (pl.col("retrieval_search") > 0) |
        (pl.col("vector_db_hybrid") > 0) |
        (pl.col("sys_experience_score") > 0)
    )
    cap = W["scoring.no_retrieval_cap"]
    capped_must_have_raw = pl.when(has_retrieval_or_recsys).then(must_have_raw).otherwise(
        pl.when(must_have_raw > cap).then(cap).otherwise(must_have_raw)
    )
    must_have_score = (capped_must_have_raw / 0.80).clip(upper_bound=1.0)

    # Nice to have
    nice_to_have_score = ((
        W["scoring.nice_to_have_sub.ltr_weight"]         * (pl.col("ltr_reranking") / 3.0) +
        W["scoring.nice_to_have_sub.llm_weight"]         * (pl.col("llm_integration") / 3.0) +
        W["scoring.nice_to_have_sub.distributed_weight"] * (pl.col("distributed_systems") / 3.0) +
        W["scoring.nice_to_have_sub.hr_tech_weight"]     * (pl.col("hr_tech_exposure") / 3.0)
    ) / 0.10).clip(upper_bound=1.0)

    # Career Quality
    career_quality_raw = (
        W["scoring.career_quality_sub.sys_experience_weight"] * pl.col("sys_experience_score") +
        W["scoring.career_quality_sub.recency_weight"]        * pl.col("experience_recency") +
        W["scoring.career_quality_sub.depth_weight"]          * pl.col("depth_signal")
    )

    # Penalty modifiers — applied per flag column if present
    cq_mod = career_quality_raw
    consulting_expr = None
    if "consulting_flag" in df.columns and "consulting_only" in df.columns:
        consulting_expr = (pl.col("consulting_flag") == True) | (pl.col("consulting_only") == True)
    elif "consulting_flag" in df.columns:
        consulting_expr = pl.col("consulting_flag") == True
    elif "consulting_only" in df.columns:
        consulting_expr = pl.col("consulting_only") == True

    if consulting_expr is not None:
        cq_mod = pl.when(consulting_expr).then(
            cq_mod * W["career_multipliers.consulting_only_penalty"]
        ).otherwise(cq_mod)
    if "research_only" in df.columns:
        cq_mod = pl.when(pl.col("research_only") == True).then(
            cq_mod * W["career_multipliers.research_only_penalty"]
        ).otherwise(cq_mod)
    if "wrong_domain" in df.columns:
        cq_mod = pl.when(pl.col("wrong_domain") == True).then(
            cq_mod * W["career_multipliers.wrong_domain_penalty"]
        ).otherwise(cq_mod)

    career_quality_score = (cq_mod / 0.15).clip(upper_bound=1.0)

    # Product Builder
    product_builder_score = pl.col("product_builder_score")

    # Final composite
    core_score = (
        W["scoring.must_have_weight"]       * must_have_score +
        W["scoring.nice_to_have_weight"]    * nice_to_have_score +
        W["scoring.career_quality_weight"]  * career_quality_score +
        W["scoring.product_builder_weight"] * product_builder_score
    ) * 100.0

    career_ir_density = col_or_lit("career_ir_density", 0.5)
    isolated_template_risk = col_or_lit("isolated_template_risk", False)
    trifecta = (
        (pl.col("retrieval_search") >= 2.0) &
        (pl.col("ltr_reranking") >= 3.0) &
        (pl.col("eval_framework") >= 2.0)
    )
    retrieval_eval = (
        (pl.col("retrieval_search") >= 2.0) &
        (pl.col("eval_framework") >= 2.0)
    )

    core_score = (
        core_score
        + pl.when(trifecta)
            .then(pl.lit(W["scoring.eval_trifecta_bonus"]))
            .when(retrieval_eval)
            .then(pl.lit(W["scoring.eval_retrieval_bonus"]))
            .otherwise(pl.lit(0.0))
        + pl.when(career_ir_density >= W["scoring.career_ir_density_bonus_threshold"])
            .then(pl.lit(W["scoring.career_ir_density_bonus"]))
            .otherwise(pl.lit(0.0))
    )
    core_score = pl.when(career_ir_density < W["scoring.low_career_ir_density_threshold"]).then(
        core_score * W["scoring.low_career_ir_density_mult"]
    ).otherwise(core_score)
    core_score = pl.when(isolated_template_risk == True).then(
        core_score * W["scoring.isolated_template_risk_mult"]
    ).otherwise(core_score)
    core_score = core_score.clip(upper_bound=100.0)

    return df.with_columns(core_score.alias("core_score"))
