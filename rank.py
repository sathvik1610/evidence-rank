"""
rank.py — Stage B Runtime Entry Point

Lightning-fast runtime orchestrator.
Executes Phase 4 (Core Scoring), Phase 5 (Behavioral), and Phase 6 (Explanations).
Relies entirely on precomputed artifacts to stay well under the 5-minute constraint.

No imports of torch, faiss, or FlagEmbedding.
"""

import sys
import argparse
import polars as pl
import pandas as pd
from datetime import date
import time
import json
import os

import constants
from src.scorer import score_candidates_vectorized
from src.reranker import merge_cross_encoder_scores
from src.behavioral import (
    compute_final_score,
    assign_ranks,
    ce_core_delta,
    has_location_risk,
    has_notice_risk,
    has_top100_must_have_gap,
    hard_disqualification_reason,
    is_hard_disqualified,
    missing_must_have_buckets,
)
from src.explainer import generate_reasoning, get_largest_concern
from src.weights import W
from src.runtime_calibration import load_candidate_records, calibrate_candidate


def load_candidate_ids(path: str) -> set[str]:
    """
    Read only candidate_id values from a JSON array or JSONL file.
    This guards against stale artifacts being ranked for the wrong input file.
    """
    ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            data = json.load(f)
            for item in data:
                if isinstance(item, dict) and item.get("candidate_id"):
                    ids.add(str(item["candidate_id"]))
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict) and item.get("candidate_id"):
                    ids.add(str(item["candidate_id"]))
    return ids


def visible_submission_score(cand: dict, raw_min: float, raw_max: float) -> float:
    """Map the real internal score onto a readable fixed 1-100 submission scale."""
    raw_score = float(cand.get("true_unclamped_final_score", cand.get("final_score", 0.0)) or 0.0)
    calibrated = 12.0 + 0.79 * raw_score
    return round(max(1.0, min(96.0, calibrated)), 3)


def _as_float(cand: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(cand.get(key, default) or 0.0)
    except (TypeError, ValueError):
        return default


def _flag(cand: dict, key: str) -> bool:
    return bool(cand.get(key, False))


def score_bonus_total(cand: dict) -> float:
    return round(
        _as_float(cand, "runtime_same_project_full_system_bonus_value")
        + _as_float(cand, "runtime_recruiter_candidate_workflow_bonus_value")
        + _as_float(cand, "runtime_passive_responsive_exact_fit_bonus")
        + _as_float(cand, "runtime_split_career_bonus_value"),
        6,
    )


def bonus_reasons(cand: dict) -> str:
    reasons = []
    if _as_float(cand, "runtime_same_project_full_system_bonus_value") > 0:
        reasons.append(f"same_project_{cand.get('runtime_same_project_bonus_type', 'none')}")
    if _flag(cand, "runtime_recruiter_workflow_bonus_applied"):
        reasons.append("recruiter_workflow")
    if _flag(cand, "runtime_passive_responsive_exact_fit_bonus_applied"):
        reasons.append("passive_responsive_exact_fit")
    if _as_float(cand, "runtime_split_career_bonus_value") > 0:
        reasons.append("split_career_core_coverage")
    if _as_float(cand, "ninety_day_alignment") > 0:
        reasons.append("ninety_day_alignment")
    return ";".join(reasons) or "none"


def penalty_multiplier_total(cand: dict) -> float:
    return round(
        _as_float(cand, "runtime_evidence_gating_multiplier", 1.0)
        * _as_float(cand, "runtime_partial_system_with_logistics_risk_multiplier", 1.0)
        * _as_float(cand, "runtime_partial_system_low_ce_low_response_multiplier", 1.0),
        6,
    )


def penalty_reasons(cand: dict) -> str:
    reasons = []
    if _flag(cand, "runtime_partial_system_with_logistics_risk_penalty_applied"):
        reasons.append("partial_system_logistics_risk")
    if _flag(cand, "runtime_partial_system_low_ce_low_response_penalty_applied"):
        reasons.append("partial_system_low_ce_low_response")
    if _flag(cand, "runtime_adjacent_internal_only_flag"):
        reasons.append("adjacent_internal_only")
    if has_location_risk(cand):
        reasons.append("location_risk")
    if has_notice_risk(cand):
        reasons.append("notice_risk")
    if _as_float(cand, "beh_recruiter_response_rate", 1.0) < 0.50:
        reasons.append("low_response")
    if cand.get("beh_open_to_work") is False:
        reasons.append("not_open_to_work")
    return ";".join(reasons) or "none"


def suspected_gap_cause(upper: dict, lower: dict) -> str:
    causes = []
    if _flag(upper, "runtime_same_project_full_system_bonus_applied") != _flag(lower, "runtime_same_project_full_system_bonus_applied"):
        causes.append("full-system bonus difference")
    if _flag(upper, "runtime_recruiter_workflow_bonus_applied") != _flag(lower, "runtime_recruiter_workflow_bonus_applied"):
        causes.append("recruiter workflow bonus difference")
    if _flag(upper, "runtime_partial_system_with_logistics_risk_penalty_applied") != _flag(lower, "runtime_partial_system_with_logistics_risk_penalty_applied"):
        causes.append("partial-system logistics penalty difference")
    if _flag(upper, "runtime_partial_system_low_ce_low_response_penalty_applied") != _flag(lower, "runtime_partial_system_low_ce_low_response_penalty_applied"):
        causes.append("partial-system low-CE/low-response penalty difference")
    if abs(_as_float(upper, "ce_score") - _as_float(lower, "ce_score")) >= 15.0:
        causes.append("CE semantic-match gap")
    if abs(_as_float(upper, "core_score") - _as_float(lower, "core_score")) >= 10.0:
        causes.append("core technical-score gap")
    if penalty_reasons(upper) != penalty_reasons(lower):
        causes.append("logistics/availability penalty difference")
    return "; ".join(causes) or "true technical gap or unclear"


def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates JSONL")
    parser.add_argument("--out", type=str, required=True, help="Output CSV path")
    parser.add_argument("--debug-top-n", type=int, default=150, help="How many ranked rows to write to artifacts/ranking_debug.csv")
    args = parser.parse_args()

    print("--- Stage B: Ranking Execution ---")
    
    # 1. Load Precomputed Features (Phase 1c output)
    # The offline pipeline already filtered this to the Top 5000 from Phase 1d (RRF)
    if not os.path.exists(constants.CANDIDATE_FEATURES_PARQUET):
        print(f"Error: Precomputed features not found at {constants.CANDIDATE_FEATURES_PARQUET}.")
        print("Run preprocess.py first.")
        sys.exit(1)
        
    candidate_ids = load_candidate_ids(args.candidates)
    if not candidate_ids:
        print(f"Error: no candidate_id values found in {args.candidates}.")
        sys.exit(1)

    df = pl.read_parquet(constants.CANDIDATE_FEATURES_PARQUET)
    before_filter = len(df)
    df = df.filter(pl.col("candidate_id").is_in(candidate_ids))
    print(f"Loaded {len(df)} candidates from precomputed feature pool after filtering {before_filter} artifact rows to the input file.")
    if len(df) == 0:
        print("Error: input candidate IDs do not overlap with precomputed features. Re-run preprocess.py for this candidate file.")
        sys.exit(1)

    # Phase 2: if RRF retrieval scores are available, honor the runtime retrieval
    # cutoff before core scoring. Sample --skip-embed runs do not create this file,
    # so they correctly exercise all sample candidates.
    if os.path.exists(constants.RETRIEVAL_SCORES_PARQUET):
        retrieval_df = pl.read_parquet(constants.RETRIEVAL_SCORES_PARQUET)
        runtime_top_k = int(W.get("retrieval.runtime_top_k", constants.RUNTIME_RETRIEVAL_TOPK))
        df = (
            df.join(retrieval_df, on="candidate_id", how="left")
              .with_columns(pl.col("rrf_score").fill_null(0.0))
              .sort("rrf_score", descending=True)
              .head(runtime_top_k)
        )
        print(f"Applied Phase 2 RRF cutoff: top {len(df)} by retrieval score.")

    # 2. Phase 4a: Core Scoring (Vectorized)
    df = score_candidates_vectorized(df)
    
    # 3. Phase 4b: Cross-Encoder Merge
    # Merge before the Phase 5 slice so CE can rescue semantically strong
    # candidates that are slightly underweighted by handcrafted regex features.
    df = merge_cross_encoder_scores(df)

    # 4. Slice to a wide Phase 5 pool by blended Phase 4 score. The pool is
    # intentionally wider than 500 because strict JD hard gates may remove
    # otherwise high-scoring but ineligible candidates before final ranking.
    phase5_pool_size = int(W.get("ranking.phase5_candidate_pool", 1000))
    df = df.sort("final_phase4_score", descending=True).head(phase5_pool_size)
    print(f"Sliced to top {len(df)} candidates by blended Phase 4 score for Phase 5 gate/backfill.")
    
    # We now move to python dictionaries for Phase 5 & 6 row-level operations
    # since these are complex heuristic formulas and the Phase 5 pool is bounded.
    candidates = df.to_dicts()

    # Runtime JD calibration: cheap profile-text checks for current/recent
    # full-plan evidence. This avoids expensive preprocessing while allowing
    # the final ranker to respect plain-language JD intent in the full profiles.
    profile_records = load_candidate_records(
        args.candidates,
        {str(c.get("candidate_id", "")) for c in candidates},
    )
    candidates = [
        calibrate_candidate(c, profile_records.get(str(c.get("candidate_id", ""))))
        for c in candidates
    ]

    hard_disqualified = []
    for cand in candidates:
        reason = hard_disqualification_reason(cand)
        if reason:
            cand["hard_disqualified"] = True
            cand["hard_disqualification_reason"] = reason
            cand["final_score"] = 0.0
            hard_disqualified.append(cand)
        else:
            cand["hard_disqualified"] = False
            cand["hard_disqualification_reason"] = ""
    # Reference date for ghost detection and reachability
    # We load run metadata to get the actual reference time, or default to mid-2026
    ref_date = date(2026, 6, 1)
    if os.path.exists(constants.RUN_METADATA_JSON):
        with open(constants.RUN_METADATA_JSON, "r") as f:
            meta = json.load(f)
            # GAP 3 FIX: use reference_date (= max last_active_date from corpus)
            # Fall back to run_time for backwards compatibility with old artifacts
            date_str = meta.get("reference_date") or (
                meta.get("run_time", "").split("T")[0] if meta.get("run_time") else None
            )
            if date_str:
                try:
                    ref_date = date.fromisoformat(date_str)
                except ValueError:
                    pass

    print("Running Phase 5 (Behavioral Modifiers)...")
    for cand in candidates:
        raw = compute_final_score(cand, ref_date)
        # Preserve the internal unclamped score which is computed inside compute_final_score
        true_score = cand.get("true_unclamped_final_score", raw)
        cand["final_score"] = min(true_score, 100.0)

    # Phase 5b: Rank Assignment
    candidates = assign_ranks(candidates)

    # Official submissions must contain exactly 100 rows. For sample/sandbox
    # inputs with fewer than 100 candidates, output the available ranked rows.
    if len(candidate_ids) >= 100:
        top_100 = candidates[:100]
    else:
        top_100 = candidates
    
    debug_top_n = max(100, int(args.debug_top_n))
    debug_candidates = candidates[: min(debug_top_n, len(candidates))]
    submission_raw_scores = [
        float(c.get("true_unclamped_final_score", c.get("final_score", 0.0)) or 0.0)
        for c in top_100
    ]
    score_raw_min = min(submission_raw_scores) if submission_raw_scores else 0.0
    score_raw_max = max(submission_raw_scores) if submission_raw_scores else 0.0

    print(f"Running Phase 6 (Reason Generation) for Top {len(top_100)} submission rows and Top {len(debug_candidates)} debug rows...")
    debug_records = []
    output_records = []
    reason_cache = {}
    
    for cand in top_100:
        reason = generate_reasoning(cand)
        reason_cache[cand["candidate_id"]] = reason
        visible_score = visible_submission_score(cand, score_raw_min, score_raw_max)
        
        output_records.append({
            "candidate_id": cand["candidate_id"],
            "rank": cand["rank"],
            "score": visible_score,
            "reasoning": reason
        })

    for cand in debug_candidates:
        reason = reason_cache.get(cand["candidate_id"])
        if reason is None:
            reason = generate_reasoning(cand)

        missing = missing_must_have_buckets(cand)
        # Debug trace
        visible_score = visible_submission_score(cand, score_raw_min, score_raw_max)
        debug_records.append({
            "candidate_id": cand["candidate_id"],
            "rank": cand["rank"],
            "score": visible_score,
            "final_score": round(cand.get("final_score", 0.0), 6),
            "core_score": round(cand.get("core_score", 0.0), 2),
            "ce_score": round(cand.get("ce_score", 0.0), 2),
            "ce_core_delta": round(ce_core_delta(cand), 2),
            "final_phase4_score": round(cand.get("final_phase4_score", 0.0), 2),
            "retrieval_search": cand.get("retrieval_search", 0.0),
            "vector_db_hybrid": cand.get("vector_db_hybrid", 0.0),
            "ltr_reranking": cand.get("ltr_reranking", 0.0),
            "eval_framework": cand.get("eval_framework", 0.0),
            "python_coding": cand.get("python_coding", 0.0),
            "sys_experience_score": cand.get("sys_experience_score", 0.0),
            "career_ir_density": cand.get("career_ir_density", 0.0),
            "career_eval_density": cand.get("career_eval_density", 0.0),
            "adjacent_career_ratio": cand.get("adjacent_career_ratio", 0.0),
            "product_builder_score": cand.get("product_builder_score", 0.0),
            "target_skill_duration_contradiction": cand.get("target_skill_duration_contradiction", 0),
            "max_target_skill_overclaim_months": cand.get("max_target_skill_overclaim_months", 0.0),
            "repeated_long_descriptions": cand.get("repeated_long_descriptions", 0),
            "missing_must_have_buckets": ";".join(missing),
            "must_have_gap_count": len(missing),
            "top100_must_have_exclusion": has_top100_must_have_gap(cand),
            "notice_risk": has_notice_risk(cand),
            "location_risk": has_location_risk(cand),
            "hard_disqualified": is_hard_disqualified(cand),
            "hard_disqualification_reason": hard_disqualification_reason(cand),
            "impossible_flag": cand.get("impossible_flag", False),
            "suspicious_flag": cand.get("suspicious_flag", False),
            "is_ghost": cand.get("is_ghost", False),
            "keyword_stuffer_flag": cand.get("keyword_stuffer_flag", False),
            "langchain_only_flag": cand.get("langchain_only_flag", False),
            "research_only": cand.get("research_only", False),
            "consulting_only": cand.get("consulting_only", False),
            "wrong_domain": cand.get("wrong_domain", False),
            "title_chaser_flag": cand.get("title_chaser_flag", False),
            "code_stopped": cand.get("code_stopped", False),
            "beh_github_activity_score": cand.get("beh_github_activity_score", -1),
            "runtime_full_plan_signal": cand.get("runtime_full_plan_signal", 0.0),
            "runtime_current_retrieval_signal": cand.get("runtime_current_retrieval_signal", 0.0),
            "runtime_current_vector_signal": cand.get("runtime_current_vector_signal", 0.0),
            "runtime_current_ranking_signal": cand.get("runtime_current_ranking_signal", 0.0),
            "runtime_current_eval_signal": cand.get("runtime_current_eval_signal", 0.0),
            "runtime_current_ship_signal": cand.get("runtime_current_ship_signal", 0.0),
            "runtime_career_retrieval_signal": cand.get("runtime_career_retrieval_signal", 0.0),
            "runtime_career_vector_signal": cand.get("runtime_career_vector_signal", 0.0),
            "runtime_career_ranking_signal": cand.get("runtime_career_ranking_signal", 0.0),
            "runtime_career_eval_signal": cand.get("runtime_career_eval_signal", 0.0),
            "runtime_career_eval_adjacent_signal": cand.get("runtime_career_eval_adjacent_signal", 0.0),
            "runtime_production_retrieval_signal": cand.get("runtime_production_retrieval_signal", 0.0),
            "runtime_production_vector_signal": cand.get("runtime_production_vector_signal", 0.0),
            "runtime_vector_skill_signal": cand.get("runtime_vector_skill_signal", 0.0),
            "runtime_concrete_vector_tool_signal": cand.get("runtime_concrete_vector_tool_signal", 0.0),
            "runtime_corroborated_vector_signal": cand.get("runtime_corroborated_vector_signal", 0.0),
            "runtime_career_python_signal": cand.get("runtime_career_python_signal", 0.0),
            "runtime_current_services_signal": cand.get("runtime_current_services_signal", 0.0),
            # Part 9: raw vs clamped score
            "true_unclamped_final_score": round(cand.get("true_unclamped_final_score", cand.get("raw_final_score", cand["final_score"])), 6),
            "raw_final_score": round(cand.get("raw_final_score", cand["final_score"]), 6),
            # Fix 2: Partial system logistics risk penalty
            "partial_system_with_logistics_risk_penalty_applied": cand.get("runtime_partial_system_with_logistics_risk_penalty_applied", False),
            "partial_system_with_logistics_risk_multiplier": cand.get("runtime_partial_system_with_logistics_risk_multiplier", 1.0),
            "partial_system_with_logistics_risk_reason": cand.get("runtime_partial_system_with_logistics_risk_reason", ""),
            # Part 4: Same-project full-system bonus
            "same_project_full_system_bonus_applied": cand.get("runtime_same_project_full_system_bonus_applied", False),
            "same_project_partial_system_bonus_applied": cand.get("runtime_same_project_partial_system_bonus_applied", False),
            "runtime_same_project_bonus_type": cand.get("runtime_same_project_bonus_type", "none"),
            "runtime_split_career_bonus_value": cand.get("runtime_split_career_bonus_value", 0.0),
            "runtime_evidence_gating_multiplier": cand.get("runtime_evidence_gating_multiplier", 1.0),
            "runtime_adjacent_internal_only_flag": cand.get("runtime_adjacent_internal_only_flag", False),
            "same_project_full_system_bonus": cand.get("runtime_same_project_full_system_bonus_value", 0.0),
            "same_project_full_system_evidence_groups": cand.get("runtime_same_project_full_system_evidence_groups", ""),
            "same_project_full_system_evidence_snippet": cand.get("runtime_same_project_full_system_evidence_snippet", ""),
            # Part 5: Recruiter/Candidate workflow bonus
            "recruiter_candidate_workflow_bonus_applied": cand.get("runtime_recruiter_workflow_bonus_applied", False),
            "recruiter_candidate_workflow_bonus": cand.get("runtime_recruiter_candidate_workflow_bonus_value", 0.0),
            "recruiter_candidate_workflow_evidence_snippet": cand.get("runtime_recruiter_workflow_evidence_snippet", ""),
            "passive_responsive_exact_fit_bonus_applied": cand.get("runtime_passive_responsive_exact_fit_bonus_applied", False),
            "passive_responsive_exact_fit_bonus": cand.get("runtime_passive_responsive_exact_fit_bonus", 0.0),
            "passive_responsive_exact_fit_reason": cand.get("runtime_passive_responsive_exact_fit_reason", ""),
            "partial_system_low_ce_low_response_penalty_applied": cand.get("runtime_partial_system_low_ce_low_response_penalty_applied", False),
            "partial_system_low_ce_low_response_multiplier": cand.get("runtime_partial_system_low_ce_low_response_multiplier", 1.0),
            "partial_system_low_ce_low_response_reason": cand.get("runtime_partial_system_low_ce_low_response_reason", ""),
            # Part 3: Evidence bonus levels
            "retrieval_bonus_level": cand.get("runtime_retrieval_bonus_level", 0),
            "vector_bonus_level": cand.get("runtime_vector_bonus_level", 0),
            "ltr_bonus_level": cand.get("runtime_ltr_bonus_level", 0),
            "eval_bonus_level": cand.get("runtime_eval_bonus_level", 0),
            "product_bonus_level": cand.get("runtime_product_bonus_level", 0),
            # Part 6: Adjacent/internal-only penalty
            "adjacent_internal_only_penalty_applied": cand.get("runtime_adjacent_internal_only_penalty_applied", False),
            "adjacent_internal_only_penalty_multiplier": cand.get("runtime_adjacent_internal_only_penalty_multiplier", 1.0),
            "adjacent_internal_only_reason": cand.get("runtime_adjacent_internal_only_reason", ""),
            # Part 2: Skill-duration bypass
            "skill_duration_penalty_bypassed": cand.get("runtime_skill_duration_penalty_bypassed", False),
            "skill_duration_penalty_bypass_reason": cand.get("runtime_skill_duration_penalty_bypass_reason", ""),
            # Part 7: Elite logistics cap
            "elite_fit_logistics_cap_applied": cand.get("runtime_elite_fit_logistics_cap_applied", False),
            "raw_logistics_multiplier_before_cap": cand.get("runtime_raw_logistics_multiplier_before_cap", 0.0),
            "final_logistics_multiplier_after_cap": cand.get("runtime_final_logistics_multiplier_after_cap", 0.0),
            "reasoning": reason,
            "concern": get_largest_concern(cand)
        })
        
    # Save Outputs
    out_df = pd.DataFrame(output_records)
    out_df.to_csv(args.out, index=False)
    
    # Save Debug (Offline only)
    debug_df = pd.DataFrame(debug_records)
    debug_df.to_csv("artifacts/ranking_debug.csv", index=False)

    hard_disqualified_records = []
    for cand in hard_disqualified:
        missing = missing_must_have_buckets(cand)
        hard_disqualified_records.append({
            "candidate_id": cand.get("candidate_id", ""),
            "phase4_rank_candidate": cand.get("rank", ""),
            "final_phase4_score": round(cand.get("final_phase4_score", 0.0), 2),
            "hard_disqualification_reason": cand.get("hard_disqualification_reason", ""),
            "missing_must_have_buckets": ";".join(missing),
            "must_have_gap_count": len(missing),
            "runtime_current_retrieval_signal": cand.get("runtime_current_retrieval_signal", 0.0),
            "runtime_current_vector_signal": cand.get("runtime_current_vector_signal", 0.0),
            "runtime_current_ranking_signal": cand.get("runtime_current_ranking_signal", 0.0),
            "runtime_current_eval_signal": cand.get("runtime_current_eval_signal", 0.0),
            "runtime_current_ship_signal": cand.get("runtime_current_ship_signal", 0.0),
            "runtime_career_retrieval_signal": cand.get("runtime_career_retrieval_signal", 0.0),
            "runtime_career_vector_signal": cand.get("runtime_career_vector_signal", 0.0),
            "runtime_career_ranking_signal": cand.get("runtime_career_ranking_signal", 0.0),
            "runtime_career_eval_signal": cand.get("runtime_career_eval_signal", 0.0),
            "runtime_career_eval_adjacent_signal": cand.get("runtime_career_eval_adjacent_signal", 0.0),
            "runtime_production_retrieval_signal": cand.get("runtime_production_retrieval_signal", 0.0),
            "runtime_production_vector_signal": cand.get("runtime_production_vector_signal", 0.0),
            "runtime_vector_skill_signal": cand.get("runtime_vector_skill_signal", 0.0),
            "runtime_concrete_vector_tool_signal": cand.get("runtime_concrete_vector_tool_signal", 0.0),
            "runtime_corroborated_vector_signal": cand.get("runtime_corroborated_vector_signal", 0.0),
            "runtime_career_python_signal": cand.get("runtime_career_python_signal", 0.0),
            "notice_risk": has_notice_risk(cand),
            "location_risk": has_location_risk(cand),
            "impossible_flag": cand.get("impossible_flag", False),
            "suspicious_flag": cand.get("suspicious_flag", False),
            "is_ghost": cand.get("is_ghost", False),
            "keyword_stuffer_flag": cand.get("keyword_stuffer_flag", False),
            "langchain_only_flag": cand.get("langchain_only_flag", False),
            "research_only": cand.get("research_only", False),
            "consulting_only": cand.get("consulting_only", False),
            "wrong_domain": cand.get("wrong_domain", False),
            "title_chaser_flag": cand.get("title_chaser_flag", False),
            "code_stopped": cand.get("code_stopped", False),
        })
    pd.DataFrame(hard_disqualified_records).to_csv("artifacts/hard_disqualified_debug.csv", index=False)

    score_gap_records = []
    for prev, curr in zip(debug_candidates, debug_candidates[1:]):
        score_gap_records.append({
            "boundary": f"{prev['rank']}->{curr['rank']}",
            "upper_candidate_id": prev["candidate_id"],
            "lower_candidate_id": curr["candidate_id"],
            "upper_score": prev["final_score"],
            "lower_score": curr["final_score"],
            "gap": round(float(prev["final_score"]) - float(curr["final_score"]), 6),
        })
    pd.DataFrame(score_gap_records).to_csv("artifacts/rank_score_gaps.csv", index=False)

    detailed_gap_records = []
    large_gap_warning_records = []
    for upper, lower in zip(top_100, top_100[1:]):
        upper_true = _as_float(upper, "true_unclamped_final_score", _as_float(upper, "final_score"))
        lower_true = _as_float(lower, "true_unclamped_final_score", _as_float(lower, "final_score"))
        gap = round(upper_true - lower_true, 6)
        cause = suspected_gap_cause(upper, lower)
        detailed_gap_records.append({
            "upper_rank": upper.get("rank", ""),
            "lower_rank": lower.get("rank", ""),
            "upper_candidate_id": upper.get("candidate_id", ""),
            "lower_candidate_id": lower.get("candidate_id", ""),
            "upper_true_unclamped_final_score": round(upper_true, 6),
            "lower_true_unclamped_final_score": round(lower_true, 6),
            "score_gap": gap,
            "upper_bonus_total": score_bonus_total(upper),
            "lower_bonus_total": score_bonus_total(lower),
            "upper_penalty_multiplier_total": penalty_multiplier_total(upper),
            "lower_penalty_multiplier_total": penalty_multiplier_total(lower),
            "upper_ce_score": round(_as_float(upper, "ce_score"), 2),
            "lower_ce_score": round(_as_float(lower, "ce_score"), 2),
            "upper_core_score": round(_as_float(upper, "core_score"), 2),
            "lower_core_score": round(_as_float(lower, "core_score"), 2),
            "upper_bonus_reasons": bonus_reasons(upper),
            "lower_bonus_reasons": bonus_reasons(lower),
            "upper_penalty_reasons": penalty_reasons(upper),
            "lower_penalty_reasons": penalty_reasons(lower),
            "suspected_gap_cause": cause,
        })

        if int(upper.get("rank", 9999)) <= 40 and gap > 4.0:
            expected = (
                _flag(upper, "runtime_same_project_full_system_bonus_applied")
                and _flag(upper, "runtime_recruiter_workflow_bonus_applied")
                and _as_float(upper, "ce_score") >= 80.0
                and (
                    not _flag(lower, "runtime_recruiter_workflow_bonus_applied")
                    or _as_float(lower, "ce_score") < 50.0
                    or lower.get("runtime_same_project_bonus_type") in ("partial", "none")
                )
            )
            large_gap_warning_records.append({
                "upper_rank": upper.get("rank", ""),
                "lower_rank": lower.get("rank", ""),
                "upper_candidate_id": upper.get("candidate_id", ""),
                "lower_candidate_id": lower.get("candidate_id", ""),
                "gap": gap,
                "likely_reason": cause,
                "is_expected_gap": expected,
            })

    pd.DataFrame(detailed_gap_records).to_csv("artifacts/score_gap_diagnostics.csv", index=False)
    pd.DataFrame(large_gap_warning_records).to_csv("artifacts/large_gap_warnings.csv", index=False)

    yoe_records = []
    for label, subset in (("top10", candidates[:10]), ("top25", candidates[:25]), ("top100", top_100)):
        yoe_values = [
            float(c.get("profile_years_of_experience", 0.0))
            for c in subset
            if c.get("profile_years_of_experience", None) is not None
        ]
        if not yoe_values:
            continue
        yoe_series = pd.Series(yoe_values)
        yoe_records.append({
            "band": label,
            "count": len(yoe_values),
            "min": round(float(yoe_series.min()), 2),
            "median": round(float(yoe_series.median()), 2),
            "mean": round(float(yoe_series.mean()), 2),
            "max": round(float(yoe_series.max()), 2),
            "below_5": int((yoe_series < 5.0).sum()),
            "above_9": int((yoe_series > 9.0).sum()),
        })
    pd.DataFrame(yoe_records).to_csv("artifacts/yoe_distribution.csv", index=False)
    
    # Generate Markdown Statistics
    stats_md_path = "docs/ranking_statistics.md"
    try:
        subsets = [
            ("Top 10", candidates[:10]),
            ("Top 50", candidates[:50]),
            ("Top 100", top_100),
            ("Top 1000", candidates[:1000])
        ]
        
        md_lines = ["# Dynamic Ranking Statistics\n\n"]
        
        # 1. Score Distributions
        md_lines.append("## Score Distributions\n")
        md_lines.append("| Metric | Top 10 | Top 50 | Top 100 | Top 1000 |\n")
        md_lines.append("|---|---|---|---|---|\n")
        
        metrics = [
            ("Final Score", "final_score"),
            ("Core Score", "core_score"),
            ("Cross-Encoder Score", "ce_score"),
            ("Phase 4 Score", "final_phase4_score"),
            ("Years of Experience", "profile_years_of_experience")
        ]
        
        for label, key in metrics:
            row_mean = [f"**{label} (Mean)**"]
            row_med = [f"**{label} (Median)**"]
            row_min = [f"**{label} (Min)**"]
            row_max = [f"**{label} (Max)**"]
            for _, sub_cands in subsets:
                vals = pd.Series([float(c.get(key, 0.0) or 0.0) for c in sub_cands if c.get(key) is not None])
                if not vals.empty:
                    row_mean.append(f"{vals.mean():.2f}")
                    row_med.append(f"{vals.median():.2f}")
                    row_min.append(f"{vals.min():.2f}")
                    row_max.append(f"{vals.max():.2f}")
                else:
                    row_mean.append("N/A")
                    row_med.append("N/A")
                    row_min.append("N/A")
                    row_max.append("N/A")
            md_lines.append("| " + " | ".join(row_mean) + " |\n")
            md_lines.append("| " + " | ".join(row_med) + " |\n")
            md_lines.append("| " + " | ".join(row_min) + " |\n")
            md_lines.append("| " + " | ".join(row_max) + " |\n")
            
        md_lines.append("\n## Behavioral & Risk Flags (Candidate Count)\n")
        md_lines.append("| Flag | Top 10 | Top 50 | Top 100 | Top 1000 |\n")
        md_lines.append("|---|---|---|---|---|\n")
        
        flags = [
            ("Location Risk", lambda c: has_location_risk(c)),
            ("Notice Risk", lambda c: has_notice_risk(c)),
            ("Skill Duration Overclaims", lambda c: c.get("target_skill_duration_contradiction", 0) > 0),
            ("High Adj. Career Ratio (>0.5)", lambda c: float(c.get("adjacent_career_ratio", 0.0) or 0.0) > 0.5),
            ("Keyword Stuffer Flag", lambda c: c.get("keyword_stuffer_flag", False)),
            ("LangChain Only Flag", lambda c: c.get("langchain_only_flag", False)),
            ("Consulting Only", lambda c: c.get("consulting_only", False)),
            ("Research Only", lambda c: c.get("research_only", False)),
            ("Suspicious Flag", lambda c: c.get("suspicious_flag", False))
        ]
        
        for label, func in flags:
            row = [f"**{label}**"]
            for _, sub_cands in subsets:
                count = sum(1 for c in sub_cands if func(c))
                row.append(str(count))
            md_lines.append("| " + " | ".join(row) + " |\n")
            
        with open(stats_md_path, "w", encoding="utf-8") as f:
            f.writelines(md_lines)
            
        print(f"Ranking statistics written to {stats_md_path}")
    except Exception as e:
        print(f"Warning: Failed to generate statistics md: {e}")
        
    elapsed = time.time() - start_time
    print(f"Successfully wrote {len(out_df)} candidates to {args.out}")
    print(f"Debug trace written to artifacts/ranking_debug.csv")
    print("Hard-disqualification trace written to artifacts/hard_disqualified_debug.csv")
    print(
        "Diagnostics written to artifacts/rank_score_gaps.csv, "
        "artifacts/score_gap_diagnostics.csv, artifacts/large_gap_warnings.csv, "
        "and artifacts/yoe_distribution.csv"
    )
    print(f"Large Top-40 gap warnings: {len(large_gap_warning_records)}")
    print(f"Runtime: {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
