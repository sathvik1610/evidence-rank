"""
src/behavioral.py — Phase 5: Behavioral Re-ranking + Penalization

Applies reachability modifiers, social proof boosts, and soft penalties.
All numeric thresholds and multipliers sourced from weights.yaml via W.

Applied late in the pipeline (top 500 candidates) to avoid dropping strong
passive candidates during early retrieval.
"""

from datetime import date
import constants
from src.jd_intelligence import build_feature_contract
from src.weights import W

JD_FEATURE_CONTRACT = build_feature_contract(constants.JD_CONTRACT_YAML)
PREFERRED_CITIES = set(JD_FEATURE_CONTRACT["location_bands"].get("preferred", []))
WELCOME_CITIES = set(JD_FEATURE_CONTRACT["location_bands"].get("welcome", []))
FLOOR_EXEMPT_MULTIPLIERS = set(JD_FEATURE_CONTRACT["floor_exempt_multiplier_ids"])
ADJACENT_DOMAIN_TITLE_TERMS = tuple(W["soft_penalties.adjacent_domain_title_terms"])
REMOTE_WORK_MODE_ALIASES = set(W["soft_penalties.remote_work_mode_aliases"])


def _current_title(cand: dict) -> str:
    return (cand.get("profile_current_title") or "").lower()


def _current_company(cand: dict) -> str:
    return (cand.get("profile_current_company") or "").lower()


def has_weak_core_ir_evidence(cand: dict) -> bool:
    """Weak project evidence for the core retrieval/vector/eval requirements."""
    return (
        cand.get("retrieval_search", 0.0) < W["soft_penalties.weak_ir_retrieval_threshold"]
        and cand.get("vector_db_hybrid", 0.0) < W["soft_penalties.weak_ir_vector_threshold"]
        and cand.get("eval_framework", 0.0) < W["soft_penalties.weak_ir_eval_threshold"]
    )


def has_adjacent_domain_weak_ir(cand: dict) -> bool:
    title = _current_title(cand)
    return any(term in title for term in ADJACENT_DOMAIN_TITLE_TERMS) and has_weak_core_ir_evidence(cand)


def has_current_consulting_weak_ir(cand: dict) -> bool:
    company = _current_company(cand)
    current_consulting = any(firm in company for firm in constants.CONSULTING_FIRMS)
    return current_consulting and (
        has_weak_core_ir_evidence(cand)
        or (
            cand.get("product_builder_score", 1.0) < W["soft_penalties.current_consulting_product_builder_threshold"]
            and cand.get("career_ir_density", 1.0) < W["soft_penalties.career_ir_density_threshold"]
        )
    )


def has_long_notice_weak_eval(cand: dict) -> bool:
    notice_days = cand.get("beh_notice_period_days")
    if notice_days is None:
        return False
    try:
        notice_days = float(notice_days)
    except (TypeError, ValueError):
        return False
    return (
        notice_days >= W["behavioral.notice_moderate_days"]
        and cand.get("eval_framework", 0.0) < W["soft_penalties.weak_ir_eval_threshold"]
    )


def yoe_floor_modifier(cand: dict) -> float:
    """Apply the JD's higher bar for candidates below the 5-year preference."""
    try:
        yoe = float(cand.get("profile_years_of_experience", 0.0))
    except (TypeError, ValueError):
        return 1.0

    if yoe <= 0:
        return 1.0
    if yoe < W["soft_penalties.yoe_far_below_floor_years"]:
        return W["soft_penalties.yoe_far_below_floor_mult"]
    if yoe < W["soft_penalties.yoe_floor_years"]:
        return W["soft_penalties.yoe_below_floor_mult"]
    return 1.0


def _is_preferred_or_welcome_location(cand: dict) -> bool:
    location = (cand.get("beh_location") or "").lower()
    return any(city in location for city in PREFERRED_CITIES | WELCOME_CITIES)


def _outside_preferred_no_reloc(cand: dict) -> bool:
    country = (cand.get("beh_country") or "").lower()
    if not country:
        return False
    if country != "india":
        return not cand.get("beh_willing_to_relocate", False)
    return (
        not _is_preferred_or_welcome_location(cand)
        and not cand.get("beh_willing_to_relocate", False)
    )


def has_bad_logistics_combo(cand: dict) -> bool:
    notice_days = cand.get("beh_notice_period_days")
    try:
        notice_bad = notice_days is not None and float(notice_days) > W["behavioral.notice_moderate_days"]
    except (TypeError, ValueError):
        notice_bad = False
    return notice_bad and _outside_preferred_no_reloc(cand)


def has_short_notice_strong_plan_fit(cand: dict) -> bool:
    """Flexible-location exception for evidence-rich India candidates with reachable notice."""
    country = (cand.get("beh_country") or "").lower()
    if country != "india" or cand.get("beh_willing_to_relocate", False):
        return False
    try:
        notice_days = float(cand.get("beh_notice_period_days"))
    except (TypeError, ValueError):
        return False
    return (
        notice_days <= W["behavioral.notice_mild_days"]
        and cand.get("ninety_day_alignment", 0.0) >= 0.85
        and cand.get("retrieval_search", 0.0) >= 2.0
        and cand.get("eval_framework", 0.0) >= 2.0
    )


def has_reachable_elite_plan_fit(cand: dict) -> bool:
    """Full JD-plan coverage plus practical reachability should not be buried."""
    try:
        notice_days = float(cand.get("beh_notice_period_days"))
        response_rate = float(cand.get("beh_recruiter_response_rate", 0.0))
        ce_score = float(cand.get("ce_score", 0.0))
    except (TypeError, ValueError):
        return False
    return (
        cand.get("retrieval_search", 0.0) >= 2.0
        and cand.get("vector_db_hybrid", 0.0) >= 2.0
        and cand.get("eval_framework", 0.0) >= 2.0
        and cand.get("ltr_reranking", 0.0) >= 3.0
        and cand.get("career_eval_density", 0.0) >= 0.80
        and cand.get("ninety_day_alignment", 0.0) >= 0.90
        and response_rate >= W["behavioral.passive_response_skip_threshold"]
        and notice_days <= W["behavioral.notice_mild_days"]
        and cand.get("beh_willing_to_relocate", False)
        and ce_score >= W["behavioral.reachable_elite_ce_threshold"]
        and cand.get("final_phase4_score", 0.0) <= W["behavioral.reachable_elite_phase4_max"]
        and (cand.get("target_skill_duration_contradiction", 0) or 0) <= 1
    )


def has_full_plan_coverage(cand: dict) -> bool:
    """Candidate covers the JD's retrieval, vector/hybrid, ranking, and eval plan."""
    return (
        cand.get("retrieval_search", 0.0) >= 2.0
        and cand.get("vector_db_hybrid", 0.0) >= 2.0
        and cand.get("eval_framework", 0.0) >= 2.0
        and cand.get("ltr_reranking", 0.0) >= 2.0
        and cand.get("ninety_day_alignment", 0.0) >= W["behavioral.full_plan_band_ninety_min"]
    )


def full_plan_band_bonus(cand: dict, base_score: float) -> float:
    """
    Small rescue for the review band when all JD-plan signals agree.

    This is not a top-candidate boost: it only applies to candidates already in
    the strong-but-not-top score band and requires CE agreement plus reachable
    hiring signals. It intentionally leaves long-notice/no-relocation candidates
    to the normal logistics penalties.
    """
    try:
        notice_days = float(cand.get("beh_notice_period_days"))
        response_rate = float(cand.get("beh_recruiter_response_rate", 0.0))
        ce_score = float(cand.get("ce_score", 0.0))
        phase4_score = float(cand.get("final_phase4_score", 0.0))
        target_contradictions = int(cand.get("target_skill_duration_contradiction", 0) or 0)
        base_score = float(base_score)
    except (TypeError, ValueError):
        return 0.0

    if (
        not has_full_plan_coverage(cand)
        or phase4_score > W["behavioral.full_plan_band_phase4_max"]
        or base_score < W["behavioral.full_plan_band_score_min"]
        or base_score > W["behavioral.full_plan_band_score_max"]
    ):
        return 0.0

    if cand.get("beh_willing_to_relocate", False):
        if (
            notice_days <= W["behavioral.notice_mild_days"]
            and response_rate >= W["behavioral.full_plan_band_reloc_response_min"]
            and ce_score >= W["behavioral.full_plan_band_reloc_ce_min"]
            and target_contradictions <= W["behavioral.full_plan_band_reloc_target_contradiction_max"]
        ):
            return W["behavioral.full_plan_band_reloc_bonus"]
        return 0.0

    country = (cand.get("beh_country") or "").lower()
    if country != "india":
        return 0.0

    if (
        notice_days <= W["behavioral.notice_ideal_days"]
        and response_rate >= W["behavioral.full_plan_band_no_reloc_short_response_min"]
        and ce_score >= W["behavioral.full_plan_band_no_reloc_short_ce_min"]
        and target_contradictions <= W["behavioral.full_plan_band_no_reloc_target_contradiction_max"]
    ):
        return W["behavioral.full_plan_band_no_reloc_short_bonus"]

    if (
        notice_days <= W["behavioral.notice_mild_days"]
        and response_rate >= W["behavioral.full_plan_band_no_reloc_mild_response_min"]
        and ce_score >= W["behavioral.full_plan_band_no_reloc_mild_ce_min"]
        and cand.get("career_eval_density", 0.0) >= W["behavioral.full_plan_band_no_reloc_mild_career_eval_min"]
        and target_contradictions <= W["behavioral.full_plan_band_no_reloc_target_contradiction_max"]
    ):
        return W["behavioral.full_plan_band_no_reloc_mild_bonus"]

    return 0.0


def reachability_multiplier(cand: dict, reference_date: date) -> float:
    mult = 1.0

    last_active_str = cand.get("beh_last_active_date")
    if last_active_str:
        try:
            days_inactive = (reference_date - date.fromisoformat(last_active_str)).days
            if days_inactive > W["behavioral.inactive_heavy_days"]:
                mult *= W["behavioral.inactive_heavy_mult"]
            elif days_inactive > W["behavioral.inactive_moderate_days"]:
                mult *= W["behavioral.inactive_moderate_mult"]
        except ValueError:
            pass

    try:
        response_rate = float(cand.get("beh_recruiter_response_rate", 1.0))
    except (TypeError, ValueError):
        response_rate = 1.0
    if (
        cand.get("beh_open_to_work") is False
        and response_rate < W["behavioral.passive_response_skip_threshold"]
    ):
        mult *= W["behavioral.not_open_mult"]

    if response_rate < W["behavioral.low_response_rate_threshold"]:
        mult *= W["behavioral.low_response_mult"]

    return mult


def notice_modifier(days) -> float:
    if days is None:
        return 1.00
    if days <= W["behavioral.notice_ideal_days"]:
        return W["behavioral.notice_ideal_mult"]
    elif days <= W["behavioral.notice_mild_days"]:
        return W["behavioral.notice_mild_mult"]
    elif days <= W["behavioral.notice_moderate_days"]:
        return W["behavioral.notice_moderate_mult"]
    else:
        return W["behavioral.notice_bad_mult"]


def location_modifier(cand: dict) -> float:
    location = cand.get("beh_location", "").lower()
    country  = cand.get("beh_country", "").lower()
    willing  = cand.get("beh_willing_to_relocate", False)

    if any(city in location for city in PREFERRED_CITIES):
        return W["behavioral.location_pune_noida_mult"]
    if any(city in location for city in WELCOME_CITIES):
        return W["behavioral.location_welcome_cities_mult"] if willing else W["behavioral.location_welcome_no_reloc"]
    if country == "india" and willing:
        return W["behavioral.location_india_willing_mult"]
    if country == "india":
        return W["behavioral.location_india_no_reloc_mult"]
    if willing:
        return W["behavioral.location_abroad_willing_mult"]
    return W["behavioral.location_abroad_no_reloc_mult"]


def social_proof_boost(cand: dict) -> float:
    boost = 0.0

    if cand.get("beh_github_activity_score", -1) > W["social_proof.github_threshold"]:
        boost += W["social_proof.github_boost"]
    if cand.get("beh_saved_by_recruiters_30d", -1) > W["social_proof.saved_threshold"]:
        boost += W["social_proof.saved_boost"]
    if cand.get("beh_endorsements_received", -1) > W["social_proof.endorsements_threshold"]:
        boost += W["social_proof.endorsements_boost"]
    if cand.get("beh_interview_completion_rate", 0) > W["social_proof.interview_completion_threshold"]:
        boost += W["social_proof.interview_completion_boost"]
    if cand.get("beh_offer_acceptance_rate", -1) > W["social_proof.offer_accept_threshold"]:
        boost += W["social_proof.offer_accept_boost"]
    if cand.get("beh_profile_completeness_score", 0) > W["social_proof.profile_completeness_threshold"]:
        boost += W["social_proof.profile_completeness_boost"]
    if cand.get("beh_linkedin_connected", False):
        boost += W["social_proof.linkedin_connected_boost"]

    avg_rt = cand.get("beh_avg_response_time_hours", 24.0)
    if avg_rt <= W["social_proof.fast_response_threshold"] and \
       cand.get("beh_recruiter_response_rate", 0) >= W["social_proof.fast_response_rate_min"]:
        boost += W["social_proof.fast_response_boost"]

    return min(boost, W["social_proof.social_proof_max"])


def seniority_modifier(cand: dict) -> float:
    return cand.get("seniority_score", 1.0)


def soft_penalties(cand: dict) -> float:
    multiplier = 1.0

    # Consistency score: drops per contradiction, floored at minimum
    contradictions = (
        cand.get("contradiction_skill_duration", 0) +
        cand.get("contradiction_assessment", 0)
    )
    drop = W["soft_penalties.consistency_drop_per_contradiction"]
    floor_ = W["soft_penalties.consistency_floor"]
    consistency_score = max(floor_, 1.0 - (drop * contradictions))
    multiplier *= consistency_score

    if cand.get("title_velocity_flag", False):
        multiplier *= W["soft_penalties.title_velocity_mult"]

    if cand.get("code_stopped", False):
        multiplier *= W["soft_penalties.code_stopped_mult"]

    if cand.get("langchain_only_flag", False):
        multiplier *= W["soft_penalties.langchain_only_mult"]

    if cand.get("keyword_stuffer_flag", False):
        multiplier *= JD_FEATURE_CONTRACT["multiplier_values"]["keyword_stuffer_penalty"]

    pref_mode = cand.get("beh_preferred_work_mode", "").lower().strip()
    if pref_mode in REMOTE_WORK_MODE_ALIASES:
        multiplier *= W["soft_penalties.remote_pref_mult"]

    if cand.get("research_only", False):
        multiplier *= W["soft_penalties.research_only_mult"]

    if cand.get("wrong_domain", False):
        multiplier *= W["soft_penalties.wrong_domain_mult"]

    if cand.get("closed_source_flag", False):
        multiplier *= W["soft_penalties.closed_source_mult"]

    target_duration_contradictions = cand.get("target_skill_duration_contradiction", 0)
    if target_duration_contradictions >= 2:
        multiplier *= W["soft_penalties.target_skill_duration_multi_mult"]
    elif target_duration_contradictions == 1:
        multiplier *= W["soft_penalties.target_skill_duration_one_mult"]

    if has_adjacent_domain_weak_ir(cand):
        multiplier *= W["soft_penalties.adjacent_domain_weak_ir_mult"]

    if has_current_consulting_weak_ir(cand):
        multiplier *= W["soft_penalties.current_consulting_weak_ir_mult"]

    if has_long_notice_weak_eval(cand):
        multiplier *= W["soft_penalties.long_notice_weak_eval_mult"]

    if _outside_preferred_no_reloc(cand):
        if has_short_notice_strong_plan_fit(cand):
            multiplier *= W["soft_penalties.no_reloc_strong_fit_mult"]
        else:
            multiplier *= W["soft_penalties.no_reloc_outside_preferred_mult"]

    if has_bad_logistics_combo(cand):
        multiplier *= W["soft_penalties.bad_logistics_combo_mult"]

    if cand.get("isolated_template_risk", False):
        multiplier *= W["soft_penalties.isolated_template_risk_mult"]

    if cand.get("career_ir_density", 1.0) < W["soft_penalties.career_ir_density_threshold"]:
        multiplier *= W["soft_penalties.low_career_ir_density_mult"]

    multiplier *= yoe_floor_modifier(cand)

    return multiplier


def has_floor_exempt_penalty(cand: dict) -> bool:
    """Explicit JD disqualifiers should not be rescued by the generic combined floor."""
    return (
        ("consulting_heavy_soft_penalty" in FLOOR_EXEMPT_MULTIPLIERS and cand.get("consulting_only", False))
        or ("pure_research_penalty" in FLOOR_EXEMPT_MULTIPLIERS and cand.get("research_only", False))
        or ("computer_vision_trap" in FLOOR_EXEMPT_MULTIPLIERS and cand.get("wrong_domain", False))
        or ("langchain_tourist_trap" in FLOOR_EXEMPT_MULTIPLIERS and cand.get("langchain_only_flag", False))
        or ("keyword_stuffer_penalty" in FLOOR_EXEMPT_MULTIPLIERS and cand.get("keyword_stuffer_flag", False))
        or has_adjacent_domain_weak_ir(cand)
        or has_current_consulting_weak_ir(cand)
        or has_bad_logistics_combo(cand)
        or cand.get("isolated_template_risk", False)
    )


def compute_final_score(cand: dict, reference_date: date) -> float:
    phase4_score = cand.get("final_phase4_score", 0.0)

    # Honeypot / Impossible / Ghost kill switch
    if (
        cand.get("impossible_flag", False)
        or cand.get("suspicious_flag", False)
        or cand.get("is_ghost", False)
    ):
        return phase4_score * W["behavioral.honeypot_multiplier"]

    reachability_mult = reachability_multiplier(cand, reference_date)
    penalty_mult      = soft_penalties(cand)

    notice_mult    = notice_modifier(cand.get("beh_notice_period_days"))
    loc_mult       = location_modifier(cand)
    seniority_mult = seniority_modifier(cand)
    writing_mult   = cand.get("writing_signal", 1.0)

    logistical_mult = notice_mult * loc_mult * seniority_mult * writing_mult
    logistical_mult = max(logistical_mult, W["behavioral.logistical_floor"])

    combined_mult = reachability_mult * penalty_mult * logistical_mult
    if not has_floor_exempt_penalty(cand):
        combined_mult = max(combined_mult, W["behavioral.combined_floor"])

    # Soft honeypot score compound penalty (for non-flagged suspicious profiles)
    honeypot_score = cand.get("honeypot_score", 0.0)
    honeypot_mult  = max(
        0.0,
        1.0 - (honeypot_score * W["behavioral.honeypot_score_penalty_factor"]),
    )
    combined_mult *= honeypot_mult

    # Additive bonuses
    ninety_day_alignment = cand.get("ninety_day_alignment", 0.0)
    ninety_day_bonus     = W["behavioral.ninety_day_bonus_max"] * ninety_day_alignment
    elite_plan_bonus = (
        W["behavioral.reachable_elite_plan_bonus"]
        if has_reachable_elite_plan_fit(cand)
        else 0.0
    )

    social_boost = social_proof_boost(cand)

    base_final = (
        phase4_score * combined_mult
        + ninety_day_bonus
        + elite_plan_bonus
        + social_boost
    )
    final = base_final + full_plan_band_bonus(cand, base_final)

    # Floor protection: if penalties are extreme, force to 0.0
    if penalty_mult < W["behavioral.penalty_floor_zero"]:
        return 0.0

    final = min(final, 120.0)
    return round(float(final), 6)


def assign_ranks(scored_candidates: list) -> list:
    sorted_cands = sorted(
        scored_candidates,
        key=lambda c: (-c.get("final_score", 0.0), c.get("candidate_id", ""))
    )
    for rank, c in enumerate(sorted_cands, start=1):
        c["rank"] = rank

    for i in range(1, len(sorted_cands)):
        assert sorted_cands[i]["final_score"] <= sorted_cands[i-1]["final_score"], \
            "Score sorting failed"

    return sorted_cands
