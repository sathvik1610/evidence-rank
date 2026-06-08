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


def reachability_multiplier(cand: dict, reference_date: date) -> float:
    mult = 1.0
    days_inactive: int | None = None  # cached so OTW compound can reuse it

    last_active_str = cand.get("beh_last_active_date")
    if last_active_str:
        try:
            days_inactive = (reference_date - date.fromisoformat(last_active_str)).days
            if days_inactive > W["behavioral.inactive_heavy_days"]:
                mult *= W["behavioral.inactive_heavy_mult"]
            elif days_inactive > W["behavioral.inactive_moderate_days"]:
                mult *= W["behavioral.inactive_moderate_mult"]
            elif days_inactive > W["behavioral.inactive_light_days"]:
                # New light tier: catches candidates inactive 120-270 days
                mult *= W["behavioral.inactive_light_mult"]
        except ValueError:
            pass

    if not cand.get("beh_open_to_work", True):
        mult *= W["behavioral.not_open_mult"]
        # Compound: not flagged open AND has been inactive for a while → truly unreachable
        # A candidate who hasn't checked in for 3+ months AND isn't open is a recruiter dead-end.
        compound_threshold = W["behavioral.not_open_inactive_compound_days"]
        if days_inactive is not None and days_inactive > compound_threshold:
            mult *= W["behavioral.not_open_inactive_compound_mult"]

    rrr = cand.get("beh_recruiter_response_rate", 1.0)
    if rrr < W["behavioral.low_response_rate_threshold"]:
        mult *= W["behavioral.low_response_mult"]
    elif rrr < W["behavioral.moderate_response_rate_threshold"]:
        mult *= W["behavioral.moderate_response_mult"]

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


def technical_bonus_scale(phase4_score: float) -> float:
    """Scale additive bonuses so behavior cannot rescue weak technical fit."""
    zero = W["behavioral.bonus_zero_phase4_score"]
    full = W["behavioral.bonus_full_phase4_score"]
    if full <= zero:
        return 1.0
    return max(0.0, min(1.0, (phase4_score - zero) / (full - zero)))


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

    # Current support-chatbot/RAG work is adjacent, not the JD's ranking/search
    # ownership target. Phase 3 marks this as low recency; apply a rank-time
    # penalty when there is no evaluation depth and at most weak LTR evidence.
    if (
        cand.get("experience_recency", 0.5) <= 0.3
        and cand.get("eval_framework", 0.0) == 0.0
        and cand.get("ltr_reranking", 0.0) <= 1.0
    ):
        multiplier *= W["soft_penalties.current_chatbot_adjacent_mult"]

    # JD calls ranking evaluation a must-have. Keep this mild: do not reject a
    # strong search/ranking builder, but let complete eval evidence break ties
    # ahead of otherwise similar profiles.
    if (
        cand.get("eval_framework", 0.0) == 0.0
        and cand.get("retrieval_search", 0.0) >= 2.0
        and cand.get("ltr_reranking", 0.0) >= 2.0
        and cand.get("sys_experience_score", 0.0) >= 1.0
    ):
        multiplier *= W["soft_penalties.eval_gap_strong_ranking_mult"]

    if cand.get("keyword_stuffer_flag", False):
        multiplier *= JD_FEATURE_CONTRACT["multiplier_values"]["keyword_stuffer_penalty"]

    pref_mode = cand.get("beh_preferred_work_mode", "").lower().strip()
    if pref_mode in ("remote", "wfh", "work from home"):
        multiplier *= W["soft_penalties.remote_pref_mult"]

    if cand.get("research_only", False):
        multiplier *= W["soft_penalties.research_only_mult"]

    if cand.get("wrong_domain", False):
        multiplier *= W["soft_penalties.wrong_domain_mult"]

    if cand.get("closed_source_flag", False):
        multiplier *= W["soft_penalties.closed_source_mult"]

    # Current-role consulting penalty:
    # consulting_only fires only when 100% of career is consulting.
    # But the JD is also skeptical of candidates currently at IT services firms
    # (TCS, Infosys, etc.) even if they had product company experience before.
    # Apply a softer penalty when current employer is a consulting firm.
    if not cand.get("consulting_only", False):  # full penalty already handled
        current_company = (cand.get("profile_current_company") or "").lower().strip()
        if any(firm in current_company for firm in constants.CONSULTING_FIRMS):
            multiplier *= W["soft_penalties.current_role_consulting_mult"]

    return multiplier


def has_floor_exempt_penalty(cand: dict) -> bool:
    """Explicit JD disqualifiers should not be rescued by the generic combined floor."""
    return (
        ("consulting_heavy_soft_penalty" in FLOOR_EXEMPT_MULTIPLIERS and cand.get("consulting_only", False))
        or ("pure_research_penalty" in FLOOR_EXEMPT_MULTIPLIERS and cand.get("research_only", False))
        or ("computer_vision_trap" in FLOOR_EXEMPT_MULTIPLIERS and cand.get("wrong_domain", False))
        or ("langchain_tourist_trap" in FLOOR_EXEMPT_MULTIPLIERS and cand.get("langchain_only_flag", False))
        or ("keyword_stuffer_penalty" in FLOOR_EXEMPT_MULTIPLIERS and cand.get("keyword_stuffer_flag", False))
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

    # Additive bonuses are gated by technical fit. Redrob signals help choose
    # between qualified candidates; they must not rescue adjacent profiles.
    bonus_scale = technical_bonus_scale(phase4_score)
    ninety_day_alignment = cand.get("ninety_day_alignment", 0.0)
    ninety_day_bonus     = (
        W["behavioral.ninety_day_bonus_max"]
        * ninety_day_alignment
        * bonus_scale
    )

    social_boost = social_proof_boost(cand) * bonus_scale

    final = (
        phase4_score * combined_mult
        + ninety_day_bonus
        + social_boost
    )

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
