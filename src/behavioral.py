"""
src/behavioral.py — Phase 5: Behavioral Re-ranking + Penalization

Applies reachability modifiers, social proof boosts, and soft penalties.
All numeric thresholds and multipliers sourced from weights.yaml via W.

Applied late in the pipeline over the configured Phase 5 candidate pool to
avoid dropping strong passive candidates during early retrieval.
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


def _num(cand: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(cand.get(key, default) or 0.0)
    except (TypeError, ValueError):
        return default


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


def has_core_retrieval_or_ranking(cand: dict) -> bool:
    """The JD needs real retrieval/search or recommender/ranking system evidence."""
    return (
        _num(cand, "retrieval_search") >= 3.0
        or _num(cand, "sys_experience_score") >= 1.0
        or _num(cand, "runtime_production_retrieval_signal") > 0.0
    )


def has_production_vector_evidence(cand: dict) -> bool:
    """Vector/hybrid must-have requires career/project evidence, not a skill-only mention."""
    return (
        _num(cand, "vector_db_hybrid") >= 3.0
        or _num(cand, "runtime_production_vector_signal") > 0.0
        or _num(cand, "runtime_corroborated_vector_signal") > 0.0
    )


def has_ranking_eval_evidence(cand: dict) -> bool:
    """Evaluation must-have needs career-level eval or production ranking feedback signals."""
    return (
        _num(cand, "eval_framework") >= 2.0
        or _num(cand, "runtime_career_eval_signal") > 0.0
        or _num(cand, "runtime_career_eval_adjacent_signal") > 0.0
    )


def has_python_evidence(cand: dict) -> bool:
    """Python can be proven by extracted Python evidence or runtime raw-profile evidence."""
    return (
        _num(cand, "python_coding") > 0.0
        or _num(cand, "runtime_career_python_signal") > 0.0
    )


def missing_must_have_buckets(cand: dict) -> list[str]:
    """Return true JD must-have buckets that are absent in extracted evidence."""
    missing: list[str] = []
    if (
        ("retrieval_search" in cand or "sys_experience_score" in cand)
        and not has_core_retrieval_or_ranking(cand)
    ):
        missing.append("core retrieval/ranking")
    if (
        "vector_db_hybrid" in cand
        and not has_production_vector_evidence(cand)
    ):
        missing.append("vector/hybrid search")
    if (
        "eval_framework" in cand
        and not has_ranking_eval_evidence(cand)
    ):
        missing.append("ranking evaluation")
    if (
        "python_coding" in cand
        and not has_python_evidence(cand)
    ):
        missing.append("Python")
    return missing


def has_must_have_raw_evidence_override(cand: dict) -> bool:
    """Manual/extractor override for a zero bucket proven false by raw text."""
    return bool(cand.get("must_have_raw_evidence_override", False))


def has_top100_must_have_gap(cand: dict) -> bool:
    """A true must-have zero makes the profile unfit for this Top-100 shortlist."""
    return bool(missing_must_have_buckets(cand)) and not has_must_have_raw_evidence_override(cand)


def has_severe_must_have_gap(cand: dict) -> bool:
    missing = missing_must_have_buckets(cand)
    return (
        len(missing) >= 2
        or "core retrieval/ranking" in missing
        or "vector/hybrid search" in missing
        or "ranking evaluation" in missing
    )


def has_true_disqualifier(cand: dict) -> bool:
    """JD should-not-have flags that should not remain Top-100 viable if true."""
    return any(
        bool(cand.get(flag, False))
        for flag in (
            "impossible_flag",
            "suspicious_flag",
            "is_ghost",
            "keyword_stuffer_flag",
            "langchain_only_flag",
            "research_only",
            "consulting_only",
            "wrong_domain",
            "title_chaser_flag",
            "code_stopped",
        )
    )


def hard_disqualification_reason(cand: dict) -> str:
    """Return the JD gate reason that removes a candidate from Top-100 ranking."""
    if cand.get("impossible_flag", False) or cand.get("suspicious_flag", False) or cand.get("is_ghost", False):
        return "hard trust/honeypot/ghost flag"
    if cand.get("keyword_stuffer_flag", False):
        return "keyword-stuffer profile"
    if cand.get("langchain_only_flag", False):
        return "framework-demo-only / LangChain-only profile"
    if cand.get("research_only", False):
        return "pure research without production deployment"
    if cand.get("consulting_only", False):
        return "consulting-only career"
    if cand.get("wrong_domain", False):
        return "wrong-domain CV/speech/robotics profile without enough NLP/IR"
    if cand.get("title_chaser_flag", False):
        return "title-chaser career trajectory"
    if cand.get("code_stopped", False):
        return "senior profile appears to have stopped hands-on coding"
    missing = missing_must_have_buckets(cand)
    if missing and not has_must_have_raw_evidence_override(cand):
        return f"missing true must-have evidence: {', '.join(missing)}"
    return ""


def is_hard_disqualified(cand: dict) -> bool:
    return bool(hard_disqualification_reason(cand))


def ce_core_delta(cand: dict) -> float:
    return abs(_num(cand, "core_score") - _num(cand, "ce_score"))


def candidate_ce_score(cand: dict) -> float:
    return _num(cand, "ce_score", _num(cand, "cross_encoder_score", 0.0))


def ce_rescue_with_core_gap(cand: dict) -> bool:
    return (
        _num(cand, "ce_score") - _num(cand, "core_score") >= 20.0
        and has_severe_must_have_gap(cand)
    )


def core_over_ce_disagreement(cand: dict) -> float:
    """Positive when handcrafted regex/core score is much higher than semantic CE."""
    return _num(cand, "core_score") - _num(cand, "ce_score")


def ce_ceiling_sanity_risk(cand: dict) -> bool:
    """Perfect CE scores are useful but should not create a free ride."""
    return _num(cand, "ce_score") >= W["soft_penalties.ce_ceiling_threshold"]


def must_have_gap_multiplier(cand: dict) -> float:
    """Multiplicative demotion for accurate missing must-have extraction."""
    missing = missing_must_have_buckets(cand)
    if not missing:
        return 1.0
    multiplier = 1.0
    if "core retrieval/ranking" in missing:
        multiplier *= W["soft_penalties.zero_core_retrieval_mult"]
    if "vector/hybrid search" in missing:
        multiplier *= W["soft_penalties.zero_vector_hybrid_mult"]
    if "ranking evaluation" in missing:
        multiplier *= W["soft_penalties.zero_eval_mult"]
    if "Python" in missing:
        multiplier *= W["soft_penalties.zero_python_mult"]
    if (
        "vector_db_hybrid" in cand
        and "ltr_reranking" in cand
        and _num(cand, "vector_db_hybrid") <= 0.0
        and _num(cand, "ltr_reranking") <= 0.0
    ):
        multiplier *= W["soft_penalties.double_zero_vector_ltr_mult"]
    if len(missing) >= 2:
        multiplier *= W["soft_penalties.multiple_must_have_zero_mult"]
    return multiplier


def must_have_score_ceiling(cand: dict) -> float | None:
    """Absolute score ceiling for extracted must-have zeros."""
    missing = missing_must_have_buckets(cand)
    if not missing:
        return None
    if (
        "vector_db_hybrid" in cand
        and "ltr_reranking" in cand
        and _num(cand, "vector_db_hybrid") <= 0.0
        and _num(cand, "ltr_reranking") <= 0.0
    ):
        return W["behavioral.double_zero_must_have_score_ceiling"]
    if has_severe_must_have_gap(cand):
        return W["behavioral.severe_must_have_gap_score_ceiling"]
    return W["behavioral.single_must_have_gap_score_ceiling"]


def has_missing_github_activity(cand: dict) -> bool:
    return _num(cand, "beh_github_activity_score", 0.0) < 0.0


def has_notice_risk(cand: dict) -> bool:
    try:
        notice_days = float(cand.get("beh_notice_period_days"))
    except (TypeError, ValueError):
        return False
    return notice_days >= W["behavioral.long_notice_exclusion_days"]


def has_adjacent_domain_weak_ir(cand: dict) -> bool:
    title = _current_title(cand)
    return any(term in title for term in ADJACENT_DOMAIN_TITLE_TERMS) and has_weak_core_ir_evidence(cand)


def has_current_consulting_weak_ir(cand: dict) -> bool:
    company = _current_company(cand)
    industry = (cand.get("runtime_current_role_text") or "").lower()
    current_consulting = (
        any(firm in company for firm in constants.CONSULTING_FIRMS)
        or cand.get("runtime_current_services_signal", 0.0) >= 1.0
        or "ai services" in industry
        or "it services" in industry
        or "consulting" in industry
        or "outsourcing" in industry
        or "staffing" in industry
        or "professional services" in industry
        or "system integration" in industry
        or "client delivery" in industry
    )
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


def has_location_risk(cand: dict) -> bool:
    """Location/no-relocation combination that should be a strong penalty."""
    country = (cand.get("beh_country") or "").lower()
    location = (cand.get("beh_location") or "").strip()
    if not country:
        return False
    if country != "india":
        return not cand.get("beh_willing_to_relocate", False)
    if not location:
        return False
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

    # Fix 6: Mild open-to-work tie-breaker penalties
    if cand.get("beh_open_to_work") is False and response_rate >= 0.60:
        loc = cand.get("beh_location", "").lower()
        will_reloc = cand.get("beh_willing_to_relocate", False)
        # simplistic check matching the external one
        in_pref = any(city in loc for city in ["pune", "noida"])
        in_welc = any(city in loc for city in ["delhi", "gurgaon", "gurugram", "bangalore", "bengaluru", "mumbai", "hyderabad", "chennai"])
        if not will_reloc and not in_pref and not in_welc:
            mult *= W["behavioral.not_open_outside_preferred_no_reloc_mult"]
        else:
            mult *= W["behavioral.not_open_but_reachable_mult"]

    if response_rate < W["behavioral.low_response_rate_threshold"]:
        mult *= W["behavioral.low_response_mult"]

    return mult


def notice_modifier(days) -> float:
    if days is None:
        return 1.00
    if days == 0:
        return W["behavioral.notice_zero_mult"]
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

    github_score = cand.get("beh_github_activity_score", -1)
    if github_score is None:
        github_score = -1
    if github_score > W["social_proof.github_threshold"]:
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

    multiplier *= must_have_gap_multiplier(cand)

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

    if cand.get("title_bump_flag", False):
        multiplier *= W["soft_penalties.title_bump_mult"]

    if cand.get("title_chaser_flag", False):
        multiplier *= W["soft_penalties.title_chaser_mult"]

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
        if (
            cand.get("runtime_full_plan_signal", 0.0) >= W["behavioral.runtime_full_plan_min"]
            and cand.get("eval_framework", 0.0) >= W["soft_penalties.weak_ir_eval_threshold"]
            and cand.get("career_ir_density", 0.0) >= W["soft_penalties.career_ir_density_threshold"]
        ):
            multiplier *= W["soft_penalties.target_skill_duration_strong_plan_mult"]
        else:
            multiplier *= W["soft_penalties.target_skill_duration_multi_mult"]
    elif target_duration_contradictions == 1:
        multiplier *= W["soft_penalties.target_skill_duration_one_mult"]

    try:
        max_overclaim = float(cand.get("max_target_skill_overclaim_months", 0.0))
        if max_overclaim >= 24.0:
            multiplier *= 0.85
        elif max_overclaim >= 6.0 and target_duration_contradictions < 2:
            # Only apply mild overclaim penalty when not already penalized by contradiction check above
            multiplier *= 0.90
    except (ValueError, TypeError):
        pass

    if cand.get("runtime_current_services_signal", 0.0) >= 1.0:
        if (
            cand.get("product_builder_score", 0.0) >= W["soft_penalties.current_services_strong_product_min"]
            and cand.get("career_ir_density", 0.0) >= W["soft_penalties.current_services_strong_ir_min"]
        ):
            multiplier *= W["soft_penalties.current_services_strong_fit_mult"]
        else:
            multiplier *= W["soft_penalties.current_services_default_mult"]

    if has_adjacent_domain_weak_ir(cand):
        multiplier *= W["soft_penalties.adjacent_domain_weak_ir_mult"]

    if has_current_consulting_weak_ir(cand):
        multiplier *= W["soft_penalties.current_consulting_weak_ir_mult"]

    if has_long_notice_weak_eval(cand):
        multiplier *= W["soft_penalties.long_notice_weak_eval_mult"]

    try:
        notice_days = float(cand.get("beh_notice_period_days"))
    except (TypeError, ValueError):
        notice_days = 0.0
    if notice_days >= W["behavioral.long_notice_exclusion_days"]:
        multiplier *= W["soft_penalties.long_notice_extreme_mult"]

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
    if cand.get("career_ir_density", 1.0) < W["scoring.low_career_ir_density_threshold"]:
        multiplier *= W["soft_penalties.low_career_ir_density_strict_mult"]
    adjacent_ratio = cand.get("adjacent_career_ratio", 0.0)
    if adjacent_ratio >= 0.65:
        multiplier *= W["soft_penalties.adjacent_career_high_mult"]
    elif adjacent_ratio >= 0.60:
        multiplier *= W["soft_penalties.high_adjacent_career_mult"]
    elif adjacent_ratio >= 0.50:
        multiplier *= W["soft_penalties.adjacent_career_half_mult"]
    core_ce_gap = core_over_ce_disagreement(cand)
    very_high_core = _num(cand, "core_score") >= 95.0
    no_dup_desc = int(cand.get("eval_metric_duplicate_descriptions", 0) or 0) == 0
    if core_ce_gap >= W["soft_penalties.ce_delta_high_threshold"]:
        if very_high_core and no_dup_desc:
            # core >= 95 with clean descriptions: CE likely under-read the profile, not a regex overread
            multiplier *= W["soft_penalties.ce_delta_moderate_mult"]
        else:
            multiplier *= W["soft_penalties.ce_delta_high_mult"]
    elif core_ce_gap >= W["soft_penalties.ce_delta_moderate_threshold"]:
        multiplier *= W["soft_penalties.ce_delta_moderate_mult"]
    if ce_ceiling_sanity_risk(cand) and _num(cand, "core_score") < W["behavioral.reachable_elite_phase4_max"]:
        multiplier *= W["soft_penalties.ce_ceiling_mult"]
    if ce_rescue_with_core_gap(cand):
        multiplier *= W["soft_penalties.multiple_must_have_zero_mult"]

    if cand.get("runtime_adjacent_internal_only_flag", False):
        if candidate_ce_score(cand) < 60.0 and cand.get("core_score", 0.0) >= 80.0:
            multiplier *= W["soft_penalties.adjacent_internal_only_weak_ce_mult"]
        else:
            multiplier *= W["soft_penalties.adjacent_internal_only_mult"]

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
        or cand.get("title_chaser_flag", False)
        or has_adjacent_domain_weak_ir(cand)
        or has_current_consulting_weak_ir(cand)
        or has_bad_logistics_combo(cand)
        or cand.get("isolated_template_risk", False)
        or (W["soft_penalties.must_have_gap_floor_exempt"] and has_severe_must_have_gap(cand))
    )


def compute_final_score(cand: dict, reference_date: date) -> float:
    phase4_score = cand.get("final_phase4_score", 0.0)

    # True JD should-not-have / honeypot / ghost kill switch.
    if has_true_disqualifier(cand):
        return round(float(phase4_score * W["behavioral.disqualifier_multiplier"]), 6)
    if has_top100_must_have_gap(cand):
        return round(float(phase4_score * W["behavioral.must_have_missing_multiplier"]), 6)

    reachability_mult = reachability_multiplier(cand, reference_date)
    penalty_mult      = soft_penalties(cand)

    notice_mult    = notice_modifier(cand.get("beh_notice_period_days"))
    loc_mult       = location_modifier(cand)
    seniority_mult = seniority_modifier(cand)
    writing_mult   = cand.get("writing_signal", 1.0)

    logistical_mult = notice_mult * loc_mult * seniority_mult * writing_mult
    try:
        response_rate = float(cand.get("beh_recruiter_response_rate", 0.0) or 0.0)
    except:
        response_rate = 0.0

    if (
        cand.get("runtime_same_project_full_system_bonus_applied", False)
        and candidate_ce_score(cand) >= 80.0
        and response_rate >= 0.70
    ):
        logistical_mult = max(logistical_mult, W["behavioral.elite_fit_logistics_penalty_floor_mult"])
    else:
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

    runtime_full_plan = float(cand.get("runtime_full_plan_signal", 0.0) or 0.0)
    try:
        response_rate = float(cand.get("beh_recruiter_response_rate", 0.0) or 0.0)
    except (TypeError, ValueError):
        response_rate = 0.0
    try:
        notice_days = float(cand.get("beh_notice_period_days", 999.0))
    except (TypeError, ValueError):
        notice_days = 999.0
    full_plan_bonus = 0.0
    if runtime_full_plan >= W["behavioral.runtime_full_plan_min"]:
        full_plan_bonus = runtime_full_plan * W["behavioral.runtime_full_plan_bonus_max"]
        if response_rate >= W["behavioral.runtime_full_plan_reachable_response_min"]:
            full_plan_bonus += W["behavioral.runtime_full_plan_reachable_bonus"]
        elif response_rate < W["behavioral.low_response_rate_threshold"]:
            full_plan_bonus *= W["behavioral.runtime_full_plan_low_response_mult"]
        if notice_days > W["behavioral.notice_moderate_days"]:
            full_plan_bonus *= W["behavioral.runtime_full_plan_long_notice_mult"]
        if int(cand.get("target_skill_duration_contradiction", 0) or 0) >= 3:
            full_plan_bonus *= W["behavioral.runtime_full_plan_high_overclaim_mult"]
        if has_severe_must_have_gap(cand) or cand.get("career_ir_density", 1.0) < W["scoring.low_career_ir_density_threshold"]:
            full_plan_bonus *= W["behavioral.runtime_full_plan_must_have_gap_mult"]

    # Part 4 (Fix 2): Same-project full-system bonus (STRICT) + partial bonus
    full_system_bonus = (
        W["behavioral.same_project_full_system_bonus"]
        if cand.get("runtime_same_project_full_system_bonus_applied", False)
        else (
            W["behavioral.same_project_partial_system_bonus"]
            if cand.get("runtime_same_project_partial_system_bonus_applied", False)
            else 0.0
        )
    )
    cand["runtime_same_project_full_system_bonus_value"] = full_system_bonus
    cand["runtime_same_project_bonus_type"] = "full" if cand.get("runtime_same_project_full_system_bonus_applied", False) else "partial" if cand.get("runtime_same_project_partial_system_bonus_applied", False) else "none"

    # Part 5: Recruiter/Candidate workflow bonus
    recruiter_workflow_bonus = (
        W["behavioral.recruiter_candidate_workflow_bonus"]
        if cand.get("runtime_recruiter_workflow_bonus_applied", False)
        else 0.0
    )
    cand["runtime_recruiter_candidate_workflow_bonus_value"] = recruiter_workflow_bonus

    cand["runtime_passive_responsive_exact_fit_bonus_applied"] = False
    cand["runtime_passive_responsive_exact_fit_bonus"] = 0.0
    cand["runtime_passive_responsive_exact_fit_reason"] = ""
    passive_responsive_exact_fit_bonus = 0.0
    if (
        cand.get("runtime_same_project_full_system_bonus_applied", False)
        and cand.get("runtime_recruiter_workflow_bonus_applied", False)
        and candidate_ce_score(cand) >= 80.0
        and response_rate >= 0.80
        and cand.get("beh_open_to_work") is False
        and notice_days <= W["behavioral.notice_mild_days"]
        and cand.get("beh_willing_to_relocate", False)
        and not has_true_disqualifier(cand)
    ):
        passive_responsive_exact_fit_bonus = W["behavioral.passive_responsive_exact_fit_bonus"]
        cand["runtime_passive_responsive_exact_fit_bonus_applied"] = True
        cand["runtime_passive_responsive_exact_fit_bonus"] = passive_responsive_exact_fit_bonus
        cand["runtime_passive_responsive_exact_fit_reason"] = "full_system_recruiter_workflow_passive_but_responsive"

    # Fix 3: Split-career core coverage bonus
    split_career_bonus = (
        W["behavioral.split_career_core_coverage_bonus"]
        if cand.get("runtime_split_career_core_coverage_bonus_applied", False)
        and not cand.get("runtime_same_project_full_system_bonus_applied", False)
        else 0.0
    )
    cand["runtime_split_career_bonus_value"] = split_career_bonus

    # Fix 5: Evidence-level gating — apply as a phase4_score component modifier
    retrieval_level = int(cand.get("runtime_retrieval_evidence_level", 2))
    vector_level    = int(cand.get("runtime_vector_evidence_level", 2))
    ltr_level       = int(cand.get("runtime_ltr_evidence_level", 2))
    eval_level      = int(cand.get("runtime_eval_evidence_level", 2))
    product_level   = int(cand.get("runtime_product_evidence_level", 2))

    avg_level = (retrieval_level + vector_level + ltr_level + eval_level + product_level) / 5.0
    evidence_gating_mult = 0.75 + 0.125 * avg_level
    evidence_gating_mult = max(0.75, min(1.0, evidence_gating_mult))
    cand["runtime_evidence_gating_multiplier"] = round(evidence_gating_mult, 4)

    cand["runtime_retrieval_bonus_level"] = retrieval_level
    cand["runtime_vector_bonus_level"] = vector_level
    cand["runtime_ltr_bonus_level"] = ltr_level
    cand["runtime_eval_bonus_level"] = eval_level
    cand["runtime_product_bonus_level"] = product_level
    cand["runtime_retrieval_bonus_applied"] = 0.0
    cand["runtime_vector_bonus_applied"] = 0.0
    cand["runtime_ltr_bonus_applied"] = 0.0
    cand["runtime_eval_bonus_applied"] = 0.0
    cand["runtime_product_bonus_applied"] = 0.0

    # Fix 2: Partial system logistics risk penalty
    cand["runtime_partial_system_with_logistics_risk_penalty_applied"] = False
    cand["runtime_partial_system_with_logistics_risk_multiplier"] = 1.0
    cand["runtime_partial_system_with_logistics_risk_reason"] = ""
    cand["runtime_partial_system_low_ce_low_response_penalty_applied"] = False
    cand["runtime_partial_system_low_ce_low_response_multiplier"] = 1.0
    cand["runtime_partial_system_low_ce_low_response_reason"] = ""
    partial_risk_mult = 1.0

    if (
        cand.get("runtime_same_project_bonus_type") == "partial"
        and not cand.get("runtime_recruiter_workflow_bonus_applied", False)
        and ltr_level == 0
        and has_location_risk(cand)
    ):
        partial_risk_mult = W["behavioral.partial_system_with_logistics_risk_mult"]
        cand["runtime_partial_system_with_logistics_risk_penalty_applied"] = True
        cand["runtime_partial_system_with_logistics_risk_multiplier"] = partial_risk_mult
        cand["runtime_partial_system_with_logistics_risk_reason"] = "partial_system_with_no_workflow_no_ltr_and_location_risk"

    partial_quality_mult = 1.0
    if (
        cand.get("runtime_same_project_bonus_type") == "partial"
        and not cand.get("runtime_same_project_full_system_bonus_applied", False)
        and not cand.get("runtime_recruiter_workflow_bonus_applied", False)
        and candidate_ce_score(cand) < 50.0
        and response_rate < 0.50
    ):
        partial_quality_mult = W["behavioral.partial_system_low_ce_low_response_mult"]
        cand["runtime_partial_system_low_ce_low_response_penalty_applied"] = True
        cand["runtime_partial_system_low_ce_low_response_multiplier"] = partial_quality_mult
        cand["runtime_partial_system_low_ce_low_response_reason"] = "partial_system_with_low_ce_and_low_recruiter_response"

    base_final = (
        phase4_score * combined_mult * evidence_gating_mult * partial_risk_mult * partial_quality_mult
        + ninety_day_bonus
        + elite_plan_bonus
        + social_boost
        + full_plan_bonus
        + full_system_bonus
        + recruiter_workflow_bonus
        + passive_responsive_exact_fit_bonus
        + split_career_bonus
    )
    final = base_final + full_plan_band_bonus(cand, base_final)

    cand["true_unclamped_final_score"] = round(float(final), 6)
    cand["raw_final_score"] = round(float(final), 6)

    # Floor protection: if penalties are extreme, force to 0.0
    if penalty_mult < W["behavioral.penalty_floor_zero"]:
        cand["true_unclamped_final_score"] = 0.0
        cand["raw_final_score"] = 0.0
        return 0.0

    ceiling = must_have_score_ceiling(cand)
    if ceiling is not None:
        final = min(final, ceiling)

    # Final clamping
    final = min(final, 100.0)
    cand["final_score"] = final
    return round(float(final), 6)


def assign_ranks(scored_candidates: list) -> list:
    sorted_cands = sorted(
        scored_candidates,
        key=lambda c: (-c.get("true_unclamped_final_score", c.get("raw_final_score", c.get("final_score", 0.0))), c.get("candidate_id", ""))
    )
    for rank, c in enumerate(sorted_cands, start=1):
        c["rank"] = rank

    return sorted_cands
