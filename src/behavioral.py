"""
src/behavioral.py — Phase 5: Behavioral Re-ranking + Penalization

Applies reachability modifiers, social proof boosts, and soft penalties.
These are applied late in the pipeline (to the top 500 candidates) to avoid
accidentally dropping strong passive candidates during early retrieval.

Operates on the flat candidate dictionary (features + flags + ce_score).
"""

from datetime import date

PUNE_NOIDA_CITIES = {"pune", "noida", "greater noida", "delhi", "new delhi",
                      "gurugram", "gurgaon", "faridabad", "ghaziabad"}
JD_WELCOME_CITIES = {"hyderabad", "mumbai"}
INDIA_ADJACENT = {"bangalore", "bengaluru", "chennai", "kolkata",
                   "ahmedabad", "indore", "jaipur", "chandigarh", "kochi"}

def reachability_multiplier(cand: dict, reference_date: date) -> float:
    mult = 1.0

    last_active_str = cand.get("beh_last_active_date")
    if last_active_str:
        try:
            days_inactive = (reference_date - date.fromisoformat(last_active_str)).days
            if days_inactive > 540:
                mult *= 0.60
            elif days_inactive > 270:
                mult *= 0.75
        except ValueError:
            pass

    if not cand.get("beh_open_to_work", True):
        mult *= 0.85

    if cand.get("beh_recruiter_response_rate", 1.0) < 0.10:
        mult *= 0.90

    return mult

def notice_modifier(days) -> float:
    if days is None:
        return 1.00
    if days <= 30:
        return 1.00
    elif days <= 60:
        return 0.95
    elif days <= 90:
        return 0.90
    else:
        return 0.75

def location_modifier(cand: dict) -> float:
    location = cand.get("beh_location", "").lower()
    country = cand.get("beh_country", "").lower()
    willing = cand.get("beh_willing_to_relocate", False)

    if any(city in location for city in PUNE_NOIDA_CITIES):
        return 1.0
    if any(city in location for city in JD_WELCOME_CITIES):
        return 1.00 if willing else 0.98
    if any(city in location for city in INDIA_ADJACENT):
        return 0.98 if willing else 0.95
    if country == "india" and willing:
        return 0.95
    if country == "india":
        return 0.92
    if willing:
        return 0.90
    return 0.85

def social_proof_boost(cand: dict) -> float:
    boost = 0.0

    if cand.get("beh_github_activity_score", -1) > 60:
        boost += 3.0
    if cand.get("beh_saved_by_recruiters_30d", -1) > 5:
        boost += 4.0
    # profile_views_received_30d not tracked in feature schema; fallback to other signals

    if cand.get("beh_endorsements_received", -1) > 20:
        boost += 1.0
    if cand.get("beh_interview_completion_rate", 0) > 0.80:
        boost += 2.0
    if cand.get("beh_offer_acceptance_rate", -1) > 0.70:
        boost += 1.0

    if cand.get("beh_profile_completeness_score", 0) > 80:
        boost += 1.0
    if cand.get("beh_linkedin_connected", False):
        boost += 1.0

    avg_rt = cand.get("beh_avg_response_time_hours", 24.0)
    if avg_rt <= 4.0 and cand.get("beh_recruiter_response_rate", 0) >= 0.60:
        boost += 1.0

    return min(boost, 12.0)

def seniority_modifier(cand: dict) -> float:
    return cand.get("seniority_score", 1.0)

def soft_penalties(cand: dict) -> float:
    multiplier = 1.0

    # consistency_score: drops 0.15 per contradiction, floored at 0.30.
    # Data comes from contradiction_skill_duration + contradiction_assessment
    # fields now correctly forwarded into the flat dict (BUG 1+2 fix).
    contradictions = (
        cand.get("contradiction_skill_duration", 0) +
        cand.get("contradiction_assessment", 0)
    )
    consistency_score = max(0.30, 1.0 - (0.15 * contradictions))
    multiplier *= consistency_score

    if cand.get("title_velocity_flag", False):
        multiplier *= 0.80

    if cand.get("code_stopped", False):
        multiplier *= 0.75

    if cand.get("langchain_only_flag", False):
        multiplier *= 0.45

    pref_mode = cand.get("beh_preferred_work_mode", "").lower().strip()
    if pref_mode in ("remote", "wfh", "work from home"):
        multiplier *= 0.85

    if cand.get("research_only", False):
        multiplier *= 0.40

    if cand.get("wrong_domain", False):
        multiplier *= 0.50

    if cand.get("closed_source_flag", False):
        multiplier *= 0.80

    return multiplier

def compute_final_score(cand: dict, reference_date: date) -> float:
    phase4_score = cand.get("final_phase4_score", 0.0)

    # Honeypot / Impossible
    if cand.get("impossible_flag", False) or cand.get("suspicious_flag", False):
        return phase4_score * 0.01

    reachability_mult = reachability_multiplier(cand, reference_date)
    penalty_mult = soft_penalties(cand)

    notice_mult = notice_modifier(cand.get("beh_notice_period_days"))
    loc_mult = location_modifier(cand)
    seniority_mult = seniority_modifier(cand)
    writing_mult = cand.get("writing_signal", 1.0)

    logistical_mult = notice_mult * loc_mult * seniority_mult * writing_mult
    logistical_mult = max(logistical_mult, 0.75)

    combined_mult = reachability_mult * penalty_mult * logistical_mult
    combined_mult = max(combined_mult, 0.25)

    honeypot_score = cand.get("honeypot_score", 0.0)
    honeypot_mult = 1.0 - (honeypot_score * 0.40)
    combined_mult *= honeypot_mult

    ninety_day_alignment = cand.get("ninety_day_alignment", 0.0)
    ninety_day_bonus = 8.0 * ninety_day_alignment

    social_boost = social_proof_boost(cand)

    final = (
        phase4_score
        * combined_mult
        + ninety_day_bonus
        + social_boost
    )

    if penalty_mult < 0.20:
        return 0.0

    final = min(final, 120.0)
    return round(float(final), 6)

def assign_ranks(scored_candidates: list[dict]) -> list[dict]:
    sorted_cands = sorted(
        scored_candidates,
        key=lambda c: (-c.get("final_score", 0.0), c.get("candidate_id", ""))
    )
    for rank, c in enumerate(sorted_cands, start=1):
        c["rank"] = rank
        
    for i in range(1, len(sorted_cands)):
        assert sorted_cands[i]["final_score"] <= sorted_cands[i-1]["final_score"], "Score sorting failed"
        
    return sorted_cands
