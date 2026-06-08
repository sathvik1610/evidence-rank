import sys
sys.path.insert(0, ".")
from datetime import date
import pytest
from src.behavioral import (
    reachability_multiplier, notice_modifier, location_modifier,
    social_proof_boost, seniority_modifier, soft_penalties,
    has_floor_exempt_penalty, compute_final_score, assign_ranks,
    has_adjacent_domain_weak_ir, has_current_consulting_weak_ir,
    has_long_notice_weak_eval, yoe_floor_modifier, has_bad_logistics_combo,
)
from src.weights import W

def test_reachability_multiplier():
    ref = date(2026, 6, 1)
    
    # Clean candidate (active, responsive)
    cand_clean = {
        "beh_last_active_date": "2026-05-30",
        "beh_open_to_work": True,
        "beh_recruiter_response_rate": 0.90
    }
    assert reachability_multiplier(cand_clean, ref) == 1.0
    
    # Heavy inactive candidate
    cand_inactive = {
        "beh_last_active_date": "2024-05-30", # More than 365 days ago
        "beh_open_to_work": True,
        "beh_recruiter_response_rate": 0.90
    }
    # Expected: 1.0 * inactive_heavy_mult (0.50)
    assert abs(reachability_multiplier(cand_inactive, ref) - W["behavioral.inactive_heavy_mult"]) < 1e-9

    # Responsive passive candidate: do not double-penalize if recruiters can reach them.
    cand_responsive_passive = {
        "beh_last_active_date": "2026-05-30",
        "beh_open_to_work": False,
        "beh_recruiter_response_rate": 0.90
    }
    assert reachability_multiplier(cand_responsive_passive, ref) == 1.0

    # Not open to work candidate with weak response signal.
    cand_not_open = {
        "beh_last_active_date": "2026-05-30",
        "beh_open_to_work": False,
        "beh_recruiter_response_rate": 0.40
    }
    assert abs(reachability_multiplier(cand_not_open, ref) - W["behavioral.not_open_mult"]) < 1e-9

    # Missing open_to_work should fail open, not penalize as explicitly unavailable.
    cand_missing_open_to_work = {
        "beh_last_active_date": "2026-05-30",
        "beh_recruiter_response_rate": 0.90
    }
    assert reachability_multiplier(cand_missing_open_to_work, ref) == 1.0

def test_notice_modifier():
    # Ideal (<= 30 days usually)
    assert notice_modifier(15) == W["behavioral.notice_ideal_mult"]
    
    # Mild (<= 60 days)
    assert notice_modifier(45) == W["behavioral.notice_mild_mult"]
    
    # Moderate (<= 90 days)
    assert notice_modifier(75) == W["behavioral.notice_moderate_mult"]
    
    # Bad (> 90 days)
    assert notice_modifier(100) == W["behavioral.notice_bad_mult"]
    
    # None (default)
    assert notice_modifier(None) == 1.00

def test_location_modifier():
    # Preferred city (Pune/Noida)
    assert location_modifier({"beh_location": "Pune", "beh_country": "India"}) == W["behavioral.location_pune_noida_mult"]
    
    # Welcome city + willing to relocate
    assert location_modifier({"beh_location": "Hyderabad", "beh_country": "India", "beh_willing_to_relocate": True}) == W["behavioral.location_welcome_cities_mult"]
    
    # Welcome city + NOT willing to relocate
    assert location_modifier({"beh_location": "Hyderabad", "beh_country": "India", "beh_willing_to_relocate": False}) == W["behavioral.location_welcome_no_reloc"]
    # Other India + willing
    assert location_modifier({"beh_location": "Bangalore", "beh_country": "India", "beh_willing_to_relocate": True}) == W["behavioral.location_india_willing_mult"]
    
    # Other India + NOT willing
    assert location_modifier({"beh_location": "Bangalore", "beh_country": "India", "beh_willing_to_relocate": False}) == W["behavioral.location_india_no_reloc_mult"]


def test_social_proof_boost():
    # Max boost case
    cand_max = {
        "beh_github_activity_score": 100,
        "beh_saved_by_recruiters_30d": 100,
        "beh_endorsements_received": 100,
        "beh_interview_completion_rate": 1.0,
        "beh_offer_acceptance_rate": 1.0,
        "beh_profile_completeness_score": 100,
        "beh_linkedin_connected": True
    }
    boost = social_proof_boost(cand_max)
    assert boost == W["social_proof.social_proof_max"]

    # Low boost case (only linkedin connected)
    cand_low = {
        "beh_linkedin_connected": True
    }
    assert social_proof_boost(cand_low) == W["social_proof.linkedin_connected_boost"]

def test_seniority_modifier():
    assert seniority_modifier({"seniority_score": 0.90}) == 0.90
    assert seniority_modifier({}) == 1.0

def test_soft_penalties():
    # Clean
    assert soft_penalties({}) == 1.0
    
    # Contradictions
    cand_contra = {
        "contradiction_skill_duration": 1,
        "contradiction_assessment": 1
    }
    expected = 1.0 - (W["soft_penalties.consistency_drop_per_contradiction"] * 2)
    expected = max(W["soft_penalties.consistency_floor"], expected)
    assert abs(soft_penalties(cand_contra) - expected) < 1e-9

    cand_target_single = {"target_skill_duration_contradiction": 1}
    assert abs(
        soft_penalties(cand_target_single)
        - W["soft_penalties.target_skill_duration_one_mult"]
    ) < 1e-9

    cand_target_multi = {"target_skill_duration_contradiction": 2}
    assert abs(
        soft_penalties(cand_target_multi)
        - W["soft_penalties.target_skill_duration_multi_mult"]
    ) < 1e-9

    cand_cv_weak = {
        "profile_current_title": "Computer Vision Engineer",
        "retrieval_search": 1.0,
        "vector_db_hybrid": 1.0,
        "eval_framework": 0.0,
    }
    assert has_adjacent_domain_weak_ir(cand_cv_weak) is True
    assert abs(
        soft_penalties(cand_cv_weak)
        - W["soft_penalties.adjacent_domain_weak_ir_mult"]
    ) < 1e-9

    cand_cv_strong = {
        "profile_current_title": "Computer Vision Engineer",
        "retrieval_search": 2.0,
        "vector_db_hybrid": 2.0,
        "eval_framework": 2.0,
    }
    assert has_adjacent_domain_weak_ir(cand_cv_strong) is False
    assert soft_penalties(cand_cv_strong) == 1.0

    cand_consulting_weak = {
        "profile_current_company": "TCS",
        "retrieval_search": 1.0,
        "vector_db_hybrid": 1.0,
        "eval_framework": 0.0,
        "product_builder_score": 0.44,
    }
    assert has_current_consulting_weak_ir(cand_consulting_weak) is True
    assert abs(
        soft_penalties(cand_consulting_weak)
        - W["soft_penalties.current_consulting_weak_ir_mult"]
    ) < 1e-9

    cand_long_notice = {
        "beh_notice_period_days": 120,
        "eval_framework": 0.0,
    }
    assert has_long_notice_weak_eval(cand_long_notice) is True
    assert abs(
        soft_penalties(cand_long_notice)
        - W["soft_penalties.long_notice_weak_eval_mult"]
    ) < 1e-9
    assert has_long_notice_weak_eval({
        "beh_notice_period_days": W["behavioral.notice_moderate_days"],
        "eval_framework": 0.0,
    }) is True

    cand_bad_logistics = {
        "beh_notice_period_days": W["behavioral.notice_moderate_days"],
        "beh_location": "Kolkata",
        "beh_country": "India",
        "beh_willing_to_relocate": False,
    }
    assert has_bad_logistics_combo(cand_bad_logistics) is False
    assert abs(
        soft_penalties(cand_bad_logistics)
        - (
            W["soft_penalties.long_notice_weak_eval_mult"]
            * W["soft_penalties.no_reloc_outside_preferred_mult"]
        )
    ) < 1e-9

    cand_bad_logistics["beh_notice_period_days"] = W["behavioral.notice_moderate_days"] + 30
    assert has_bad_logistics_combo(cand_bad_logistics) is True
    assert soft_penalties(cand_bad_logistics) < W["soft_penalties.no_reloc_outside_preferred_mult"]

    # Keyword stuffer
    assert soft_penalties({"keyword_stuffer_flag": True}) < 1.0

def test_yoe_floor_modifier():
    assert yoe_floor_modifier({"profile_years_of_experience": 5.0}) == 1.0
    assert yoe_floor_modifier({"profile_years_of_experience": 4.5}) == W["soft_penalties.yoe_below_floor_mult"]
    assert yoe_floor_modifier({"profile_years_of_experience": 3.8}) == W["soft_penalties.yoe_far_below_floor_mult"]

def test_has_floor_exempt_penalty():
    # consulting_only is floor exempt
    assert has_floor_exempt_penalty({"consulting_only": True}) is True
    assert has_floor_exempt_penalty({"target_skill_duration_contradiction": 2}) is False
    assert has_floor_exempt_penalty({
        "profile_current_title": "Computer Vision Engineer",
        "retrieval_search": 1.0,
        "vector_db_hybrid": 1.0,
        "eval_framework": 0.0,
    }) is True
    assert has_floor_exempt_penalty({}) is False

def test_compute_final_score():
    ref = date(2026, 6, 1)
    
    # Impossible flag
    cand_imp = {
        "final_phase4_score": 80.0,
        "impossible_flag": True
    }
    assert abs(compute_final_score(cand_imp, ref) - 80.0 * W["behavioral.honeypot_multiplier"]) < 1e-9

    # Normal computation
    cand_normal = {
        "final_phase4_score": 80.0,
        "beh_last_active_date": "2026-05-30",
        "beh_open_to_work": True,
        "beh_recruiter_response_rate": 0.90,
        "beh_notice_period_days": 15,
        "beh_location": "Pune",
        "beh_country": "India",
        "seniority_score": 1.0,
        "writing_signal": 1.0,
        "ninety_day_alignment": 0.5,
        "beh_linkedin_connected": True
    }
    
    score = compute_final_score(cand_normal, ref)
    assert score > 0.0
    assert score <= 120.0

def test_assign_ranks():
    cands = [
        {"candidate_id": "A", "final_score": 75.0},
        {"candidate_id": "B", "final_score": 90.0},
        {"candidate_id": "C", "final_score": 85.0}
    ]
    ranked = assign_ranks(cands)
    assert ranked[0]["candidate_id"] == "B"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["candidate_id"] == "C"
    assert ranked[1]["rank"] == 2
    assert ranked[2]["candidate_id"] == "A"
    assert ranked[2]["rank"] == 3
