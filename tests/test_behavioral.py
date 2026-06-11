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
    has_short_notice_strong_plan_fit, has_reachable_elite_plan_fit,
    full_plan_band_bonus, missing_must_have_buckets, must_have_gap_multiplier,
    must_have_score_ceiling, has_top100_must_have_gap, has_notice_risk,
    has_location_risk, has_true_disqualifier, hard_disqualification_reason,
    is_hard_disqualified, ce_core_delta,
)
from src.runtime_calibration import calibrate_candidate
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
        - (
            W["soft_penalties.zero_eval_mult"]
            * W["soft_penalties.zero_core_retrieval_mult"]
            * W["soft_penalties.zero_vector_hybrid_mult"]
            * W["soft_penalties.multiple_must_have_zero_mult"]
            * W["soft_penalties.adjacent_domain_weak_ir_mult"]
        )
    ) < 1e-9

    cand_cv_strong = {
        "profile_current_title": "Computer Vision Engineer",
        "retrieval_search": 3.0,
        "vector_db_hybrid": 3.0,
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
        - (
            W["soft_penalties.zero_eval_mult"]
            * W["soft_penalties.zero_core_retrieval_mult"]
            * W["soft_penalties.zero_vector_hybrid_mult"]
            * W["soft_penalties.multiple_must_have_zero_mult"]
            * W["soft_penalties.current_consulting_weak_ir_mult"]
        )
    ) < 1e-9

    cand_long_notice = {
        "beh_notice_period_days": 120,
        "eval_framework": 0.0,
    }
    assert has_long_notice_weak_eval(cand_long_notice) is True
    assert abs(
        soft_penalties(cand_long_notice)
        - (
            W["soft_penalties.zero_eval_mult"]
            * W["soft_penalties.long_notice_weak_eval_mult"]
            * W["soft_penalties.long_notice_extreme_mult"]
        )
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

    cand_short_notice_strong_fit = {
        "beh_notice_period_days": 15,
        "beh_location": "Indore",
        "beh_country": "India",
        "beh_willing_to_relocate": False,
        "retrieval_search": 3.0,
        "eval_framework": 2.0,
        "ninety_day_alignment": 0.90,
    }
    assert has_short_notice_strong_plan_fit(cand_short_notice_strong_fit) is True
    assert abs(
        soft_penalties(cand_short_notice_strong_fit)
        - W["soft_penalties.no_reloc_strong_fit_mult"]
    ) < 1e-9

    # Keyword stuffer
    assert soft_penalties({"keyword_stuffer_flag": True}) < 1.0

    short_only = soft_penalties({"title_velocity_flag": True})
    title_bump = soft_penalties({"title_velocity_flag": True, "title_bump_flag": True})
    title_chaser = soft_penalties({"title_velocity_flag": True, "title_bump_flag": True, "title_chaser_flag": True})
    assert title_bump < short_only
    assert title_chaser < title_bump


def test_missing_must_have_gap_multiplier_is_strong():
    complete = {
        "retrieval_search": 3.0,
        "sys_experience_score": 1.0,
        "vector_db_hybrid": 3.0,
        "eval_framework": 2.0,
        "python_coding": 1.0,
    }
    assert missing_must_have_buckets(complete) == []
    assert must_have_gap_multiplier(complete) == 1.0

    rank14_like = {
        "retrieval_search": 3.0,
        "sys_experience_score": 1.0,
        "vector_db_hybrid": 0.0,
        "ltr_reranking": 0.0,
        "eval_framework": 2.0,
        "python_coding": 1.0,
    }
    assert missing_must_have_buckets(rank14_like) == ["vector/hybrid search"]
    assert must_have_gap_multiplier(rank14_like) < W["soft_penalties.zero_vector_hybrid_mult"]
    assert must_have_score_ceiling(rank14_like) == W["behavioral.double_zero_must_have_score_ceiling"]
    assert has_floor_exempt_penalty(rank14_like) is True
    vector_corrob_rescued = dict(rank14_like)
    vector_corrob_rescued["runtime_corroborated_vector_signal"] = 1.0
    assert missing_must_have_buckets(vector_corrob_rescued) == []

    vector_skill_only = dict(rank14_like)
    vector_skill_only["runtime_vector_skill_signal"] = 1.0
    assert missing_must_have_buckets(vector_skill_only) == ["vector/hybrid search"]

    single_gap = dict(complete)
    single_gap["python_coding"] = 0.0
    assert missing_must_have_buckets(single_gap) == ["Python"]
    assert must_have_score_ceiling(single_gap) == W["behavioral.single_must_have_gap_score_ceiling"]
    assert has_top100_must_have_gap(single_gap) is True
    python_rescued = dict(single_gap)
    python_rescued["runtime_career_python_signal"] = 1.0
    assert missing_must_have_buckets(python_rescued) == []
    assert has_top100_must_have_gap(python_rescued) is False
    override = dict(single_gap)
    override["must_have_raw_evidence_override"] = True
    assert has_top100_must_have_gap(override) is False

    multi_gap = {
        "retrieval_search": 0.0,
        "sys_experience_score": 0.0,
        "vector_db_hybrid": 0.0,
        "ltr_reranking": 0.0,
        "eval_framework": 0.0,
        "python_coding": 0.0,
    }
    assert len(missing_must_have_buckets(multi_gap)) == 4
    assert must_have_gap_multiplier(multi_gap) < 0.05
    assert must_have_score_ceiling(multi_gap) == W["behavioral.double_zero_must_have_score_ceiling"]

    eval_rescued = {
        "retrieval_search": 3.0,
        "sys_experience_score": 1.0,
        "vector_db_hybrid": 3.0,
        "eval_framework": 0.0,
        "runtime_career_eval_signal": 1.0,
        "python_coding": 1.0,
    }
    assert missing_must_have_buckets(eval_rescued) == []

    eval_adjacent_rescued = {
        "retrieval_search": 3.0,
        "sys_experience_score": 1.0,
        "vector_db_hybrid": 3.0,
        "eval_framework": 0.0,
        "runtime_career_eval_adjacent_signal": 1.0,
        "python_coding": 1.0,
    }
    assert missing_must_have_buckets(eval_adjacent_rescued) == []


def test_runtime_vector_corroboration_requires_concrete_tool():
    cand = {}
    vague_vector_profile = {
        "profile": {
            "current_title": "Senior AI Engineer",
            "summary": "Built shipped search and ranking systems for real users.",
        },
        "career_history": [
            {
                "title": "Senior AI Engineer",
                "company": "ProductCo",
                "industry": "Internet",
                "description": "Owned production search ranking with evaluation dashboards and real users.",
            }
        ],
        "skills": [{"name": "Vector Representations"}],
    }
    calibrated = calibrate_candidate(cand, vague_vector_profile)
    assert calibrated["runtime_vector_skill_signal"] == 1.0
    assert calibrated["runtime_concrete_vector_tool_signal"] == 0.0
    assert calibrated["runtime_corroborated_vector_signal"] == 0.0

    concrete = calibrate_candidate({}, {
        **vague_vector_profile,
        "skills": [{"name": "Qdrant"}, {"name": "Sentence Transformers"}],
    })
    assert concrete["runtime_concrete_vector_tool_signal"] == 1.0
    assert concrete["runtime_corroborated_vector_signal"] == 1.0


def test_runtime_full_plan_current_role_cannot_be_carried_by_previous_role():
    profile = {
        "profile": {
            "current_title": "ML Engineer",
            "summary": "Applied ML engineer.",
        },
        "career_history": [
            {
                "title": "ML Engineer",
                "company": "CurrentCo",
                "industry": "SaaS",
                "description": "Owned churn prediction dashboards and MLflow model monitoring.",
            },
            {
                "title": "Search Engineer",
                "company": "PreviousCo",
                "industry": "Internet",
                "description": (
                    "Owned production hybrid retrieval and ranking with FAISS, "
                    "evaluation harnesses, A/B tests, and real users."
                ),
            },
        ],
        "skills": [{"name": "FAISS"}, {"name": "Python"}],
    }
    calibrated = calibrate_candidate({}, profile)
    assert calibrated["runtime_current_retrieval_signal"] == 0.0
    assert calibrated["runtime_full_plan_signal"] <= W["behavioral.runtime_full_plan_current_role_cap"]


def test_new_penalties_are_moderate_for_two_mild_flags():
    base = {
        "retrieval_search": 3.0,
        "sys_experience_score": 1.0,
        "vector_db_hybrid": 3.0,
        "eval_framework": 2.0,
        "python_coding": 1.0,
    }
    mild_two_flags = dict(base, adjacent_career_ratio=0.51, core_score=86.0, ce_score=50.0)
    penalty = soft_penalties(mild_two_flags)
    expected = W["soft_penalties.adjacent_career_half_mult"] * W["soft_penalties.ce_delta_moderate_mult"]
    assert penalty == pytest.approx(expected)
    assert penalty > 0.80

    severe_stack = dict(
        base,
        adjacent_career_ratio=0.66,
        core_score=90.0,
        ce_score=48.0,
        beh_notice_period_days=120,
        beh_location="Trivandrum",
        beh_country="India",
        beh_willing_to_relocate=False,
    )
    assert soft_penalties(severe_stack) < penalty


def test_true_disqualifiers_get_kill_switch_score():
    ref = date(2026, 6, 1)
    cand = {
        "final_phase4_score": 90.0,
        "keyword_stuffer_flag": True,
    }
    assert has_true_disqualifier(cand) is True
    assert compute_final_score(cand, ref) == round(90.0 * W["behavioral.disqualifier_multiplier"], 6)


def test_must_have_and_logistics_exclusion_gates():
    ref = date(2026, 6, 1)
    clean = {
        "final_phase4_score": 90.0,
        "retrieval_search": 3.0,
        "sys_experience_score": 1.0,
        "vector_db_hybrid": 3.0,
        "eval_framework": 2.0,
        "python_coding": 1.0,
        "beh_notice_period_days": 30,
        "beh_country": "India",
        "beh_location": "Pune",
    }
    missing_python = dict(clean)
    missing_python["python_coding"] = 0.0
    assert compute_final_score(missing_python, ref) == round(90.0 * W["behavioral.must_have_missing_multiplier"], 6)
    assert is_hard_disqualified(missing_python) is True
    assert "missing true must-have evidence" in hard_disqualification_reason(missing_python)

    long_notice = dict(clean)
    long_notice["beh_notice_period_days"] = W["behavioral.long_notice_exclusion_days"]
    assert has_notice_risk(long_notice) is True
    assert hard_disqualification_reason(long_notice) == ""
    assert compute_final_score(long_notice, ref) > 10.0
    assert compute_final_score(long_notice, ref) < compute_final_score(clean, ref)

    bad_location = dict(clean)
    bad_location["beh_location"] = "Kolkata"
    bad_location["beh_willing_to_relocate"] = False
    assert has_location_risk(bad_location) is True
    assert hard_disqualification_reason(bad_location) == ""
    assert compute_final_score(bad_location, ref) > 10.0
    assert compute_final_score(bad_location, ref) < compute_final_score(clean, ref)

    unknown_city = dict(clean)
    unknown_city["beh_location"] = ""
    unknown_city["beh_willing_to_relocate"] = False
    assert has_location_risk(unknown_city) is False


def test_ce_core_delta_helper():
    assert ce_core_delta({"core_score": 84.34, "ce_score": 33.32}) == pytest.approx(51.02)

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
    assert has_floor_exempt_penalty({"title_chaser_flag": True}) is True
    assert has_floor_exempt_penalty({}) is False

def test_compute_final_score():
    ref = date(2026, 6, 1)
    
    # Impossible flag
    cand_imp = {
        "final_phase4_score": 80.0,
        "impossible_flag": True
    }
    assert abs(compute_final_score(cand_imp, ref) - 80.0 * W["behavioral.disqualifier_multiplier"]) < 1e-9

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


def test_runtime_full_plan_bonus_is_capped_by_must_have_gap():
    ref = date(2026, 6, 1)
    base = {
        "final_phase4_score": 80.0,
        "runtime_full_plan_signal": W["behavioral.runtime_full_plan_min"],
        "beh_recruiter_response_rate": 0.90,
        "beh_notice_period_days": 30,
        "beh_location": "Pune",
        "beh_country": "India",
        "beh_willing_to_relocate": True,
        "seniority_score": 1.0,
        "writing_signal": 1.0,
        "retrieval_search": 3.0,
        "sys_experience_score": 1.0,
        "vector_db_hybrid": 3.0,
        "eval_framework": 2.0,
        "python_coding": 1.0,
        "career_ir_density": 1.0,
    }
    gap = dict(base)
    gap["vector_db_hybrid"] = 0.0
    gap["ltr_reranking"] = 0.0
    assert compute_final_score(gap, ref) < compute_final_score(base, ref)
    assert compute_final_score(gap, ref) <= W["behavioral.double_zero_must_have_score_ceiling"]


def test_reachable_elite_plan_bonus_is_narrow():
    ref = date(2026, 6, 1)
    base = {
        "final_phase4_score": 70.0,
        "retrieval_search": 3.0,
        "vector_db_hybrid": 3.0,
        "eval_framework": 2.0,
        "ltr_reranking": 3.0,
        "career_eval_density": 1.0,
        "ninety_day_alignment": 0.95,
        "beh_recruiter_response_rate": 0.86,
        "beh_notice_period_days": 60,
        "beh_willing_to_relocate": True,
        "beh_location": "Chennai",
        "beh_country": "India",
        "ce_score": W["behavioral.reachable_elite_ce_threshold"],
        "final_phase4_score": W["behavioral.reachable_elite_phase4_max"],
        "writing_signal": 1.0,
        "seniority_score": 1.0,
        "target_skill_duration_contradiction": 1,
    }
    assert has_reachable_elite_plan_fit(base) is True
    low_ce = dict(base)
    low_ce["ce_score"] = W["behavioral.reachable_elite_ce_threshold"] - 0.1
    assert has_reachable_elite_plan_fit(low_ce) is False
    long_notice = dict(base)
    long_notice["beh_notice_period_days"] = 90
    assert has_reachable_elite_plan_fit(long_notice) is False
    already_top = dict(base)
    already_top["final_phase4_score"] = W["behavioral.reachable_elite_phase4_max"] + 0.1
    assert has_reachable_elite_plan_fit(already_top) is False
    assert compute_final_score(base, ref) > compute_final_score(low_ce, ref)


def test_full_plan_band_bonus_is_bounded_to_review_band():
    base = {
        "final_phase4_score": 79.0,
        "retrieval_search": 2.0,
        "vector_db_hybrid": 2.0,
        "eval_framework": 2.0,
        "ltr_reranking": 3.0,
        "ninety_day_alignment": 0.95,
        "beh_recruiter_response_rate": 0.70,
        "beh_notice_period_days": 30,
        "beh_willing_to_relocate": True,
        "beh_country": "India",
        "ce_score": 82.0,
        "target_skill_duration_contradiction": 2,
    }
    assert full_plan_band_bonus(base, 72.0) == W["behavioral.full_plan_band_reloc_bonus"]
    assert full_plan_band_bonus(base, 80.0) == 0.0
    weak_ce = dict(base)
    weak_ce["ce_score"] = W["behavioral.full_plan_band_reloc_ce_min"] - 0.1
    assert full_plan_band_bonus(weak_ce, 72.0) == 0.0


def test_full_plan_band_bonus_requires_eval_density_for_mild_no_reloc():
    base = {
        "final_phase4_score": 82.0,
        "retrieval_search": 2.0,
        "vector_db_hybrid": 2.0,
        "eval_framework": 2.0,
        "ltr_reranking": 3.0,
        "ninety_day_alignment": 0.95,
        "beh_recruiter_response_rate": 0.80,
        "beh_notice_period_days": 60,
        "beh_willing_to_relocate": False,
        "beh_country": "India",
        "ce_score": 75.0,
        "career_eval_density": 0.79,
        "target_skill_duration_contradiction": 0,
    }
    assert full_plan_band_bonus(base, 72.0) == 0.0
    base["career_eval_density"] = W["behavioral.full_plan_band_no_reloc_mild_career_eval_min"]
    assert full_plan_band_bonus(base, 72.0) == W["behavioral.full_plan_band_no_reloc_mild_bonus"]


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
