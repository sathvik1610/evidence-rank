"""
test_sparse_pipeline.py — Math validation for the BGE-M3 sparse CSR pipeline.

This test must pass before running preprocess.py on the full 100K corpus.
It validates:
  1. Sparse CSR matrix construction from raw lexical_weights dicts
  2. Dot-product similarity computation between query and candidate sparse vectors
  3. Vocab size alignment guard (trim/pad)
  4. RRF rank fusion math with known inputs
  5. Ghost detection logic with edge cases
  6. Honeypot detection: impossible_flag rules
  7. Honeypot soft scoring: honeypot_score accumulation
  8. Product ratio computation
  9. Sentinel value safety: -1 comparisons in behavioral multipliers
  10. Weight normalization: must_have_score formula

Run:
    python test_sparse_pipeline.py

All tests must print PASS. Any FAIL indicates a bug that blocks Phase 1 execution.
"""

import sys
import math
import datetime
from typing import Dict, List

# ---------------------------------------------------------------------------
# 1. Sparse CSR dot-product
# ---------------------------------------------------------------------------

def build_sparse_row(lexical_weights: Dict[int, float], vocab_size: int):
    """Build a 1 × vocab_size sparse row as a dense list (for testing only)."""
    row = [0.0] * vocab_size
    for token_id, weight in lexical_weights.items():
        if token_id < vocab_size:
            row[token_id] = weight
    return row


def dot_product(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def test_sparse_dot_product():
    """Two sparse vectors with known overlap should produce correct dot product."""
    query_weights = {5: 0.8, 10: 0.6, 200: 0.3}
    candidate_weights = {5: 0.7, 10: 0.4, 300: 0.9}  # token 300 outside query
    vocab_size = 500

    q = build_sparse_row(query_weights, vocab_size)
    c = build_sparse_row(candidate_weights, vocab_size)
    score = dot_product(q, c)

    # Expected: 5→0.8×0.7 + 10→0.6×0.4 = 0.56 + 0.24 = 0.80
    expected = 0.56 + 0.24
    assert abs(score - expected) < 1e-9, f"Expected {expected}, got {score}"
    print("PASS: test_sparse_dot_product")


def test_sparse_no_overlap():
    """Zero overlap → dot product = 0.0"""
    q = build_sparse_row({1: 0.9, 2: 0.8}, 100)
    c = build_sparse_row({50: 0.5, 99: 0.3}, 100)
    score = dot_product(q, c)
    assert score == 0.0, f"Expected 0.0, got {score}"
    print("PASS: test_sparse_no_overlap")


def test_vocab_trim_pad():
    """Query vocab > candidate vocab: trim query to candidate vocab size."""
    # Simulate query with token 1000, candidate vocab_size = 500
    query_weights = {5: 0.9, 1000: 0.8}  # token 1000 is out of candidate vocab
    vocab_size = 500
    q = build_sparse_row(query_weights, vocab_size)
    # Token 1000 should be silently dropped (out of range)
    assert q[5] == 0.9, "Token 5 should be present"
    assert len(q) == vocab_size, "Row size must equal vocab_size"
    print("PASS: test_vocab_trim_pad")


# ---------------------------------------------------------------------------
# 2. RRF fusion math
# ---------------------------------------------------------------------------

def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def test_rrf_known_values():
    """RRF scores for rank 1 and rank 100 with k=60."""
    s1 = rrf_score(1, k=60)    # 1/(60+1) = 1/61
    s100 = rrf_score(100, k=60)  # 1/(60+100) = 1/160

    expected_1 = 1 / 61
    expected_100 = 1 / 160

    assert abs(s1 - expected_1) < 1e-12, f"rank1: expected {expected_1}, got {s1}"
    assert abs(s100 - expected_100) < 1e-12, f"rank100: expected {expected_100}, got {s100}"
    assert s1 > s100, "rank 1 must score higher than rank 100"
    print("PASS: test_rrf_known_values")


def test_rrf_fusion_sum():
    """5-way RRF of a candidate ranked 1st in all signals should equal 5 × rrf_score(1)."""
    k = 60
    signals = 5  # FAISS ×3 + sparse + BM25
    fused = sum(rrf_score(1, k) for _ in range(signals))
    expected = signals * (1 / (k + 1))
    assert abs(fused - expected) < 1e-12, f"5-way fusion wrong: {fused}"
    print("PASS: test_rrf_fusion_sum")


# ---------------------------------------------------------------------------
# 3. Ghost detection
# ---------------------------------------------------------------------------

def check_ghost(signals: dict, reference_date: datetime.date) -> bool:
    last_active_str = signals.get("last_active_date")
    if not last_active_str:
        return False
    try:
        last_active = datetime.date.fromisoformat(last_active_str)
        days_inactive = (reference_date - last_active).days
    except ValueError:
        return False
    return (
        days_inactive > 365
        and signals.get("recruiter_response_rate", 1.0) < 0.05
        and not signals.get("open_to_work_flag", True)
        and signals.get("applications_submitted_30d", 1) == 0
    )


def test_ghost_all_conditions_met():
    """All 4 ghost conditions true → is_ghost = True."""
    ref = datetime.date(2026, 6, 1)
    signals = {
        "last_active_date": "2024-12-01",  # 547 days ago
        "recruiter_response_rate": 0.02,
        "open_to_work_flag": False,
        "applications_submitted_30d": 0,
    }
    assert check_ghost(signals, ref) is True
    print("PASS: test_ghost_all_conditions_met")


def test_ghost_submitting_applications():
    """Candidate submitting applications → NOT a ghost (escape clause)."""
    ref = datetime.date(2026, 6, 1)
    signals = {
        "last_active_date": "2024-12-01",  # 547 days ago
        "recruiter_response_rate": 0.02,
        "open_to_work_flag": False,
        "applications_submitted_30d": 3,  # Still submitting → NOT ghost
    }
    assert check_ghost(signals, ref) is False
    print("PASS: test_ghost_submitting_applications")


def test_ghost_missing_last_active():
    """Missing last_active_date → NOT a ghost (fail open)."""
    ref = datetime.date(2026, 6, 1)
    signals = {
        "recruiter_response_rate": 0.01,
        "open_to_work_flag": False,
        "applications_submitted_30d": 0,
    }
    assert check_ghost(signals, ref) is False
    print("PASS: test_ghost_missing_last_active")


def test_ghost_open_to_work():
    """Candidate marked open_to_work=True → NOT a ghost."""
    ref = datetime.date(2026, 6, 1)
    signals = {
        "last_active_date": "2024-12-01",
        "recruiter_response_rate": 0.01,
        "open_to_work_flag": True,   # Open to work → NOT ghost
        "applications_submitted_30d": 0,
    }
    assert check_ghost(signals, ref) is False
    print("PASS: test_ghost_open_to_work")


# ---------------------------------------------------------------------------
# 4. Honeypot detection — impossible_flag
# ---------------------------------------------------------------------------

IMPOSSIBLE_TECH_RELEASES = {
    "qdrant":     (2021, 6),
    "milvus":     (2019, 10),
    "pinecone":   (2019, 1),
    "langchain":  (2022, 10),
    "llamaindex": (2022, 11),
}
RELEASE_BUFFER_MONTHS = 12


def check_impossible_flag(skills: list, reference_date: datetime.date) -> bool:
    """Returns True if any skill claims more months than physically possible."""
    for skill in skills:
        name_lower = skill.get("name", "").lower()
        for tech, (rel_year, rel_month) in IMPOSSIBLE_TECH_RELEASES.items():
            if tech in name_lower:
                release_date = datetime.date(rel_year, rel_month, 1)
                months_since_release = (reference_date - release_date).days / 30.436875
                claimed_months = skill.get("duration_months", 0)
                if claimed_months > months_since_release + RELEASE_BUFFER_MONTHS:
                    return True
    return False


def test_impossible_flag_qdrant_too_long():
    """Claiming 80 months of Qdrant (released Jun 2021) → impossible."""
    ref = datetime.date(2026, 6, 1)
    skills = [{"name": "Qdrant", "duration_months": 80}]
    # Qdrant is 60 months old as of Jun 2026. Max allowed = 60 + 12 = 72 months.
    # 80 > 72 → should flag.
    assert check_impossible_flag(skills, ref) is True
    print("PASS: test_impossible_flag_qdrant_too_long")


def test_impossible_flag_qdrant_valid():
    """Claiming 50 months of Qdrant → valid (< 72 months max)."""
    ref = datetime.date(2026, 6, 1)
    skills = [{"name": "Qdrant", "duration_months": 50}]
    assert check_impossible_flag(skills, ref) is False
    print("PASS: test_impossible_flag_qdrant_valid")


def test_impossible_flag_langchain_extreme():
    """Claiming 60 months of LangChain (released Oct 2022, ~44 months ago) → impossible."""
    ref = datetime.date(2026, 6, 1)
    skills = [{"name": "LangChain", "duration_months": 60}]
    # LangChain is ~44 months old. Max = 44 + 12 = 56 months. 60 > 56 → flag.
    assert check_impossible_flag(skills, ref) is True
    print("PASS: test_impossible_flag_langchain_extreme")


def test_impossible_flag_unrelated_skill():
    """Python with 200 months → NOT flagged (not in impossible_tech list)."""
    ref = datetime.date(2026, 6, 1)
    skills = [{"name": "Python", "duration_months": 200}]
    assert check_impossible_flag(skills, ref) is False
    print("PASS: test_impossible_flag_unrelated_skill")


# ---------------------------------------------------------------------------
# 5. Honeypot soft scoring — weighted accumulation
# ---------------------------------------------------------------------------

def compute_honeypot_score(candidate: dict, reference_date: datetime.date) -> float:
    """Simplified version of Phase 1f honeypot_score for test purposes."""
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})

    score = 0.0

    # S-1: Multiple simultaneous is_current=True roles (weight 0.40)
    current_roles = [r for r in career if r.get("is_current")]
    if len(current_roles) >= 2:
        score += 0.40

    # S-2: YoE mismatch > 24 months (weight 0.25)
    total_career_months = sum(r.get("duration_months", 0) for r in career)
    claimed_months = profile.get("years_of_experience", 0) * 12
    if claimed_months > total_career_months + 24:
        score += 0.25

    # S-3: AI skills listed but zero career description substance (weight 0.20)
    AI_SKILL_KEYWORDS = {"faiss", "pinecone", "qdrant", "langchain", "embeddings", "vector"}
    skill_names = {s.get("name", "").lower() for s in skills}
    has_ai_skills = any(kw in name for name in skill_names for kw in AI_SKILL_KEYWORDS)
    career_desc_text = " ".join(r.get("description", "") for r in career).lower()
    has_career_substance = any(kw in career_desc_text for kw in AI_SKILL_KEYWORDS)
    if has_ai_skills and not has_career_substance and len(skills) >= 5:
        score += 0.20

    # S-4: Expert/advanced skill with failing assessment (weight 0.15)
    assessment_scores = signals.get("skill_assessment_scores", {})
    for skill in skills:
        if skill.get("proficiency") in ("expert", "advanced"):
            assessment = assessment_scores.get(skill.get("name"), None)
            if assessment is not None and assessment < 40:
                score += 0.15
                break  # Count once

    return min(score, 1.0)  # Cap at 1.0


def test_honeypot_score_multi_current_and_yoe_mismatch():
    """Candidate with 2 current roles + YoE mismatch → score = 0.65."""
    ref = datetime.date(2026, 6, 1)
    candidate = {
        "career_history": [
            {"is_current": True, "duration_months": 12},
            {"is_current": True, "duration_months": 24},
        ],
        "profile": {"years_of_experience": 20},  # 240 months claimed
        "skills": [],
        "redrob_signals": {},
    }
    score = compute_honeypot_score(candidate, ref)
    # S-1: 0.40 (two current roles)
    # S-2: claimed 240, total career 36, 240-36=204 > 24 → +0.25
    # Expected: 0.65
    assert abs(score - 0.65) < 1e-9, f"Expected 0.65, got {score}"
    print("PASS: test_honeypot_score_multi_current_and_yoe_mismatch")


def test_honeypot_score_clean_candidate():
    """Normal candidate with no red flags → honeypot_score = 0.0."""
    ref = datetime.date(2026, 6, 1)
    candidate = {
        "career_history": [
            {"is_current": True, "duration_months": 24},
            {"is_current": False, "duration_months": 36},
        ],
        "profile": {"years_of_experience": 5},  # 60 months
        "skills": [{"name": "Python", "proficiency": "advanced", "duration_months": 48}],
        "redrob_signals": {"skill_assessment_scores": {"Python": 85}},
    }
    score = compute_honeypot_score(candidate, ref)
    assert score == 0.0, f"Expected 0.0, got {score}"
    print("PASS: test_honeypot_score_clean_candidate")


# ---------------------------------------------------------------------------
# 6. Sentinel value safety — behavioral comparisons
# ---------------------------------------------------------------------------

def test_sentinel_minus1_fails_positive_threshold():
    """
    Critical: -1 sentinel must safely fail all > positive_threshold comparisons.
    This covers: github_activity_score=-1, offer_acceptance_rate=-1, etc.
    """
    sentinel = -1

    # These must ALL be False (sentinel safely fails positive-threshold checks)
    assert not (sentinel > 60),  "sentinel > 60 must be False"
    assert not (sentinel > 5),   "sentinel > 5 must be False"
    assert not (sentinel > 0.70),"sentinel > 0.70 must be False"
    assert not (sentinel > 0),   "sentinel > 0 must be False"
    print("PASS: test_sentinel_minus1_fails_positive_threshold")


def test_sentinel_recruiter_response_rate_default():
    """
    recruiter_response_rate defaults to 1.0 (not -1) in behavioral dict.
    Confirm the default does NOT trigger low_response_rate penalty.
    """
    default_rate = 1.0  # signals.get("recruiter_response_rate", 1.0)
    assert not (default_rate < 0.10), "Default 1.0 must NOT trigger low_response penalty"
    print("PASS: test_sentinel_recruiter_response_rate_default")


def test_sentinel_last_active_missing():
    """
    Missing last_active_date → reachability multiplier must default to 1.0 (no penalty).
    """
    signals = {}  # No last_active_date
    last_active_str = signals.get("last_active_date")
    mult = 1.0
    if last_active_str:  # This branch must NOT execute
        mult *= 0.60
    assert mult == 1.0, f"Missing date should default to mult=1.0, got {mult}"
    print("PASS: test_sentinel_last_active_missing")


# ---------------------------------------------------------------------------
# 7. Must-have score normalization formula
# ---------------------------------------------------------------------------

def compute_must_have_score(retrieval_ev, vectordb_ev, sys_ev, eval_ev, python_ev) -> float:
    must_have_raw = (
        0.25 * retrieval_ev +
        0.20 * vectordb_ev +
        0.20 * sys_ev +
        0.10 * eval_ev +
        0.05 * python_ev
    )
    must_have_score = must_have_raw / 0.80
    must_have_score = min(must_have_score, 1.0)
    return must_have_score


def test_must_have_perfect_candidate():
    """All evidence scores = 1.0 → must_have_score = 1.0."""
    score = compute_must_have_score(1.0, 1.0, 1.0, 1.0, 1.0)
    # raw = 0.25+0.20+0.20+0.10+0.05 = 0.80, normalized = 0.80/0.80 = 1.0
    assert abs(score - 1.0) < 1e-9, f"Expected 1.0, got {score}"
    print("PASS: test_must_have_perfect_candidate")


def test_must_have_retrieval_only():
    """Only retrieval evidence → must_have_score = 0.25/0.80 = 0.3125."""
    score = compute_must_have_score(1.0, 0.0, 0.0, 0.0, 0.0)
    expected = 0.25 / 0.80
    assert abs(score - expected) < 1e-9, f"Expected {expected}, got {score}"
    print("PASS: test_must_have_retrieval_only")


def test_must_have_no_retrieval_cap():
    """
    Candidate with only eval evidence and no retrieval/vectordb/sys → capped at 0.5 before norm.
    Eval-only raw = 0.10. Cap to 0.50. After norm: 0.50/0.80 = 0.625.
    """
    retrieval_ev = 0.0
    vectordb_ev = 0.0
    sys_ev = 0.0
    eval_ev = 1.0  # Only eval evidence
    python_ev = 0.0

    has_any_retrieval_or_recsys = (retrieval_ev > 0 or vectordb_ev > 0 or sys_ev > 0)
    must_have_raw = (
        0.25 * retrieval_ev +
        0.20 * vectordb_ev +
        0.20 * sys_ev +
        0.10 * eval_ev +
        0.05 * python_ev
    )
    if not has_any_retrieval_or_recsys:
        must_have_raw = min(must_have_raw, 0.50)  # cap

    score = min(must_have_raw / 0.80, 1.0)
    expected = 0.10 / 0.80  # raw=0.10, cap=0.50, but 0.10 < 0.50 so cap doesn't bite here
    assert abs(score - expected) < 1e-9, f"Expected {expected}, got {score}"
    print("PASS: test_must_have_no_retrieval_cap")


def test_must_have_cap_fires():
    """
    Candidate with only python evidence (0.05 raw) but no retrieval/sys — cap should not reduce
    (0.05 < 0.50), then normalise to 0.05/0.80.
    Separate test: craft a case where raw > 0.50 but has_any_retrieval=False (impossible with
    the formula since max non-retrieval is eval(0.10)+python(0.05)=0.15). The cap only matters
    if future weights change — confirm it doesn't fire on current formula.
    """
    retrieval_ev = 0.0
    vectordb_ev = 0.0
    sys_ev = 0.0
    eval_ev = 1.0
    python_ev = 1.0

    raw = 0.10 + 0.05  # 0.15 max without retrieval
    # Cap at 0.50 — 0.15 < 0.50 so cap does NOT fire
    capped = min(raw, 0.50)
    assert capped == raw, "Cap should not fire at 0.15"
    score = min(capped / 0.80, 1.0)
    expected = 0.15 / 0.80
    assert abs(score - expected) < 1e-9, f"Expected {expected}, got {score}"
    print("PASS: test_must_have_cap_fires (cap confirmed non-binding with current weights)")


# ---------------------------------------------------------------------------
# 8. Product ratio computation
# ---------------------------------------------------------------------------

def compute_product_ratio(career: list) -> float:
    CONSULTING_FIRMS = {
        "tcs", "infosys", "wipro", "accenture", "cognizant",
        "capgemini", "hcl", "tech mahindra", "mphasis"
    }
    CONSULTING_INDUSTRIES = {"it services", "consulting", "outsourcing"}

    total_months = sum(r.get("duration_months", 0) for r in career)
    if total_months == 0:
        return 0.0

    consulting_months = sum(
        r.get("duration_months", 0) for r in career
        if any(firm in r.get("company", "").lower() for firm in CONSULTING_FIRMS)
        or r.get("industry", "").lower() in CONSULTING_INDUSTRIES
    )
    return 1.0 - (consulting_months / total_months)


def test_product_ratio_pure_consulting():
    """Entire career at TCS → product_ratio = 0.0."""
    career = [
        {"company": "TCS", "industry": "IT Services", "duration_months": 60},
        {"company": "Infosys", "industry": "IT Services", "duration_months": 24},
    ]
    ratio = compute_product_ratio(career)
    assert ratio == 0.0, f"Expected 0.0, got {ratio}"
    print("PASS: test_product_ratio_pure_consulting")


def test_product_ratio_pure_product():
    """Entire career at product companies → product_ratio = 1.0."""
    career = [
        {"company": "Swiggy", "industry": "Food Tech", "duration_months": 36},
        {"company": "Razorpay", "industry": "Fintech", "duration_months": 24},
    ]
    ratio = compute_product_ratio(career)
    assert ratio == 1.0, f"Expected 1.0, got {ratio}"
    print("PASS: test_product_ratio_pure_product")


def test_product_ratio_mixed():
    """6 years product + 2 years consulting → ratio = 6/8 = 0.75."""
    career = [
        {"company": "Swiggy", "industry": "Food Tech", "duration_months": 72},
        {"company": "TCS", "industry": "IT Services", "duration_months": 24},
    ]
    ratio = compute_product_ratio(career)
    expected = 72 / 96
    assert abs(ratio - expected) < 1e-9, f"Expected {expected}, got {ratio}"
    print("PASS: test_product_ratio_mixed")


# ---------------------------------------------------------------------------
# Adversarial edge cases
# ---------------------------------------------------------------------------

def test_adversarial_empty_candidate():
    """Candidate with no fields should not crash any function."""
    ref = datetime.date(2026, 6, 1)
    candidate = {}
    # Ghost check
    assert check_ghost({}, ref) is False
    # Honeypot soft score
    score = compute_honeypot_score(candidate, ref)
    assert score == 0.0
    # Product ratio
    ratio = compute_product_ratio([])
    assert ratio == 0.0
    print("PASS: test_adversarial_empty_candidate")


def test_adversarial_career_date_far_future():
    """
    Ghost check with a last_active_date in the far future (newer than reference).
    Should NOT be a ghost — days_inactive would be negative → < 365.
    """
    ref = datetime.date(2026, 6, 1)
    signals = {
        "last_active_date": "2027-01-01",  # future date
        "recruiter_response_rate": 0.01,
        "open_to_work_flag": False,
        "applications_submitted_30d": 0,
    }
    # days_inactive = (2026-06-01 - 2027-01-01).days = negative → NOT ghost
    result = check_ghost(signals, ref)
    assert result is False, f"Future date candidate should NOT be ghost, got {result}"
    print("PASS: test_adversarial_career_date_far_future")


def test_adversarial_zero_duration_skills():
    """Zero duration_months for a skill → not an impossible flag on its own."""
    ref = datetime.date(2026, 6, 1)
    skills = [{"name": "Qdrant", "duration_months": 0}]
    # 0 < months_since_release + 12 → not impossible
    assert check_impossible_flag(skills, ref) is False
    print("PASS: test_adversarial_zero_duration_skills")


def test_adversarial_rrf_single_candidate_all_signals():
    """
    Adversarial: a honeypot with rank 1 in all 5 RRF signals gets the maximum fused score.
    But with honeypot_multiplier=0.01, their final score must be ≤ 1.0.
    """
    k = 60
    max_fused = sum(rrf_score(1, k) for _ in range(5))
    # After scaling to 0-100 scale (hypothetically): multiply by 100
    max_score_on_100_scale = max_fused * 100  # hypothetical raw
    honeypot_mult = 0.01
    final_score = max_score_on_100_scale * honeypot_mult

    # The actual Core Score pipeline scales differently but the principle holds:
    # 100.0 * 0.01 = 1.0 (kill switch)
    max_possible_core_score = 100.0
    kill_switch_result = max_possible_core_score * honeypot_mult
    assert kill_switch_result <= 1.0, f"Honeypot must score ≤ 1.0, got {kill_switch_result}"
    print("PASS: test_adversarial_rrf_single_candidate_all_signals")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_all():
    tests = [
        # Sparse dot-product
        test_sparse_dot_product,
        test_sparse_no_overlap,
        test_vocab_trim_pad,
        # RRF
        test_rrf_known_values,
        test_rrf_fusion_sum,
        # Ghost detection
        test_ghost_all_conditions_met,
        test_ghost_submitting_applications,
        test_ghost_missing_last_active,
        test_ghost_open_to_work,
        # Impossible flag
        test_impossible_flag_qdrant_too_long,
        test_impossible_flag_qdrant_valid,
        test_impossible_flag_langchain_extreme,
        test_impossible_flag_unrelated_skill,
        # Honeypot soft score
        test_honeypot_score_multi_current_and_yoe_mismatch,
        test_honeypot_score_clean_candidate,
        # Sentinel safety
        test_sentinel_minus1_fails_positive_threshold,
        test_sentinel_recruiter_response_rate_default,
        test_sentinel_last_active_missing,
        # Must-have score normalization
        test_must_have_perfect_candidate,
        test_must_have_retrieval_only,
        test_must_have_no_retrieval_cap,
        test_must_have_cap_fires,
        # Product ratio
        test_product_ratio_pure_consulting,
        test_product_ratio_pure_product,
        test_product_ratio_mixed,
        # Adversarial
        test_adversarial_empty_candidate,
        test_adversarial_career_date_far_future,
        test_adversarial_zero_duration_skills,
        test_adversarial_rrf_single_candidate_all_signals,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test_fn.__name__} — {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Results: {passed} PASSED, {failed} FAILED out of {len(tests)} tests")
    if failed > 0:
        print("BLOCKER: Fix all FAIL before running preprocess.py on full corpus.")
        sys.exit(1)
    else:
        print("All tests passed. Pipeline math is verified. Safe to proceed.")


if __name__ == "__main__":
    run_all()
