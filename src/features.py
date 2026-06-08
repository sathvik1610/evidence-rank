"""
src/features.py — Phase 3: Candidate Feature Extraction (Bucket A / B / C)

Called by:
  - preprocess.py (Phase 1c): offline, on top-5000 retrieved candidates
  - rank.py (Phase 3 fallback): on <=100 candidates in sandbox mode

Rules:
  - Pure Python + regex only. No NLP library imports (no SpaCy, no transformers).
  - All pattern matching is case-insensitive.
  - Missing fields default to empty string / empty list / sentinel values.
  - Sentinel values: -1 (no data), "UNKNOWN" (missing string), 0.5 (neutral default for floats)
  - No candidate is silently dropped from this function.
"""

import re
import json
from typing import Dict, Any, List, Tuple

import constants
from src.weights import W
from src.jd_intelligence import build_feature_contract

# ---------------------------------------------------------------------------
# Evidence Pattern Sets
# ---------------------------------------------------------------------------

JD_FEATURE_CONTRACT = build_feature_contract(constants.JD_CONTRACT_YAML)

RETRIEVAL_PATTERNS = JD_FEATURE_CONTRACT["retrieval_patterns"]
RANKING_PATTERNS = JD_FEATURE_CONTRACT["ranking_patterns"]
RECOMMENDATION_PATTERNS = JD_FEATURE_CONTRACT["recommendation_patterns"]
EVALUATION_PATTERNS = JD_FEATURE_CONTRACT["target_skills"]["eval_framework"]
PRODUCTION_PATTERNS = JD_FEATURE_CONTRACT["production_patterns"]
SHIPPER_TERMS = JD_FEATURE_CONTRACT["shipper_terms"]
RESEARCHER_TERMS = JD_FEATURE_CONTRACT["researcher_terms"]
SYSTEM_SEMANTICS_PATTERNS = JD_FEATURE_CONTRACT["system_semantics_patterns"]

# ---------------------------------------------------------------------------
# Bucket A — Target Skills
# ---------------------------------------------------------------------------

TARGET_SKILLS: Dict[str, List[str]] = JD_FEATURE_CONTRACT["target_skills"]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _build_career_text(candidate: Dict[str, Any]) -> str:
    """Concatenate all career description text into a single string for regex search."""
    parts = []
    for role in candidate.get("career_history", []):
        title = role.get("title", "")
        desc = role.get("description", "")
        if title:
            parts.append(title)
        if desc:
            parts.append(desc)
    return " ".join(parts)


def _skill_names_lower(candidate: Dict[str, Any]) -> List[str]:
    """Return a list of lowercase skill names."""
    return [s.get("name", "").lower() for s in candidate.get("skills", [])]


def _get_snippet(career_text: str, match_start: int, context: int = 30) -> str:
    """Extract a short context snippet around a regex match start position."""
    start = max(0, match_start - context)
    end = min(len(career_text), match_start + 60)
    return career_text[start:end].strip()


# ---------------------------------------------------------------------------
# Bucket A — Skill Evidence Scoring
# ---------------------------------------------------------------------------

def score_skill_bucket(
    candidate: Dict[str, Any],
    career_text: str,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Score each TARGET_SKILL bucket 0–3:
      0 = not mentioned anywhere
      1 = skill in skills[] only (no career description evidence)
      2 = mentioned in career description (project-level evidence)
      3 = mentioned in career description WITH production/scale signals

    Returns:
      scores:   {bucket_name: float}    — primary feature scores
      snippets: {bucket_name: str}      — best evidence snippet per bucket (for Phase 6 reasons)
    """
    scores: Dict[str, float] = {}
    snippets: Dict[str, str] = {}
    skill_names = _skill_names_lower(candidate)
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})

    for bucket_name, keywords in TARGET_SKILLS.items():
        # Check skills[] section (score=1 signal)
        skill_mentioned = any(
            re.search(kw, sn, re.IGNORECASE)
            for sn in skill_names
            for kw in keywords
        )

        # Search career descriptions for each keyword
        career_evidence_snippets: List[str] = []
        for kw in keywords:
            matches = list(re.finditer(kw, career_text, re.IGNORECASE))
            if matches:
                snippet = _get_snippet(career_text, matches[0].start())
                career_evidence_snippets.append(snippet)

        # Check if production signals appear near the found evidence
        # (localised to snippets, not entire career text, to avoid false positives)
        has_production = any(
            re.search(prod_p, snippet, re.IGNORECASE)
            for prod_p in PRODUCTION_PATTERNS
            for snippet in career_evidence_snippets
        )

        # Assign score
        if career_evidence_snippets and has_production:
            score = 3.0
        elif career_evidence_snippets:
            score = 2.0
        elif skill_mentioned:
            score = 1.0
        else:
            score = 0.0

        # Assessment bonus: verified high score for a matching skill
        for s in candidate.get("skills", []):
            if any(re.search(kw, s.get("name", ""), re.IGNORECASE) for kw in keywords):
                asc = assessment_scores.get(s["name"])
                if asc is not None and asc >= W["scoring.assessment_bonus_threshold"] and score >= 1:
                    score = min(score + W["scoring.assessment_bonus_value"], 3.0)
                    break  # Apply bonus once per bucket

        scores[bucket_name] = score
        snippets[bucket_name] = career_evidence_snippets[0] if career_evidence_snippets else ""

    return scores, snippets


# ---------------------------------------------------------------------------
# Bucket B — Career Quality Scoring
# ---------------------------------------------------------------------------

_OWNERSHIP_PATTERNS = JD_FEATURE_CONTRACT["ownership_patterns"]


def compute_product_ratio(candidate: Dict[str, Any]) -> float:
    """
    Fraction of total career months at product companies (vs consulting/services).
    Returns:
      0.0 = entire career at consulting
      1.0 = entirely at product companies
      0.5 = fallback if no career data
    """
    career = candidate.get("career_history", [])
    total_months = sum(r.get("duration_months", 0) or 0 for r in career)
    if total_months == 0:
        return 0.5  # Fail open: no career data → neutral

    consulting_months = sum(
        r.get("duration_months", 0) or 0 for r in career
        if any(firm in r.get("company", "").lower() for firm in constants.CONSULTING_FIRMS)
        or r.get("industry", "").lower() in constants.CONSULTING_INDUSTRIES
    )
    return round(1.0 - (consulting_months / total_months), 4)


def score_career_quality(
    candidate: Dict[str, Any],
    career_text: str,
    flags: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute Bucket B features: product ratio, deployment signals, shipper/researcher ratio,
    recency, depth, writing quality, and the composite product_builder_score.

    flags: candidate flags from Phase 1f (consulting_only, research_only, wrong_domain)
    """
    career = candidate.get("career_history", [])

    # Product ratio: fraction of career at product companies (also used in product_builder)
    product_ratio = flags.get("product_ratio", compute_product_ratio(candidate))

    # Deploy signal: count of unique production signals in full career text
    deploy_count = sum(
        1 for p in PRODUCTION_PATTERNS
        if re.search(p, career_text, re.IGNORECASE)
    )
    deploy_signal = min(deploy_count / 5.0, 1.0)

    # Experience recency: is the most recent role in a relevant domain?
    # Career history is typically ordered most-recent first (index 0 = current).
    recent_role = career[0] if career else {}
    recent_desc = recent_role.get("description", "")
    recent_relevant = any(
        re.search(p, recent_desc, re.IGNORECASE)
        for p in RETRIEVAL_PATTERNS + RANKING_PATTERNS
    )
    experience_recency = 1.0 if recent_relevant else 0.5

    # Depth signal: retrieval/ranking work appears across multiple roles
    roles_with_retrieval = sum(
        1 for role in career
        if any(
            re.search(p, role.get("description", ""), re.IGNORECASE)
            for p in RETRIEVAL_PATTERNS + RANKING_PATTERNS
        )
    )
    depth_signal = min(roles_with_retrieval / 2.0, 1.0)

    # Search/Ranking/Recommendation System Experience Score
    has_sys_evidence = any(
        re.search(p, career_text, re.IGNORECASE)
        for p in (
            RETRIEVAL_PATTERNS + RANKING_PATTERNS +
            RECOMMENDATION_PATTERNS + SYSTEM_SEMANTICS_PATTERNS
        )
    )
    has_sys_production = has_sys_evidence and any(
        re.search(p, career_text, re.IGNORECASE) for p in PRODUCTION_PATTERNS
    )
    sys_experience_score = 1.0 if has_sys_production else (0.5 if has_sys_evidence else 0.0)

    # Shipper vs Researcher ratio
    shipper_count = sum(
        1 for p in SHIPPER_TERMS if re.search(p, career_text, re.IGNORECASE)
    )
    researcher_count = sum(
        1 for p in RESEARCHER_TERMS if re.search(p, career_text, re.IGNORECASE)
    )
    total_vocab = shipper_count + researcher_count
    shipper_ratio = shipper_count / total_vocab if total_vocab > 0 else 0.5

    # Writing signal: JD says "we write a lot. If you find writing painful, you'll find this
    # role painful." Average description length as proxy for writing culture.
    descriptions = [r.get("description", "") or "" for r in career]
    avg_desc_len = (
        sum(len(d) for d in descriptions) / len(descriptions)
        if descriptions else 0
    )
    writing_signal = (
        1.00 if avg_desc_len >= 150
        else 0.95 if avg_desc_len >= 60
        else 0.90
    )

    # Ownership signal: founding-team / built-from-scratch language
    ownership_signal = any(
        re.search(p, career_text, re.IGNORECASE) for p in _OWNERSHIP_PATTERNS
    )

    # Product Builder Score composite — [0, 1]
    product_builder_score = (
        0.35 * product_ratio +
        0.30 * deploy_signal +
        0.20 * shipper_ratio +
        0.15 * (1.0 if ownership_signal else 0.0)
    )
    # Disqualifier multipliers (consulting/research backgrounds penalised)
    if flags.get("consulting_only"):
        product_builder_score *= JD_FEATURE_CONTRACT["multiplier_values"]["consulting_heavy_soft_penalty"]
    if flags.get("research_only"):
        product_builder_score *= JD_FEATURE_CONTRACT["multiplier_values"]["pure_research_penalty"]
    if flags.get("wrong_domain"):
        product_builder_score *= JD_FEATURE_CONTRACT["multiplier_values"]["computer_vision_trap"]

    return {
        "product_ratio":         product_ratio,
        "deploy_signal":         round(deploy_signal, 4),
        "experience_recency":    experience_recency,
        "depth_signal":          round(depth_signal, 4),
        "shipper_ratio":         round(shipper_ratio, 4),
        "writing_signal":        writing_signal,
        "sys_experience_score":  sys_experience_score,
        "product_builder_score": round(min(product_builder_score, 1.0), 4),
        "ownership_signal":      ownership_signal,
    }


# ---------------------------------------------------------------------------
# Bucket C — JD Fit Gaps
# ---------------------------------------------------------------------------

_EXTERNAL_VALIDATION_TERMS = JD_FEATURE_CONTRACT["external_validation_terms"]
_STOPPED_CODING_TITLES = frozenset(JD_FEATURE_CONTRACT["stopped_coding_titles"])
_HANDS_ON_TITLE_TERMS = frozenset(JD_FEATURE_CONTRACT["hands_on_title_terms"])
_FRAMEWORK_DEMO_TERMS = JD_FEATURE_CONTRACT["framework_demo_terms"]
_PRE_LLM_PRODUCTION_TERMS = JD_FEATURE_CONTRACT["pre_llm_production_terms"]


def score_fit_gaps(
    candidate: Dict[str, Any],
    career_text: str,
    flags: Dict[str, Any],
    bucket_a: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """
    Compute Bucket C: gap flags that down-weight or disqualify.
    No candidate is dropped here — flags are used as multipliers downstream.
    """
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    bucket_a = bucket_a or {}

    # --- Title velocity: avg tenure < 18 months across 3+ roles ---
    # Exclude current role (duration still accumulating)
    past_roles = career[1:] if len(career) > 1 else []
    valid_durations = [
        r.get("duration_months") for r in past_roles
        if r.get("duration_months") is not None
    ]
    if len(past_roles) > 0 and len(valid_durations) == len(past_roles):
        avg_tenure = sum(valid_durations) / len(valid_durations)
        title_velocity_flag = (avg_tenure < 18.0) and (len(career) >= 3)
    else:
        title_velocity_flag = False  # Fail open if durations missing

    # --- Consulting flag (from Phase 1 flags) ---
    consulting_flag = flags.get("consulting_only", False)

    # --- External validation: GitHub, papers, open-source ---
    signals = candidate.get("redrob_signals", {})
    github_score = signals.get("github_activity_score", -1)  # -1 = no GitHub account
    has_external_text = any(
        re.search(p, career_text, re.IGNORECASE) for p in _EXTERNAL_VALIDATION_TERMS
    )
    # Sentinel guard: -1 > 0 is False → safely handled
    external_validation = (github_score > 0) or has_external_text

    # --- Code stopped: high-level title + long YoE (likely stopped coding) ---
    yoe = profile.get("years_of_experience", -1)  # -1 sentinel if missing
    current_title = profile.get("current_title", "UNKNOWN").lower()
    # Sentinel guard: yoe=-1, -1 > 8 is False → not flagged when YoE missing
    code_stopped = (
        yoe > 8
        and not any(t in current_title for t in _HANDS_ON_TITLE_TERMS)
        and any(t in current_title for t in _STOPPED_CODING_TITLES)
    )

    # --- Seniority score: continuous bands aligned with JD_contract.yaml ---
    if yoe == -1:  # Sentinel: missing YoE -> neutral
        seniority_score = 1.0
    else:
        seniority_score = 1.0
        for band in JD_FEATURE_CONTRACT["seniority_bands"]:
            try:
                if float(band["min"]) <= yoe <= float(band["max"]):
                    seniority_score = float(band["multiplier"])
                    break
            except (KeyError, TypeError, ValueError):
                continue

    # --- LangChain-only flag ---
    has_framework_demo = (
        sum(1 for p in _FRAMEWORK_DEMO_TERMS if re.search(p, career_text, re.IGNORECASE)) >= 2
    )
    has_pre_llm_production = any(
        re.search(p, career_text, re.IGNORECASE) for p in _PRE_LLM_PRODUCTION_TERMS
    )
    ai_skill_months = sum(
        s.get("duration_months", 0) or 0
        for s in candidate.get("skills", [])
        if any(kw in s.get("name", "").lower() for kw in _FRAMEWORK_DEMO_TERMS + ["llm", "gpt", "ai"])
    )
    langchain_only_flag = (
        has_framework_demo and not has_pre_llm_production and ai_skill_months < 12
    )

    ai_skill_count = 0
    for skill_name in _skill_names_lower(candidate):
        if any(
            re.search(pattern, skill_name, re.IGNORECASE)
            for patterns in TARGET_SKILLS.values()
            for pattern in patterns
        ):
            ai_skill_count += 1

    has_core_career_evidence = any(
        bucket_a.get(name, 0.0) >= 2.0
        for name in ("retrieval_search", "vector_db_hybrid", "eval_framework", "ltr_reranking")
    )
    keyword_stuffer_flag = (
        ai_skill_count >= 5
        and not has_core_career_evidence
        and not any(t in current_title for t in _HANDS_ON_TITLE_TERMS)
    )

    # --- Closed-source flag: 5+ years without any external validation ---
    # Sentinel guard: yoe=-1, -1 >= 5 is False → not flagged when missing
    closed_source_flag = (yoe >= 5) and not external_validation

    return {
        "title_velocity_flag":  title_velocity_flag,
        "consulting_flag":      consulting_flag,
        "external_validation":  external_validation,
        "code_stopped":         code_stopped,
        "seniority_score":      seniority_score,
        "langchain_only_flag":  langchain_only_flag,
        "keyword_stuffer_flag": keyword_stuffer_flag,
        "closed_source_flag":   closed_source_flag,
    }


# ---------------------------------------------------------------------------
# 90-Day Plan Alignment Score
# ---------------------------------------------------------------------------

def compute_ninety_day_alignment(bucket_a: Dict[str, float], product_ratio: float) -> float:
    """
    Computes alignment with the JD's 90-day onboarding plan:
      Weeks 1-3: Audit BM25 / Retrieval
      Weeks 4-8: Ship v2 ranker (vector DB / hybrid search or LTR/reranking)
      Weeks 9-12: Evaluation framework (NDCG / MRR / A-B)
    """
    m1 = bucket_a.get("retrieval_search", 0) / 3.0
    m2 = max(bucket_a.get("vector_db_hybrid", 0), bucket_a.get("ltr_reranking", 0)) / 3.0
    m3 = bucket_a.get("eval_framework", 0) / 3.0

    readiness = (m1 + m2 + m3) / 3.0

    # Bonus for full plan coverage; penalty for missing milestones
    coverage = sum(1 for m in [m1, m2, m3] if m > 0)
    if coverage == 3:
        readiness = min(readiness + 0.15, 1.0)
    elif coverage == 1:
        readiness = max(readiness - 0.10, 0.0)
    elif coverage == 0:
        readiness = 0.0

    alignment = 0.8 * readiness + 0.2 * product_ratio
    return round(alignment, 4)


# ---------------------------------------------------------------------------
# Behavioral Signal Extraction
# ---------------------------------------------------------------------------

def extract_behavioral(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract behavioral/redrob signals into a flat dict for Phase 5.

    IMPORTANT: days_inactive is NOT computed here.
    The raw last_active_date string is passed through so that rank.py can
    compute (reference_date - last_active_date).days dynamically at rank time.
    This prevents negative inactivity values if the sandbox receives candidates
    with dates newer than the precompute run.

    Sentinel conventions:
      -1    = no data (github_activity_score, offer_acceptance_rate, saved_by_recruiters_30d)
      0.5   = neutral default for response rates
      False = default for booleans
    """
    signals = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})

    return {
        # Availability signals (used in Phase 5 reachability_multiplier)
        "last_active_date":          signals.get("last_active_date", None),
        "open_to_work":              signals.get("open_to_work_flag", False),
        "recruiter_response_rate":   signals.get("recruiter_response_rate", 0.5),  # Default 0.5 (not -1)
        "avg_response_time_hours":   signals.get("avg_response_time_hours", 24.0),
        "notice_period_days":        signals.get("notice_period_days", 60),
        "interview_completion_rate": signals.get("interview_completion_rate", 0.5),

        # Social proof signals (sentinel -1 = no data; safely fails > threshold checks)
        "offer_acceptance_rate":     signals.get("offer_acceptance_rate", -1),
        "github_activity_score":     signals.get("github_activity_score", -1),
        "saved_by_recruiters_30d":   signals.get("saved_by_recruiters_30d", -1),
        "endorsements_received":     signals.get("endorsements_received", -1),
        "applications_submitted_30d": signals.get("applications_submitted_30d", -1),
        "profile_completeness_score": signals.get("profile_completeness_score", -1.0),

        # Verification signals
        "verified_email":     signals.get("verified_email", False),
        "verified_phone":     signals.get("verified_phone", False),
        "linkedin_connected": signals.get("linkedin_connected", False),

        # Location / logistics signals (used in Phase 5 location_modifier)
        "willing_to_relocate":  signals.get("willing_to_relocate", False),
        "preferred_work_mode":  signals.get("preferred_work_mode", "UNKNOWN"),
        "location":             profile.get("location", "UNKNOWN").lower(),
        "country":              profile.get("country", "UNKNOWN").lower(),
    }


# ---------------------------------------------------------------------------
# Main Feature Extraction Entry Point
# ---------------------------------------------------------------------------

def extract_features(
    candidate: Dict[str, Any],
    flags: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run all three buckets (A, B, C) + behavioral extraction for one candidate.

    Args:
        candidate: raw candidate dict from candidates.jsonl
        flags:     Phase 1f flags dict (is_honeypot, is_ghost, product_ratio,
                   consulting_only, research_only, wrong_domain, etc.)

    Returns:
        flat feature dict matching the candidate_features.parquet schema
    """
    career_text = _build_career_text(candidate)

    # Bucket A
    bucket_a, snippets = score_skill_bucket(candidate, career_text)

    # Bucket B
    bucket_b = score_career_quality(candidate, career_text, flags)

    # Bucket C
    bucket_c = score_fit_gaps(candidate, career_text, flags, bucket_a)

    # 90-day alignment
    alignment = compute_ninety_day_alignment(bucket_a, bucket_b["product_ratio"])

    # Behavioral (pass-through for Phase 5)
    behavioral = extract_behavioral(candidate)
    profile = candidate.get("profile", {})

    # Assemble flat output dict (matches candidate_features.parquet schema)
    features: Dict[str, Any] = {
        "candidate_id": candidate.get("candidate_id", "UNKNOWN"),
        # Profile facts for Phase 6 reasoning. These are copied directly from
        # the candidate JSON so explanations can be specific without guessing.
        "profile_current_title": profile.get("current_title", "UNKNOWN"),
        "profile_current_company": profile.get("current_company", "UNKNOWN"),
        "profile_years_of_experience": profile.get("years_of_experience", -1),
        "profile_location": profile.get("location", "UNKNOWN"),
        # Bucket A — evidence scores (0.0–3.0 + 0.5 assessment bonus)
        "retrieval_search":    bucket_a.get("retrieval_search", 0.0),
        "vector_db_hybrid":    bucket_a.get("vector_db_hybrid", 0.0),
        "eval_framework":      bucket_a.get("eval_framework", 0.0),
        "ltr_reranking":       bucket_a.get("ltr_reranking", 0.0),
        "llm_integration":     bucket_a.get("llm_integration", 0.0),
        "python_coding":       bucket_a.get("python_coding", 0.0),
        "distributed_systems": bucket_a.get("distributed_systems", 0.0),
        "hr_tech_exposure":    bucket_a.get("hr_tech_exposure", 0.0),
        # Bucket B — career quality signals
        "product_ratio":          bucket_b["product_ratio"],
        "deploy_signal":          bucket_b["deploy_signal"],
        "experience_recency":     bucket_b["experience_recency"],
        "depth_signal":           bucket_b["depth_signal"],
        "shipper_ratio":          bucket_b["shipper_ratio"],
        "writing_signal":         bucket_b["writing_signal"],
        "sys_experience_score":   bucket_b["sys_experience_score"],
        "product_builder_score":  bucket_b["product_builder_score"],
        "ownership_signal":       bucket_b["ownership_signal"],
        # Bucket C — gap flags
        "title_velocity_flag":  bucket_c["title_velocity_flag"],
        "consulting_flag":       bucket_c["consulting_flag"],
        "external_validation":   bucket_c["external_validation"],
        "code_stopped":          bucket_c["code_stopped"],
        "seniority_score":       bucket_c["seniority_score"],
        "langchain_only_flag":   bucket_c["langchain_only_flag"],
        "keyword_stuffer_flag":  bucket_c["keyword_stuffer_flag"],
        "closed_source_flag":    bucket_c["closed_source_flag"],
        # 90-day plan alignment
        "ninety_day_alignment": alignment,
        # Behavioral (raw signals for Phase 5)
        **{f"beh_{k}": v for k, v in behavioral.items()},
        # Evidence snippets (JSON-serialized for Phase 6 reason generation)
        "snippets_json": json.dumps(snippets, ensure_ascii=False),
        # ---------------------------------------------------------------
        # Phase 1f flags — MUST be forwarded explicitly into the flat dict
        # so that behavioral.py soft_penalties() and compute_final_score()
        # can read them. These fields exist in candidate_flags.parquet but
        # are consumed internally above; without explicit forwarding here
        # they silently default to False/0.0 at rank time.
        # ---------------------------------------------------------------
        "wrong_domain":    flags.get("wrong_domain", False),
        "research_only":   flags.get("research_only", False),
        "consulting_only": flags.get("consulting_only", False),
        "impossible_flag": flags.get("impossible_flag", False),
        "suspicious_flag": flags.get("suspicious_flag", False),
        "honeypot_score":  flags.get("honeypot_score", 0.0),
        "is_ghost":        flags.get("is_ghost", False),
        # contradiction counts (computed in Phase 1f; 0 if not available)
        "contradiction_skill_duration": flags.get("contradiction_skill_duration", 0),
        "contradiction_assessment":     flags.get("contradiction_assessment", 0),
    }

    return features
