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
    """Concatenate candidate evidence text into a single string for regex search."""
    parts = []
    profile = candidate.get("profile", {})
    for key in ("current_title", "headline", "summary", "current_industry"):
        value = profile.get(key, "")
        if value:
            parts.append(value)
    for role in candidate.get("career_history", []):
        title = role.get("title", "")
        desc = role.get("description", "")
        if title:
            parts.append(title)
        if desc:
            parts.append(desc)
    return " ".join(parts)


def _normalize_description_for_dedupe(text: str) -> str:
    """Normalize exact repeated role descriptions for evidence de-duplication."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _dedupe_roles_for_semantic_evidence(career: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep role structure intact elsewhere, but avoid counting the same long role
    paragraph as independent retrieval/ranking/eval depth across companies.
    """
    seen_descriptions = set()
    semantic_roles: List[Dict[str, Any]] = []
    for role in career:
        desc = _normalize_description_for_dedupe(role.get("description", ""))
        if len(desc) >= 160:
            if desc in seen_descriptions:
                continue
            seen_descriptions.add(desc)
        semantic_roles.append(role)
    return semantic_roles


def _skill_names_lower(candidate: Dict[str, Any]) -> List[str]:
    """Return a list of lowercase skill names."""
    return [s.get("name", "").lower() for s in candidate.get("skills", [])]


def _get_snippet(career_text: str, match_start: int, context: int = 30) -> str:
    """Extract a short context snippet around a regex match start position."""
    start = max(0, match_start - context)
    end = min(len(career_text), match_start + 60)
    return career_text[start:end].strip()


def _snippet_quality(snippet: str) -> int:
    """Prefer concrete role evidence over generic summary/profile language."""
    text = str(snippet or "").lower()
    score = 0
    concrete_patterns = (
        EVALUATION_PATTERNS
        + RETRIEVAL_PATTERNS
        + RANKING_PATTERNS
        + PRODUCTION_PATTERNS
        + [
            r"\b\d+(?:\.\d+)?\s*(?:m\+|k\+|million|ms|qps|gb|%)\b",
            r"\bndcg@\d+\b",
            r"\bmrr\b",
            r"\brecall@k\b",
            r"\ba/b\b",
            r"\bxgboost\b",
            r"\bbge\b",
            r"\bfaiss\b",
            r"\bpinecone\b",
            r"\bbm25\b",
        ]
    )
    for pattern in concrete_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += 2
    generic_fragments = (
        "strong background",
        "comfortable across",
        "machine learning engineer",
        "applied ai",
        "my academic background",
        "open to senior",
    )
    for fragment in generic_fragments:
        if fragment in text:
            score -= 4
    if any(verb in text for verb in ("built", "owned", "designed", "shipped", "migrated", "deployed", "serving")):
        score += 3
    return score


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

        # Assessment bonus: corroborates career evidence, never upgrades skill-only claims.
        for s in candidate.get("skills", []):
            if any(re.search(kw, s.get("name", ""), re.IGNORECASE) for kw in keywords):
                asc = assessment_scores.get(s["name"])
                if (
                    asc is not None
                    and asc >= W["scoring.assessment_bonus_threshold"]
                    and career_evidence_snippets
                    and score >= 2
                ):
                    score = min(score + W["scoring.assessment_bonus_value"], 3.0)
                    break  # Apply bonus once per bucket

        scores[bucket_name] = score
        snippets[bucket_name] = (
            max(career_evidence_snippets, key=_snippet_quality)
            if career_evidence_snippets else ""
        )

    return scores, snippets


# ---------------------------------------------------------------------------
# Bucket B — Career Quality Scoring
# ---------------------------------------------------------------------------

_OWNERSHIP_PATTERNS = JD_FEATURE_CONTRACT["ownership_patterns"]

_ADJACENT_CAREER_PATTERNS = (
    r"\bchurn\b",
    r"\bfraud\b",
    r"\bclassification\b",
    r"\bforecast(?:ing)?\b",
    r"\bmlops\b",
    r"\bkubeflow\b",
    r"\bmlflow\b",
    r"\bchatbot\b",
    r"\bsupport bot\b",
    r"\bbleu\b",
    r"\brouge\b",
    r"\bcomputer vision\b",
    r"\bspeech\b",
    r"\btts\b",
    r"\basr\b",
    r"\byolo\b",
    r"customer[\s._/-]+support[\s._/-]+chatbot",
    r"ticketing[\s._/-]+system",
)

_PRODUCT_IR_PATTERNS = [
    r"\bmatching[\s._/-]+layer\b",
    r"\brelevant[\s._/-]+matches\b",
    r"\blearned[\s._/-]+relevance\b",
    r"\bsearch[\s._/-]+and[\s._/-]+discovery\b",
    r"\bpersonalization[\s._/-]+infrastructure\b",
    r"\boffline[\s._/-]+experimentation[\s._/-]+environment\b",
    r"\bonline[\s._/-]+a/b[\s._/-]+testing[\s._/-]+framework\b",
]

_CORE_ROLE_PATTERNS = (
    RETRIEVAL_PATTERNS +
    RANKING_PATTERNS +
    RECOMMENDATION_PATTERNS +
    EVALUATION_PATTERNS +
    _PRODUCT_IR_PATTERNS
)


def _role_has(patterns: List[str], text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


_CORE_ANCHORED_AB_TEST_RE = re.compile(
    r"\b(?:a/b|ab)[\s._/-]+test(?:ing|s)?\b",
    re.IGNORECASE,
)


def _role_has_eval_evidence(text: str, has_core: bool) -> bool:
    if _role_has(EVALUATION_PATTERNS, text):
        return True
    # The JD explicitly values A/B interpretation for ranking systems, but
    # generic A/B testing appears in many non-ML growth/marketing profiles.
    # Count it only when the same role already has core search/ranking evidence.
    return has_core and bool(_CORE_ANCHORED_AB_TEST_RE.search(text))


def compute_career_density(career: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Measure whether IR/ranking/eval is a sustained career pattern.

    The audits found repeated synthetic templates across unrelated employers.
    A single strong role is still useful evidence, but if most of the candidate's
    months are churn/MLOps/chatbot/CV work, that one template should not dominate
    the top-10 ranking.
    """
    semantic_career = _dedupe_roles_for_semantic_evidence(career)
    total_months = 0.0
    core_months = 0.0
    eval_months = 0.0
    adjacent_months = 0.0
    core_roles = 0
    eval_roles = 0
    adjacent_roles = 0

    for role in career:
        months = float(role.get("duration_months") or 0.0)
        total_months += months if months > 0 else 1.0

    for role in semantic_career:
        role_text = " ".join(
            str(role.get(k) or "")
            for k in ("title", "description", "industry")
        )
        months = float(role.get("duration_months") or 0.0)
        if months <= 0:
            months = 1.0

        has_core = _role_has(_CORE_ROLE_PATTERNS, role_text)
        has_eval = _role_has_eval_evidence(role_text, has_core)
        has_adjacent = _role_has(_ADJACENT_CAREER_PATTERNS, role_text)

        if has_core:
            core_roles += 1
            core_months += months
        if has_eval:
            eval_roles += 1
            eval_months += months

    for role in career:
        role_text = " ".join(
            str(role.get(k) or "")
            for k in ("title", "description", "industry")
        )
        months = float(role.get("duration_months") or 0.0)
        if months <= 0:
            months = 1.0
        if _role_has(_ADJACENT_CAREER_PATTERNS, role_text):
            adjacent_roles += 1
            adjacent_months += months

    role_count = len(career) or 1
    if total_months <= 0:
        total_months = float(role_count)

    career_ir_density = core_months / total_months
    career_eval_density = eval_months / total_months
    adjacent_career_ratio = adjacent_months / total_months
    rag_support_template_risk = (
        adjacent_career_ratio >= 0.65
        and adjacent_roles >= 2
        and career_eval_density < 0.30
    )
    isolated_template_risk = (
        core_roles <= 1
        and adjacent_career_ratio >= 0.35
    ) or (
        career_ir_density < 0.40
        and adjacent_career_ratio >= 0.50
    ) or (
        rag_support_template_risk
    )

    return {
        "career_ir_density": round(career_ir_density, 4),
        "career_eval_density": round(career_eval_density, 4),
        "adjacent_career_ratio": round(adjacent_career_ratio, 4),
        "core_ir_role_count": core_roles,
        "eval_role_count": eval_roles,
        "adjacent_role_count": adjacent_roles,
        "isolated_template_risk": isolated_template_risk,
    }


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


def _company_size_floor(size: str) -> int:
    text = str(size or "").strip().lower().replace(",", "")
    if not text:
        return 0
    if text.endswith("+"):
        text = text[:-1]
    if "-" in text:
        text = text.split("-", 1)[0]
    try:
        return int(float(text))
    except ValueError:
        return 0


def compute_large_product_company_exposure(candidate: Dict[str, Any]) -> float:
    """Small corroborative signal for operating in mature product orgs."""
    career = candidate.get("career_history", [])
    total_months = sum(r.get("duration_months", 0) or 0 for r in career)
    if total_months <= 0:
        return 0.0

    large_product_months = sum(
        r.get("duration_months", 0) or 0
        for r in career
        if _is_productish_role(r)
        and _company_size_floor(r.get("company_size")) >= W["features.large_company_size_floor"]
    )
    return round(large_product_months / total_months, 4)


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
    semantic_career = _dedupe_roles_for_semantic_evidence(career)

    # Product ratio: fraction of career at product companies (also used in product_builder)
    product_ratio = flags.get("product_ratio", compute_product_ratio(candidate))
    large_company_exposure = compute_large_product_company_exposure(candidate)

    # Deploy signal: count of unique production signals in full career text
    deploy_count = sum(
        1 for p in PRODUCTION_PATTERNS
        if re.search(p, career_text, re.IGNORECASE)
    )
    deploy_signal = min(deploy_count / W["features.deploy_signal_hit_cap"], 1.0)

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
        1 for role in semantic_career
        if any(
            re.search(p, role.get("description", ""), re.IGNORECASE)
            for p in RETRIEVAL_PATTERNS + RANKING_PATTERNS
        )
    )
    depth_signal = min(roles_with_retrieval / W["features.depth_signal_role_cap"], 1.0)
    density = compute_career_density(career)

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
        W["features.writing_strong_mult"] if avg_desc_len >= W["features.writing_strong_desc_len"]
        else W["features.writing_ok_mult"] if avg_desc_len >= W["features.writing_ok_desc_len"]
        else W["features.writing_weak_mult"]
    )

    # Ownership signal: founding-team / built-from-scratch language
    ownership_signal = any(
        re.search(p, career_text, re.IGNORECASE) for p in _OWNERSHIP_PATTERNS
    )

    # Product Builder Score composite — [0, 1]
    product_builder_score = (
        W["scoring.product_builder_sub.product_ratio_weight"] * product_ratio +
        W["scoring.product_builder_sub.deploy_signal_weight"] * deploy_signal +
        W["scoring.product_builder_sub.shipper_ratio_weight"] * shipper_ratio +
        W["scoring.product_builder_sub.ownership_weight"] * (1.0 if ownership_signal else 0.0)
    )
    if density["career_ir_density"] >= 0.60 and density["core_ir_role_count"] >= 2:
        product_builder_score += 0.04
    if (
        product_ratio >= W["features.large_product_company_product_ratio_min"]
        and large_company_exposure >= W["features.large_product_company_exposure_min"]
    ):
        product_builder_score += W["features.large_product_company_bonus"]
    if density["isolated_template_risk"]:
        product_builder_score *= 0.90
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
        "large_product_company_exposure": large_company_exposure,
        "ownership_signal":      ownership_signal,
        **density,
    }


# ---------------------------------------------------------------------------
# Bucket C — JD Fit Gaps
# ---------------------------------------------------------------------------

_EXTERNAL_VALIDATION_TERMS = JD_FEATURE_CONTRACT["external_validation_terms"]
_STOPPED_CODING_TITLES = frozenset(JD_FEATURE_CONTRACT["stopped_coding_titles"])
_HANDS_ON_TITLE_TERMS = frozenset(JD_FEATURE_CONTRACT["hands_on_title_terms"])
_FRAMEWORK_DEMO_TERMS = JD_FEATURE_CONTRACT["framework_demo_terms"]
_PRE_LLM_PRODUCTION_TERMS = JD_FEATURE_CONTRACT["pre_llm_production_terms"]

_TITLE_BAND_PATTERNS = (
    (5, re.compile(r"\b(principal|distinguished)\b", re.IGNORECASE)),
    (4, re.compile(r"\bstaff\b", re.IGNORECASE)),
    (3, re.compile(r"\b(lead|tech lead|engineering manager|manager|head|architect)\b", re.IGNORECASE)),
    (2, re.compile(r"\b(senior|sr\.?|sde\s*iii|data scientist iii|ml\s*iii)\b", re.IGNORECASE)),
    (1, re.compile(r"\b(engineer|scientist|developer|analyst|researcher|specialist|sde|mle)\b", re.IGNORECASE)),
)


def _title_seniority_band(title: str | None) -> int:
    text = title or ""
    for band, pattern in _TITLE_BAND_PATTERNS:
        if pattern.search(text):
            return band
    return 0


def _is_productish_role(role: Dict[str, Any]) -> bool:
    text = f"{role.get('industry', '')} {role.get('company', '')}".lower()
    services_terms = (
        "it services",
        "consulting",
        "outsourcing",
        "services",
        "ai services",
        "professional services",
        "system integration",
        "client delivery",
        "staffing",
    )
    return not any(term in text for term in services_terms)


def _compute_title_velocity_signals(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Separate ordinary job hopping from JD-literal title chasing."""
    career = candidate.get("career_history", []) or []
    if len(career) < 2:
        return {
            "avg_past_tenure_months": -1.0,
            "avg_all_tenure_months": -1.0,
            "max_tenure_months": 0.0,
            "current_tenure_months": 0.0,
            "short_role_ratio": 0.0,
            "stable_tenure_flag": False,
            "short_tenure_flag": False,
            "title_bump_flag": False,
            "title_chaser_flag": False,
            "title_company_bump_count": 0,
        }

    past_roles = [r for r in career if not r.get("is_current")]
    valid_durations = [
        r.get("duration_months") for r in past_roles
        if r.get("duration_months") is not None
    ]
    avg_tenure = -1.0
    if past_roles and len(valid_durations) == len(past_roles):
        avg_tenure = float(sum(valid_durations) / len(valid_durations))

    all_durations = [
        r.get("duration_months") or 0
        for r in career
        if r.get("duration_months") is not None
    ]
    avg_all_tenure = float(sum(all_durations) / len(all_durations)) if all_durations else -1.0
    max_tenure = float(max(all_durations)) if all_durations else 0.0
    current_tenure = float(max(
        (r.get("duration_months") or 0)
        for r in career
        if r.get("is_current")
    ) if any(r.get("is_current") for r in career) else 0.0)
    short_role_ratio = (
        sum(1 for months in all_durations if months <= 18) / len(all_durations)
        if all_durations else 0.0
    )

    chron = sorted(career, key=lambda r: str(r.get("start_date") or ""))
    bands = [_title_seniority_band(r.get("title")) for r in chron]
    companies = {
        str(r.get("company") or "").strip().lower()
        for r in career
        if str(r.get("company") or "").strip()
    }
    switches = max(0, len(companies) - 1)
    company_bumps = 0
    literal_jd_ladder = False
    for idx in range(1, len(chron)):
        prev_band = bands[idx - 1]
        curr_band = bands[idx]
        if curr_band > prev_band and chron[idx].get("company") != chron[idx - 1].get("company"):
            company_bumps += 1
        if (prev_band == 2 and curr_band in (4, 5)) or (prev_band == 4 and curr_band == 5):
            literal_jd_ladder = True

    current_band = _title_seniority_band(candidate.get("profile", {}).get("current_title"))
    has_stable_product_tenure = any(
        (r.get("duration_months") or 0) >= 30 and _is_productish_role(r)
        for r in career
    )
    has_stable_tenure = has_stable_product_tenure or current_tenure >= 30 or max_tenure >= 36

    short_tenure = (
        len(career) >= 3
        and short_role_ratio >= 0.50
        and not has_stable_tenure
    )
    title_bump = (
        len(career) >= 3
        and switches >= 2
        and short_role_ratio > 0.50
        and company_bumps >= 1
        and not has_stable_tenure
    )
    title_chaser = (
        len(career) >= 4
        and switches >= 3
        and short_role_ratio > 0.50
        and company_bumps >= 2
        and (current_band >= 4 or literal_jd_ladder)
        and not has_stable_tenure
    )

    return {
        "avg_past_tenure_months": round(avg_tenure, 2),
        "avg_all_tenure_months": round(avg_all_tenure, 2),
        "max_tenure_months": round(max_tenure, 2),
        "current_tenure_months": round(current_tenure, 2),
        "short_role_ratio": round(short_role_ratio, 4),
        "stable_tenure_flag": has_stable_tenure,
        "short_tenure_flag": short_tenure,
        "title_bump_flag": title_bump,
        "title_chaser_flag": title_chaser,
        "title_company_bump_count": company_bumps,
    }


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

    title_signals = _compute_title_velocity_signals(candidate)

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
    has_career_retrieval_eval_production = any(
        re.search(pattern, career_text, re.IGNORECASE)
        for pattern in (
            RETRIEVAL_PATTERNS
            + RANKING_PATTERNS
            + RECOMMENDATION_PATTERNS
            + EVALUATION_PATTERNS
            + PRODUCTION_PATTERNS
        )
    )
    langchain_only_flag = (
        has_framework_demo
        and not has_pre_llm_production
        and not has_career_retrieval_eval_production
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
        "title_velocity_flag":  title_signals["short_tenure_flag"],
        "short_tenure_flag":    title_signals["short_tenure_flag"],
        "title_bump_flag":      title_signals["title_bump_flag"],
        "title_chaser_flag":    title_signals["title_chaser_flag"],
        "avg_past_tenure_months": title_signals["avg_past_tenure_months"],
        "avg_all_tenure_months": title_signals["avg_all_tenure_months"],
        "max_tenure_months": title_signals["max_tenure_months"],
        "current_tenure_months": title_signals["current_tenure_months"],
        "short_role_ratio": title_signals["short_role_ratio"],
        "stable_tenure_flag": title_signals["stable_tenure_flag"],
        "title_company_bump_count": title_signals["title_company_bump_count"],
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
        "open_to_work":              signals.get("open_to_work_flag", None),
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
        "large_product_company_exposure": bucket_b["large_product_company_exposure"],
        "ownership_signal":       bucket_b["ownership_signal"],
        "career_ir_density":      bucket_b["career_ir_density"],
        "career_eval_density":    bucket_b["career_eval_density"],
        "adjacent_career_ratio":  bucket_b["adjacent_career_ratio"],
        "core_ir_role_count":     bucket_b["core_ir_role_count"],
        "eval_role_count":        bucket_b["eval_role_count"],
        "adjacent_role_count":    bucket_b["adjacent_role_count"],
        "isolated_template_risk": bucket_b["isolated_template_risk"],
        # Bucket C — gap flags
        "title_velocity_flag":  bucket_c["title_velocity_flag"],
        "short_tenure_flag":    bucket_c["short_tenure_flag"],
        "title_bump_flag":      bucket_c["title_bump_flag"],
        "title_chaser_flag":    bucket_c["title_chaser_flag"],
        "avg_past_tenure_months": bucket_c["avg_past_tenure_months"],
        "avg_all_tenure_months": bucket_c["avg_all_tenure_months"],
        "max_tenure_months": bucket_c["max_tenure_months"],
        "current_tenure_months": bucket_c["current_tenure_months"],
        "short_role_ratio": bucket_c["short_role_ratio"],
        "stable_tenure_flag": bucket_c["stable_tenure_flag"],
        "title_company_bump_count": bucket_c["title_company_bump_count"],
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
        "target_skill_duration_contradiction": flags.get("target_skill_duration_contradiction", 0),
        "max_target_skill_overclaim_months":   flags.get("max_target_skill_overclaim_months", 0.0),
    }

    return features
