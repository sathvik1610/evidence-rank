"""
src/explainer.py — Phase 6: Reason Generation

Generates safe, hallucination-free reasoning strings for the final submission.
Follows the Phase 6 architecture: Evidence-driven lead selection based on the
candidate's strongest domain, supported by snippet extraction, and ranking-aware
gap acknowledgment.
"""

import json

STRENGTH_DOMAINS = [
    ("retrieval_search", "Retrieval Systems"),
    ("sys_experience_score", "Recommendation and Ranking Systems"),
    ("vector_db_hybrid", "Vector Search Infrastructure"),
    ("ltr_reranking", "Learning-to-Rank and Reranking"),
    ("eval_framework", "Evaluation Metrics"),
    ("product_builder_score", "Product ML & Scaling"),
    ("python_coding", "Python Engineering"),
    ("llm_integration", "LLM Integration & RAG")
]

LEAD_TEMPLATES_STRONG = {
    "retrieval_search": "Strong evidence in Retrieval Systems, specifically: '{snippet}'.",
    "sys_experience_score": "Significant experience building Recommendation and Ranking Systems.",
    "vector_db_hybrid": "Deep expertise in Vector Search Infrastructure, notably: '{snippet}'.",
    "ltr_reranking": "Strong evidence in Learning-to-Rank and Reranking, specifically: '{snippet}'.",
    "eval_framework": "Rigorous focus on Evaluation Metrics and testing methodologies: '{snippet}'.",
    "product_builder_score": "Proven track record in Product ML & Scaling across engineering teams.",
    "python_coding": "Solid foundation in Python Engineering: '{snippet}'.",
    "llm_integration": "Strong evidence of production LLM Integration: '{snippet}'."
}

LEAD_TEMPLATES_WEAK = {
    "retrieval_search": "Profile-text evidence in Retrieval Systems.",
    "sys_experience_score": "Profile-text evidence in Recommendation and Ranking Systems.",
    "vector_db_hybrid": "Profile-text evidence in Vector Search Infrastructure.",
    "ltr_reranking": "Profile-text evidence in Learning-to-Rank and Reranking.",
    "eval_framework": "Profile-text evidence in Evaluation Metrics and testing methodologies.",
    "product_builder_score": "Product-company ML experience with some scaling/ownership signal.",
    "python_coding": "Profile-text evidence in Python Engineering.",
    "llm_integration": "Profile-text evidence in LLM Integration or RAG."
}

def get_90day_milestone(domain_key: str) -> str:
    if domain_key in ("retrieval_search", "sys_experience_score"):
        return "the Weeks 1-3 audit (BM25/Retrieval)"
    elif domain_key in ("vector_db_hybrid", "ltr_reranking", "product_builder_score", "llm_integration"):
        return "the Weeks 4-8 mandate (Ship v2 Hybrid Ranker)"
    elif domain_key == "eval_framework":
        return "the Weeks 9-12 mandate (Build Evaluation Framework)"
    return "early engineering milestones"

def get_largest_concern(cand: dict) -> str:
    # Ranked by severity/impact
    if cand.get("impossible_flag", False) or cand.get("suspicious_flag", False):
        return "Profile flagged as highly suspicious or containing impossible timelines."
    if cand.get("research_only", False):
        return "Research-heavy background with limited production ML exposure."
    if cand.get("wrong_domain", False):
        return "Primary experience is outside NLP/Search domains."
    if cand.get("langchain_only_flag", False):
        return "Heavy reliance on LLM wrappers (LangChain) without deep ML infrastructure evidence."
    if cand.get("consulting_flag", False) or cand.get("consulting_only", False):
        return "Career arc leans heavily toward consulting rather than core product ownership."
    if cand.get("code_stopped", False):
        return "Seniority indicates candidate may have shifted away from hands-on coding."
    if cand.get("title_velocity_flag", False):
        return "Frequent title changes noted across recent roles."
    return ""


def _profile_prefix(cand: dict) -> str:
    title = str(cand.get("profile_current_title") or "").strip()
    company = str(cand.get("profile_current_company") or "").strip()
    yoe = cand.get("profile_years_of_experience", -1)

    parts = []
    if title and title != "UNKNOWN":
        parts.append(title)
    if company and company != "UNKNOWN":
        parts.append(f"at {company}")
    try:
        yoe_float = float(yoe)
        if yoe_float >= 0:
            parts.append(f"with {yoe_float:.1f} years of experience")
    except (TypeError, ValueError):
        pass

    return " ".join(parts)



# Location acceptability list (mirrors JD + location_tier_multiplier in weights.yaml)
_PREFERRED_LOCATIONS = {
    "pune", "noida", "gurgaon", "gurugram", "delhi", "new delhi",
    "faridabad", "ghaziabad", "delhi ncr",
}
_WELCOME_LOCATIONS = {"hyderabad", "mumbai"}


def _location_concern(cand: dict) -> str:
    """Return a location concern string if the candidate is outside the JD-listed cities."""
    loc = (cand.get("beh_location") or cand.get("profile_location") or "").lower()
    country = (cand.get("beh_country") or "").lower()
    if country and country not in ("india", ""):
        return f"Located outside India ({loc.title()}) — case-by-case per JD."
    for city in _PREFERRED_LOCATIONS:
        if city in loc:
            return ""  # Preferred city — no concern
    for city in _WELCOME_LOCATIONS:
        if city in loc:
            return ""  # Welcome city — no concern
    if loc and loc not in ("unknown", ""):
        return f"Located in {loc.title()}, outside the JD's preferred cities (Pune/Noida/Delhi NCR/Hyderabad/Mumbai)."
    return ""


def _behavioral_detail(cand: dict) -> str:
    """One-line summary of the strongest behavioral signal (positive or negative)."""
    rrr = cand.get("beh_recruiter_response_rate", -1.0)
    notice = cand.get("beh_notice_period_days")
    parts = []
    if rrr >= 0:
        if rrr >= 0.75:
            parts.append(f"strong recruiter response rate ({int(rrr * 100)}%)")
        elif rrr < 0.40:
            parts.append(f"low recruiter response rate ({int(rrr * 100)}%)")
    if notice is not None:
        if notice == 0:
            parts.append("immediately available")
        elif notice <= 30:
            parts.append(f"{notice}-day notice")
        elif notice <= 60:
            parts.append(f"{notice}-day notice (buyout possible)")
        else:
            parts.append(f"{notice}-day notice (significant barrier)")
    return "; ".join(parts)


def _eval_gap_concern(cand: dict) -> str:
    """Honest Stage-4 caveat for strong ranking/search profiles missing explicit eval evidence."""
    if (
        cand.get("eval_framework", 0.0) == 0.0
        and cand.get("retrieval_search", 0.0) >= 2.0
        and cand.get("ltr_reranking", 0.0) >= 2.0
    ):
        return "No explicit ranking-evaluation metric evidence surfaced; this is the main technical gap."
    return ""


def generate_reasoning(cand: dict) -> str:
    """
    Evidence-driven lead selection, strictly modulated by rank/score.
    Every string must reference at least one specific fact from the profile
    to pass the Stage 4 variation/hallucination checks.
    """
    try:
        snippets = json.loads(cand.get("snippets_json", "{}"))
    except Exception:
        snippets = {}

    # 1. Rank domains by score to find primary and secondary
    ranked_domains = sorted(
        STRENGTH_DOMAINS,
        key=lambda d: cand.get(d[0], 0.0),
        reverse=True
    )

    primary_key, primary_name = ranked_domains[0]
    secondary_key, secondary_name = ranked_domains[1] if len(ranked_domains) > 1 else (None, None)

    # Tone depends on rank and the actual domain score
    rank = cand.get("rank", 100)
    primary_score = cand.get(primary_key, 0.0)

    # 2. Lead sentence
    # A score >= 3.0 means career description evidence WITH production/scale context.
    # Score 2.0 means mentioned in text. Score 1.0 means just in skills list.
    if primary_score >= 3.0:
        snippet = snippets.get(primary_key, "")
        if snippet and "{snippet}" in LEAD_TEMPLATES_STRONG.get(primary_key, ""):
            clean_snippet = snippet.strip()
            if len(clean_snippet) > 80:
                clean_snippet = clean_snippet[:77] + "..."
            lead = LEAD_TEMPLATES_STRONG[primary_key].format(snippet=clean_snippet)
        else:
            lead = f"Candidate shows strong evidence in {primary_name}."
    elif primary_score >= 2.0:
        snippet = snippets.get(primary_key, "")
        if snippet:
            clean_snippet = snippet.strip()
            if len(clean_snippet) > 80:
                clean_snippet = clean_snippet[:77] + "..."
            lead = f"Profile-text evidence in {primary_name}: '{clean_snippet}'."
        else:
            lead = LEAD_TEMPLATES_WEAK.get(primary_key, f"Profile-text evidence in {primary_name}.")
    elif primary_score > 0.0:
        lead = f"Lists {primary_name} in skills without career context."
    else:
        lead = "Limited direct evidence found for core technical requirements."

    profile_prefix = _profile_prefix(cand)
    if profile_prefix:
        lead = f"{profile_prefix}: {lead}"

    # 3. Support sentence — always include a specific behavioral detail for ranks 26+
    support = ""
    if rank <= 50 and secondary_key and cand.get(secondary_key, 0.0) >= 1.0:
        support = f"Also demonstrates capabilities in {secondary_name}."
    elif rank <= 30:
        milestone = get_90day_milestone(primary_key)
        support = f"Best positioned for {milestone}."

    # For ranks > 30 with no secondary signal, append a behavioral detail so the
    # string is specific and not identical to other templated rows.
    if rank > 30 and not support:
        beh = _behavioral_detail(cand)
        if beh:
            support = f"Behavioral signals: {beh}."

    # 4. Concern sentence — always surfaced for rank > 30
    caveat = ""
    concern_text = get_largest_concern(cand)
    eval_gap = _eval_gap_concern(cand)

    # Location concern — checked independently of JD disqualifiers
    loc_concern = _location_concern(cand)

    if rank > 30:
        concern_parts = []
        if concern_text:
            concern_parts.append(concern_text)
        if eval_gap:
            concern_parts.append(eval_gap)
        if loc_concern:
            concern_parts.append(loc_concern)
        if concern_parts:
            caveat = "Note: " + " ".join(concern_parts)
        elif rank > 70:
            # Must still be specific — append behavioral detail if not already in support
            beh = _behavioral_detail(cand)
            if beh and beh not in support:
                caveat = f"Note: {beh}."
            else:
                # Final fallback: reference the actual score bucket weakness
                weak_bucket = ranked_domains[-1][1]  # The weakest domain
                caveat = f"Note: No evidence of {weak_bucket}; overall evidence density is lower than top-tier candidates."
    else:
        # Ranks 1–30: only surface a location concern if present, otherwise standard concern
        concern_parts = []
        if eval_gap:
            concern_parts.append(eval_gap)
        if loc_concern:
            concern_parts.append(loc_concern)
        elif concern_text:
            concern_parts.append(concern_text)
        if concern_parts:
            caveat = "Note: " + " ".join(concern_parts)

    second_parts = [p for p in [support, caveat] if p]
    if second_parts:
        return f"{lead} {' '.join(second_parts)}"
    return lead
