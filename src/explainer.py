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

TOP_RANK_PRIMARY_DOMAINS = (
    "retrieval_search",
    "sys_experience_score",
    "vector_db_hybrid",
    "eval_framework",
    "ltr_reranking",
    "product_builder_score",
)

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
    "retrieval_search": [
        "Profile mentions retrieval/search, though production depth is less explicit.",
        "Shows retrieval/search relevance, with limited scale evidence in the text.",
        "Has search-aligned evidence, but fewer concrete system details than top profiles."
    ],
    "sys_experience_score": [
        "Lists recommendation or ranking systems as a relevant skill.",
        "Shows recommender/ranking alignment, with limited project detail.",
        "Has adjacent ranking-system evidence in the profile."
    ],
    "vector_db_hybrid": [
        "Mentions vector search or vector databases, with limited production context.",
        "Shows vector-search relevance, but fewer deployment details than stronger profiles.",
        "Has vector/hybrid search evidence at a lighter level."
    ],
    "ltr_reranking": [
        "Mentions learning-to-rank or reranking, with limited production context.",
        "Shows ranking-model alignment, but less concrete deployment evidence.",
        "Has reranking evidence, though system ownership is not deeply described."
    ],
    "eval_framework": [
        "References evaluation metrics without a full ranking-quality framework.",
        "Shows measurement awareness, with limited experiment-system detail.",
        "Has evaluation evidence, though the system framing is lighter."
    ],
    "product_builder_score": [
        "Has product-company experience, with limited ML scaling detail.",
        "Shows product exposure, though the ML ownership signal is lighter.",
        "Product background is relevant, but technical scaling evidence is thinner."
    ],
    "python_coding": [
        "Mentions Python, with limited core engineering context.",
        "Shows Python relevance, but implementation depth is not strongly described.",
        "Has coding evidence at a lighter level than top engineering profiles."
    ],
    "llm_integration": [
        "Lists LLM/RAG capabilities, with limited production-scale evidence.",
        "Shows LLM/RAG relevance, though deployment context is lighter.",
        "Has LLM integration evidence adjacent to the retrieval role."
    ]
}

LEAD_TEMPLATES_MODERATE = {
    "retrieval_search": [
        "Relevant retrieval/search evidence appears in profile text.",
        "Career text shows hands-on search or retrieval exposure.",
        "Shows direct alignment with retrieval-system work in the JD."
    ],
    "sys_experience_score": [
        "Profile evidence points to recommender or ranking-system work.",
        "Shows practical exposure to recommendation and ranking systems.",
        "Career text aligns with marketplace ranking and personalization work."
    ],
    "vector_db_hybrid": [
        "Shows vector-search infrastructure exposure in profile text.",
        "Has evidence around vector or hybrid-search systems.",
        "Profile text connects to the JD's hybrid retrieval mandate."
    ],
    "ltr_reranking": [
        "Shows ranking or reranking evidence relevant to the JD.",
        "Profile text aligns with learning-to-rank or reranking work.",
        "Has ranking-system evidence suited to the hybrid ranker track."
    ],
    "eval_framework": [
        "Shows evaluation or experimentation evidence for ranking systems.",
        "Profile text includes evaluation signals useful for search quality work.",
        "Has measurement-oriented evidence aligned with ranking evaluation."
    ],
    "product_builder_score": [
        "Product-company experience supports the JD's shipper mindset.",
        "Career path shows product ownership in applied ML settings.",
        "Profile indicates product ML experience beyond isolated research."
    ],
    "python_coding": [
        "Python engineering evidence supports hands-on implementation work.",
        "Shows coding-oriented evidence for building the ranking stack.",
        "Profile text supports the JD's Python-heavy execution needs."
    ],
    "llm_integration": [
        "LLM/RAG exposure can support retrieval-product integration work.",
        "Profile shows LLM integration evidence adjacent to the JD's ranker goals.",
        "Has LLM/RAG evidence that complements retrieval-system work."
    ]
}

SUPPORT_TEMPLATES = [
    "Also demonstrates capabilities in {secondary_name}",
    "Secondary support comes from {secondary_name}",
    "The profile also has {secondary_name} coverage",
    "Additional JD alignment appears in {secondary_name}"
]


def _variant(cand: dict, count: int) -> int:
    if count <= 0:
        return 0
    seed = f"{cand.get('candidate_id', '')}:{cand.get('rank', '')}"
    return sum(ord(ch) for ch in seed) % count


def _trim_snippet(snippet: str, limit: int = 80) -> str:
    clean = " ".join(str(snippet or "").strip().split())
    clean = clean.lstrip(" ,.;:-")
    first_word = clean.split(" ", 1)[0] if clean else ""
    allowed_lower_starts = {"a", "an", "and", "at", "built", "for", "in", "of", "on", "the", "to", "with"}
    if first_word.isalpha() and first_word.islower() and first_word not in allowed_lower_starts:
        clean = clean.split(" ", 1)[1] if " " in clean else clean
    if len(clean) > limit:
        clean = clean[: limit - 3].rsplit(" ", 1)[0].rstrip(" ,.;:-") + "..."
    return clean


def _behavioral_summary(cand: dict) -> str:
    parts = []
    notice_days = cand.get("beh_notice_period_days")
    try:
        if notice_days is not None:
            notice_int = int(float(notice_days))
            if notice_int <= 30:
                parts.append(f"{notice_int}-day notice")
            elif notice_int > 90:
                parts.append(f"{notice_int}-day notice")
    except (TypeError, ValueError):
        pass

    response_rate = cand.get("beh_recruiter_response_rate")
    try:
        if response_rate is not None:
            rate = float(response_rate)
            if rate >= 0.75 or rate < 0.20:
                parts.append(f"{int(round(rate * 100))}% recruiter response")
    except (TypeError, ValueError):
        pass

    return ", ".join(parts[:2])


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
    notice_days = cand.get("beh_notice_period_days")
    if notice_days is not None:
        try:
            if float(notice_days) > 90:
                return f"Long notice period ({int(float(notice_days))} days) raises hiring friction."
        except (TypeError, ValueError):
            pass

    location = str(cand.get("beh_location") or "").strip()
    country = str(cand.get("beh_country") or "").strip().lower()
    if country and country != "india":
        return "Located outside India, which the JD treats as case-by-case."
    preferred_fragments = (
        "pune", "noida", "gurgaon", "gurugram", "delhi", "new delhi",
        "faridabad", "ghaziabad", "hyderabad", "mumbai",
    )
    if location and not any(fragment in location.lower() for fragment in preferred_fragments):
        return "Location is outside the JD's preferred or welcome city list."
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


def generate_reasoning(cand: dict) -> str:
    """
    Evidence-driven lead selection, strictly modulated by rank/score.
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
    
    # Tone depends on rank and the actual domain score
    rank = cand.get("rank", 100)
    if rank <= 15:
        primary_key, primary_name = next(
            (
                (key, name)
                for key, name in ranked_domains
                if key in TOP_RANK_PRIMARY_DOMAINS and cand.get(key, 0.0) >= 2.0
            ),
            ranked_domains[0],
        )
    else:
        primary_key, primary_name = ranked_domains[0]
    secondary_key, secondary_name = next(
        ((key, name) for key, name in ranked_domains if key != primary_key),
        (None, None),
    )
    primary_score = cand.get(primary_key, 0.0)
    
    # 2. Lead sentence
    # A score >= 3.0 means they had career description evidence WITH production/scale context.
    # Score 2.0 means they just mentioned it in text. Score 1.0 means just in skills list.
    if rank <= 30 and primary_score >= 3.0:
        snippet = snippets.get(primary_key, "")
        if snippet and "{snippet}" in LEAD_TEMPLATES_STRONG.get(primary_key, ""):
            clean_snippet = _trim_snippet(snippet)
            lead = LEAD_TEMPLATES_STRONG[primary_key].format(snippet=clean_snippet)
        else:
            lead = f"Candidate shows strong evidence in {primary_name}."
    elif rank <= 30 and primary_score >= 2.0:
        snippet = snippets.get(primary_key, "")
        if rank <= 15 and snippet:
            clean_snippet = _trim_snippet(snippet)
            lead = f"Profile evidence in {primary_name}, specifically: '{clean_snippet}'."
        else:
            options = LEAD_TEMPLATES_MODERATE.get(primary_key, [f"Profile text shows relevant {primary_name} evidence."])
            lead = options[_variant(cand, len(options))]
    elif primary_score >= 2.0:
        options = LEAD_TEMPLATES_WEAK.get(primary_key, [f"Mentions {primary_name} in profile text, though production depth is less explicit."])
        lead = options[_variant(cand, len(options))]
    elif primary_score > 0.0:
        lead = f"Lists {primary_name} in skills without career context."
    else:
        lead = "Limited direct evidence found for core technical requirements."

    profile_prefix = _profile_prefix(cand)
    if profile_prefix:
        lead = f"{profile_prefix}: {lead}"

    # 3. Support sentence
    support = ""
    # Only offer a supportive secondary capability if they are highly ranked and actually have the skill
    if rank <= 50 and secondary_key and cand.get(secondary_key, 0.0) >= 1.0:
        template = SUPPORT_TEMPLATES[_variant(cand, len(SUPPORT_TEMPLATES))]
        support = template.format(secondary_name=secondary_name)
    elif rank <= 30:
        milestone = get_90day_milestone(primary_key)
        support = f"best positioned for {milestone}"

    # 4. Concern sentence
    caveat = ""
    concern_text = get_largest_concern(cand)
    
    # JD mandates: If rank > 30, concerns must be acknowledged if present
    # If rank > 70, gap acknowledgment is mandatory
    if concern_text and (rank > 30 or "notice period" in concern_text or "Location" in concern_text or "outside India" in concern_text):
        caveat = f"note: {concern_text}"
    elif rank > 70 and not concern_text:
        # We must acknowledge a gap for low-ranked candidates
        # If score is very low, make it clear
        if cand.get("core_score", 0.0) < 40.0:
            caveat = "note: failed to meet the technical depth required for the JD"
        else:
            caveat = "note: overall evidence density is lower than top-tier candidates"

    behavioral = _behavioral_summary(cand)
    if behavioral:
        behavioral = f"hiring signal: {behavioral}"

    second_parts = [p.rstrip(".") for p in [support, behavioral, caveat] if p]
    if second_parts:
        return f"{lead} {'; '.join(second_parts)}."
    return lead
