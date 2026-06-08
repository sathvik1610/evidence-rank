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
    "retrieval_search": "Mentions Retrieval Systems but lacks deep production evidence.",
    "sys_experience_score": "Lists Recommendation or Ranking Systems as a skill.",
    "vector_db_hybrid": "Mentions Vector Search or databases but lacks production context.",
    "ltr_reranking": "Mentions Learning-to-Rank or reranking but lacks deep production context.",
    "eval_framework": "References Evaluation Metrics but without rigorous system framing.",
    "product_builder_score": "Has product company experience but lacks clear ML scaling evidence.",
    "python_coding": "Mentions Python as a skill but without core engineering context.",
    "llm_integration": "Lists LLM/RAG capabilities but lacks production-scale evidence."
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
    
    primary_key, primary_name = ranked_domains[0]
    secondary_key, secondary_name = ranked_domains[1] if len(ranked_domains) > 1 else (None, None)

    # Tone depends on rank and the actual domain score
    rank = cand.get("rank", 100)
    primary_score = cand.get(primary_key, 0.0)
    
    # 2. Lead sentence
    # A score >= 3.0 means they had career description evidence WITH production/scale context.
    # Score 2.0 means they just mentioned it in text. Score 1.0 means just in skills list.
    if rank <= 30 and primary_score >= 3.0:
        snippet = snippets.get(primary_key, "")
        if snippet and "{snippet}" in LEAD_TEMPLATES_STRONG.get(primary_key, ""):
            clean_snippet = snippet.strip()
            if len(clean_snippet) > 80:
                clean_snippet = clean_snippet[:77] + "..."
            lead = LEAD_TEMPLATES_STRONG[primary_key].format(snippet=clean_snippet)
        else:
            lead = f"Candidate shows strong evidence in {primary_name}."
    elif primary_score >= 2.0:
        lead = LEAD_TEMPLATES_WEAK.get(primary_key, f"Mentions {primary_name} in profile text but lacks deep production evidence.")
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
        support = f"Also demonstrates capabilities in {secondary_name}."
    elif rank <= 30:
        milestone = get_90day_milestone(primary_key)
        support = f"Best positioned for {milestone}."

    # 4. Concern sentence
    caveat = ""
    concern_text = get_largest_concern(cand)
    
    # JD mandates: If rank > 30, concerns must be acknowledged if present
    # If rank > 70, gap acknowledgment is mandatory
    if concern_text and rank > 30:
        caveat = f"Note: {concern_text}"
    elif rank > 70 and not concern_text:
        # We must acknowledge a gap for low-ranked candidates
        # If score is very low, make it clear
        if cand.get("core_score", 0.0) < 40.0:
            caveat = "Note: Failed to meet the technical depth required for the JD."
        else:
            caveat = "Note: Overall evidence density is lower than top-tier candidates."

    second_parts = [p for p in [support, caveat] if p]
    if second_parts:
        return f"{lead} {' '.join(second_parts)}"
    return lead
