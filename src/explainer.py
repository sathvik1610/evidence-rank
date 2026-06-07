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
    ("eval_framework", "Evaluation Metrics"),
    ("product_builder_score", "Product ML & Scaling"),
    ("python_coding", "Python Engineering")
]

LEAD_TEMPLATES = {
    "retrieval_search": "Strong evidence in Retrieval Systems, specifically: '{snippet}'.",
    "sys_experience_score": "Significant experience building Recommendation and Ranking Systems.",
    "vector_db_hybrid": "Deep expertise in Vector Search Infrastructure, notably: '{snippet}'.",
    "eval_framework": "Rigorous focus on Evaluation Metrics and testing methodologies: '{snippet}'.",
    "product_builder_score": "Proven track record in Product ML & Scaling across engineering teams.",
    "python_coding": "Solid foundation in Python Engineering: '{snippet}'."
}

def get_90day_milestone(domain_key: str) -> str:
    if domain_key in ("retrieval_search", "sys_experience_score"):
        return "the Weeks 1-3 audit (BM25/Retrieval)"
    elif domain_key in ("vector_db_hybrid", "product_builder_score"):
        return "the Weeks 4-8 mandate (Ship v2 Hybrid Ranker)"
    elif domain_key == "eval_framework":
        return "the Weeks 9-12 mandate (Build Evaluation Framework)"
    return "early engineering milestones"

def get_largest_concern(cand: dict) -> str:
    # Ranked by severity/impact
    if cand.get("research_only", False):
        return "Research-heavy background with limited production ML exposure."
    if cand.get("wrong_domain", False):
        return "Primary experience is outside NLP/Search domains."
    if cand.get("langchain_only_flag", False):
        return "Heavy reliance on LLM wrappers (LangChain) without deep ML infrastructure evidence."
    if cand.get("consulting_flag", False):
        return "Career arc leans heavily toward consulting rather than core product ownership."
    if cand.get("code_stopped", False):
        return "Seniority indicates candidate may have shifted away from hands-on coding."
    if cand.get("title_velocity_flag", False):
        return "Frequent title changes noted across recent roles."
    return ""

def generate_reasoning(cand: dict) -> str:
    """
    Evidence-driven lead selection.
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

    # 2. Lead sentence
    primary_score = cand.get(primary_key, 0.0)
    if primary_score > 0.0:
        snippet = snippets.get(primary_key, "")
        if snippet and "{snippet}" in LEAD_TEMPLATES[primary_key]:
            # Clean snippet for formatting
            clean_snippet = snippet.strip()
            if len(clean_snippet) > 80:
                clean_snippet = clean_snippet[:77] + "..."
            lead = LEAD_TEMPLATES[primary_key].format(snippet=clean_snippet)
        else:
            # Fallback if no snippet available or template doesn't use it
            lead = f"Candidate shows strong evidence in {primary_name}."
    else:
        lead = "Limited direct evidence found for core technical requirements."

    # 3. Support sentence
    support = ""
    if secondary_key and cand.get(secondary_key, 0.0) > 0.5:
        support = f"Also demonstrates capabilities in {secondary_name}."
    else:
        milestone = get_90day_milestone(primary_key)
        support = f"Best positioned for {milestone}."

    # 4. Concern sentence
    caveat = ""
    rank = cand.get("rank", 100)
    
    # JD mandates: If rank > 30, concerns must be acknowledged if present
    # If rank > 70, gap acknowledgment is mandatory
    concern_text = get_largest_concern(cand)
    if concern_text and rank > 30:
        caveat = f"Note: {concern_text}"
    elif rank > 70 and not concern_text:
        # We must acknowledge a gap
        caveat = "Note: Overall evidence density is lower than top-tier candidates."

    parts = [p for p in [lead, support, caveat] if p]
    return " ".join(parts)
