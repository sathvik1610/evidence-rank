"""
src/explainer.py — Phase 6: Reason Generation

Generates safe, hallucination-free reasoning strings for the final submission.
Follows the Phase 6 architecture: Evidence-driven lead selection based on the
candidate's strongest domain, supported by snippet extraction, and ranking-aware
gap acknowledgment.
"""

import json
import re

STRENGTH_DOMAINS = [
    ("retrieval_search", "Retrieval Systems"),
    ("eval_framework", "Evaluation Metrics"),
    ("vector_db_hybrid", "Vector Search Infrastructure"),
    ("sys_experience_score", "Recommendation and Ranking Systems"),
    ("ltr_reranking", "Learning-to-Rank and Reranking"),
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

REVIEW_BAND_SECONDARY_DOMAINS = (
    "retrieval_search",
    "eval_framework",
    "vector_db_hybrid",
    "sys_experience_score",
    "ltr_reranking",
)

LEAD_TEMPLATES_STRONG = {
    "retrieval_search": "direct retrieval/search evidence, including {snippet}",
    "sys_experience_score": "recommendation or ranking-system evidence, including {snippet}",
    "vector_db_hybrid": "vector or hybrid-search infrastructure evidence, including {snippet}",
    "ltr_reranking": "direct learning-to-rank/reranking evidence",
    "eval_framework": "ranking-evaluation evidence, including {snippet}",
    "product_builder_score": "product ML and scaling evidence, including {snippet}",
    "python_coding": "Python engineering evidence, including {snippet}",
    "llm_integration": "LLM/RAG integration evidence, including {snippet}"
}

LEAD_TEMPLATES_WEAK = {
    "retrieval_search": [
        "Profile mentions retrieval/search, though production depth is less explicit.",
        "Profile has retrieval/search evidence, with limited scale evidence in the extracted text.",
        "Has search-aligned evidence, but fewer concrete system details than top profiles."
    ],
    "sys_experience_score": [
        "Lists recommendation or ranking systems as a relevant skill.",
        "Profile has recommender/ranking alignment, with limited extracted project detail.",
        "Has adjacent ranking-system evidence in the profile."
    ],
    "vector_db_hybrid": [
        "Mentions vector search or vector databases, with limited production context.",
        "Profile has vector-search evidence, but fewer extracted deployment details than stronger profiles.",
        "Has vector/hybrid search evidence at a lighter level."
    ],
    "ltr_reranking": [
        "Mentions learning-to-rank or reranking, with limited production context.",
        "Profile has ranking-model evidence, but less concrete deployment evidence.",
        "Has reranking evidence, though system ownership is not deeply described."
    ],
    "eval_framework": [
        "References evaluation metrics without a full ranking-quality framework.",
        "Profile has measurement evidence, with limited experiment-system detail.",
        "Has evaluation evidence, though the system framing is lighter."
    ],
    "product_builder_score": [
        "Has product-company experience, with limited ML scaling detail.",
        "Profile has product exposure, though the ML ownership signal is lighter.",
        "Product background is relevant, but technical scaling evidence is thinner."
    ],
    "python_coding": [
        "Has Python evidence, but broader ranking/retrieval evidence is limited.",
        "Profile has Python evidence, while implementation depth is not strongly described.",
        "Has coding evidence at a lighter level than stronger engineering profiles."
    ],
    "llm_integration": [
        "Lists LLM/RAG capabilities, with limited production-scale evidence.",
        "Profile has LLM/RAG evidence, though deployment context is lighter.",
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
    "Also has {secondary_name} evidence: {snippet}",
    "Also shows {secondary_name} evidence: {snippet}",
    "The profile also has {secondary_name} evidence: {snippet}",
    "Adds {secondary_name} support: {snippet}"
]

DOMAIN_SHORT_NAMES = {
    "retrieval_search": "retrieval/search",
    "eval_framework": "ranking evaluation",
    "vector_db_hybrid": "vector or hybrid search",
    "sys_experience_score": "recommendation/ranking systems",
    "ltr_reranking": "learning-to-rank/reranking",
    "product_builder_score": "product ML",
    "python_coding": "Python engineering",
    "llm_integration": "LLM/RAG",
}


def _variant(cand: dict, count: int) -> int:
    if count <= 0:
        return 0
    seed = f"{cand.get('candidate_id', '')}:{cand.get('rank', '')}"
    return sum(ord(ch) for ch in seed) % count


def _trim_snippet(snippet: str, limit: int = 80) -> str:
    clean = " ".join(str(snippet or "").strip().split())
    clean = clean.lstrip(" ,.;:-")
    exact_rewrites = {
        "OpenAI embeddings, storing in Pinecone) and the answer-generation layer":
            "RAG pipeline using OpenAI embeddings and Pinecone-backed retrieval",
        "Shipped the personalization infrastructure: the system that learns":
            "shipped personalization and ranking infrastructure",
        "Designed the relevance labeling pipeline mix of click-through data":
            "Designed a relevance labeling pipeline using click-through data",
        "9 months, Designed the relevance labeling pipeline mix of click-through data":
            "Designed a relevance labeling pipeline using click-through data",
        "Recently, I shipped our first RAG-based feature this year and now own the eval":
            "shipped a RAG-based feature and owned evaluation work",
        "Implemented a RAG-based customer support chatbot integrated with our exist":
            "Implemented a RAG-based customer support chatbot",
        "BM25 setup, validated through human relevance judgments, Owned the ranking la":
            "BM25 retrieval validated through human relevance judgments and ranking-layer ownership",
        "BM25 setup, validated through human relevance judgments, Owned the ranking lay":
            "BM25 retrieval validated through human relevance judgments and ranking-layer ownership",
        "BM25 setup, validated through human relevance judgments, Owned t":
            "BM25 retrieval validated through human relevance judgments",
        "reranking edge cases": "reranking work",
    }
    for bad, good in exact_rewrites.items():
        if bad.lower() in clean.lower():
            return good
    clean = re.sub(r"^scratch\s+[—-]\s+", "built an evaluation harness using ", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\b(?:NLP|AI|ML|Search|Machine Learning|Applied ML)\s+Engineer\s+(?=Owned\b)", "", clean)
    clean = re.sub(r"\b(?:NLP|AI|ML|Search|Machine Learning|Applied ML)\s+Engineer\s+(?=Trained\b)", "", clean)
    clean = re.sub(r"\b(?:Learning\s+)?Engineer\s+(?=Shipped|Implemented\b)", "", clean)
    clean = clean.replace("the works, I care a lot about", "").strip(" ,.;:-")
    clean = re.sub(
        r"^(?:Search\s*&\s*Ranking|Ranking|Retrieval|Evaluation|Vector Search)\s+",
        "",
        clean,
        flags=re.IGNORECASE,
    ).lstrip(" ,.;:-")
    clean = re.sub(
        r"^(?:(?:Senior|Staff|Lead)\s+)?(?:Machine Learning Engineer|ML Engineer|AI Engineer|NLP Engineer|Search Engineer|Applied ML Engineer|Applied Scientist|Data Scientist|Recommendation Systems Engineer)\s+(?=(?:Built|Owned|Designed|Shipped|Trained|Implemented|Developed|Led|Migrated)\b)",
        "",
        clean,
        flags=re.IGNORECASE,
    ).lstrip(" ,.;:-")
    first_piece = clean.split(" ", 1)[0] if clean else ""
    if len(first_piece.rstrip(" ,.;:-")) <= 2 and any(ch in first_piece for ch in ",;:-") and " " in clean:
        clean = clean.split(" ", 1)[1].lstrip(" ,.;:-")
    first_piece = clean.split(" ", 1)[0] if clean else ""
    if "-" in first_piece:
        prefix = first_piece.split("-", 1)[0]
        if prefix.islower() and len(prefix) <= 2 and " " in clean:
            clean = clean.split(" ", 1)[1].lstrip(" ,.;:-")
    first_piece = clean.split(" ", 1)[0] if clean else ""
    if (
        first_piece.endswith(".")
        and first_piece[:-1].islower()
        and len(first_piece[:-1]) <= 8
        and " " in clean
    ):
        clean = clean.split(" ", 1)[1].lstrip(" ,.;:-")
    weak_starts = {"a", "an", "and", "or", "the"}
    while clean and " " in clean:
        first_word = clean.split(" ", 1)[0].lower().strip(" ,.;:-")
        if first_word not in weak_starts:
            break
        clean = clean.split(" ", 1)[1].lstrip(" ,.;:-")
    first_word = clean.split(" ", 1)[0] if clean else ""
    allowed_lower_starts = {
        "a", "an", "at", "built", "designed", "developed", "for", "implemented",
        "in", "index", "migrated", "of", "offline", "on", "online", "optimizing",
        "owned", "ranking", "retrieval", "semantic", "shipped", "trained", "upgraded",
        "used", "vector", "with",
    }
    dropped_lower_prefixes = 0
    while (
        first_word.isalpha()
        and first_word.islower()
        and first_word not in allowed_lower_starts
        and " " in clean
        and dropped_lower_prefixes < 3
    ):
        clean = clean.split(" ", 1)[1] if " " in clean else clean
        clean = clean.lstrip(" ,.;:-—")
        first_word = clean.split(" ", 1)[0] if clean else ""
        dropped_lower_prefixes += 1
    while clean and " " in clean:
        first_word = clean.split(" ", 1)[0].lower().strip(" ,.;:-")
        if first_word not in weak_starts:
            break
        clean = clean.split(" ", 1)[1].lstrip(" ,.;:-")
    if len(clean) > limit:
        clean = clean[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:-")
    elif clean.endswith("."):
        clean = clean[:-1]
    clean = clean.replace(". ", ", ")
    clean = clean.replace(". ", ", ")
    clean = re.sub(
        r"^(?:early|history),\s+(?=(?:Built|Owned|Designed|Shipped|Trained|Implemented|Developed|Led|Migrated)\b)",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"^(?:engineering\s+)?early,\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^\d+\s+months,\s+Designed\s+", "Designed ", clean, flags=re.IGNORECASE)
    clean = re.sub(
        r"^Designed the relevance labeling pipeline\s*\(?mix of click-through data.*$",
        "Designed a relevance labeling pipeline using click-through data",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(
        r"^BM25 setup, validated through human relevance judgments,\s*.*$",
        "BM25 retrieval validated through human relevance judgments",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"^parts:\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r",\s*Strong\s+\w*$", "", clean)
    clean = re.sub(r",?\s*the works,.*$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r",\s*Designed the relevance$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^\(?via\s+", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+and a gradient-boosted$", "", clean, flags=re.IGNORECASE)
    clean = re.sub(
        r",\s*(?:NLP|Rec|Senior Data Scientist|AI Engineer Developed|AI Engineer Trained|NLP Engineer Trained|Search Engineer|Engineer Implemented).*$",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = clean.rstrip(" ,.;:-")
    if clean.count(")") > clean.count("("):
        clean = clean.replace(")", "")
    if clean.count("(") > clean.count(")"):
        clean = clean.replace("(", "")
    dangling = {
        "a", "an", "and", "as", "at", "be", "building", "designed", "for",
        "determined", "exist", "from", "going", "in", "of", "or", "our", "strong", "that", "the", "to",
        "using", "which", "with",
    }
    while clean and clean.rsplit(" ", 1)[-1].lower().rstrip(" ,.;:-") in dangling and " " in clean:
        clean = clean.rsplit(" ", 1)[0].rstrip(" ,.;:-")
    last_word = clean.rsplit(" ", 1)[-1] if clean else ""
    if "-" in last_word:
        suffix = last_word.rsplit("-", 1)[-1].rstrip(" ,.;:-")
        incomplete_suffixes = {"bas", "tun", "retr", "rer", "vec", "eval"}
        if suffix in incomplete_suffixes and " " in clean:
            clean = clean.rsplit(" ", 1)[0].rstrip(" ,.;:-")
            while clean and clean.rsplit(" ", 1)[-1].lower().rstrip(" ,.;:-") in dangling and " " in clean:
                clean = clean.rsplit(" ", 1)[0].rstrip(" ,.;:-")
    return clean


def _lower_first(text: str) -> str:
    clean = str(text or "").strip()
    if not clean:
        return clean
    return clean[:1].lower() + clean[1:]


def _natural_caveat(concern_text: str) -> str:
    clean = str(concern_text or "").strip().rstrip(".")
    if not clean:
        return ""
    lower = clean.lower()
    if "outside the jd's preferred or welcome city list and relocation is not indicated" in lower:
        return (
            "The main caveat is logistics: the candidate is outside the preferred/welcome cities, "
            "with no relocation signal"
        )
    if "outside the jd's preferred or welcome city list" in lower:
        return "The main caveat is logistics: the candidate is outside the preferred/welcome cities"
    if "located outside india" in lower:
        return "The main caveat is logistics: the candidate is outside India, which the JD treats as case-by-case"
    if "notice" in lower and ("hiring friction" in lower or "hiring bar" in lower):
        return f"The main caveat is availability: {_lower_first(clean)}"
    if "skill-duration metadata" in lower:
        return (
            f"{clean}, so the explanation avoids unverified duration claims"
            if "duration claims" not in lower
            else clean
        )
    if lower == "ranking-evaluation evidence is lighter than stronger candidates":
        return "Compared with stronger candidates, the profile has lighter evidence of rigorous ranking evaluation"
    if lower == "vector or hybrid-search evidence is less complete than the top band":
        return "Compared with the top band, vector/hybrid-search evidence is less complete"
    if lower == "learning-to-rank or reranking evidence is less complete than stronger profiles":
        return "Compared with stronger candidates, learning-to-rank/reranking ownership is less complete"
    if lower == "sustained career search/ranking density is weaker than higher-ranked candidates":
        return "Compared with stronger candidates, sustained career search/ranking density is weaker"
    if lower == "product-shipping signal is lighter than the strongest jd matches":
        return "Compared with the strongest JD matches, product-shipping evidence is lighter"
    if lower == "evidence is strong but less complete than the top-ranked production owners":
        return "The profile is solid, but less complete than the strongest production retrieval/ranking owners"
    if lower == "evidence is relevant but narrower than the jd's ideal end-to-end retrieval/ranking/evaluation profile":
        return "The profile is relevant, but narrower than the JD's ideal end-to-end retrieval/ranking/evaluation profile"
    if lower == "production ranking ownership and evaluation evidence are limited compared with stronger candidates":
        return "Production ranking ownership and rigorous evaluation evidence are limited compared with stronger candidates"
    if lower == "failed to meet the technical depth required for the jd":
        return "The profile remains below stronger matches because it does not meet the JD's full technical-depth bar"
    if lower == "overall evidence density is lower than top-tier candidates":
        return "Overall evidence density is lower than the strongest candidates"
    return f"The main caveat is {_lower_first(clean)}"


def _rank_band_label(rank: int) -> str:
    if rank <= 10:
        return "Excellent JD fit"
    if rank <= 25:
        return "Strong JD fit"
    if rank <= 50:
        return "Solid JD-aligned profile"
    if rank <= 75:
        return "Relevant but partial JD fit"
    return "Borderline JD-adjacent fit"


def _band_limiter(cand: dict, primary_key: str) -> str:
    rank = int(cand.get("rank", 100) or 100)
    concern = get_largest_concern(cand)
    if concern:
        return concern
    if rank > 70 and cand.get("core_score", 0.0) < 40.0:
        return "failed to meet the technical depth required for the JD."
    if rank <= 25:
        return ""
    if cand.get("eval_framework", 0.0) < 2.0:
        return "ranking-evaluation evidence is lighter than stronger candidates."
    if cand.get("vector_db_hybrid", 0.0) < 2.0:
        return "vector or hybrid-search evidence is less complete than the top band."
    if cand.get("ltr_reranking", 0.0) < 2.0:
        return "learning-to-rank or reranking evidence is less complete than stronger profiles."
    if cand.get("career_ir_density", 1.0) < 0.60:
        return "sustained career search/ranking density is weaker than higher-ranked candidates."
    if cand.get("product_builder_score", 1.0) < 0.65:
        return "product-shipping signal is lighter than the strongest JD matches."
    if rank <= 50:
        return "evidence is strong but less complete than the top-ranked production owners."
    if rank <= 75:
        return "evidence is relevant but narrower than the JD's ideal end-to-end retrieval/ranking/evaluation profile."
    return "production ranking ownership and evaluation evidence are limited compared with stronger candidates."


def _finish_sentence(text: str) -> str:
    clean = str(text or "").rstrip()
    if not clean:
        return clean
    if clean.endswith((".", "!", "?", "...")):
        return clean
    return f"{clean}."


def _snippet_for(snippets: dict, key: str) -> str:
    return _trim_snippet(snippets.get(key, ""))


def _explanation_quality(snippet: str) -> int:
    text = str(snippet or "").lower()
    if not text:
        return -10
    score = 0
    concrete_terms = (
        "bm25", "bge", "faiss", "pinecone", "opensearch", "elasticsearch",
        "ndcg", "mrr", "recall@k", "a/b", "learning-to-rank", "xgboost",
        "hybrid", "dense retrieval", "human relevance", "production",
        "qps", "queries", "users", "latency",
    )
    action_terms = (
        "built", "owned", "designed", "shipped", "migrated", "trained",
        "implemented", "deployed", "serving", "validated",
    )
    score += sum(2 for term in concrete_terms if term in text)
    score += sum(1 for term in action_terms if term in text)
    generic_terms = (
        "strong background", "comfortable across", "machine learning engineer",
        "senior ml engineer", "staff machine learning engineer", "applied ai",
        "academic background",
    )
    score -= sum(4 for term in generic_terms if term in text)
    if text.startswith(("and ", "that ", "to ")):
        score -= 3
    return score


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
            if rate >= 0.60 or rate < 0.50:
                parts.append(f"{int(round(rate * 100))}% recruiter response")
    except (TypeError, ValueError):
        pass

    return ", ".join(parts[:2])


def _behavioral_phrase(cand: dict) -> str:
    summary = _behavioral_summary(cand)
    if not summary:
        return ""
    lower = summary.lower()
    response_match = re.search(r"(\d+)% recruiter response", lower)
    low_response = bool(response_match and int(response_match.group(1)) < 50)
    if "120-day" in lower or "90-day" in lower or low_response:
        return f"Hiring caveat: {summary}"
    return f"Hiring fit is helped by {summary}"


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
    if cand.get("isolated_template_risk", False):
        return "Core IR evidence appears isolated relative to the broader career pattern."
    if cand.get("career_ir_density", 1.0) < 0.40 and cand.get("adjacent_career_ratio", 0.0) >= 0.40:
        return "Broader career pattern leans toward adjacent ML/chatbot/MLOps rather than sustained search or ranking ownership."
    location = str(cand.get("beh_location") or "").strip()
    country = str(cand.get("beh_country") or "").strip().lower()
    preferred_fragments = (
        "pune", "noida", "gurgaon", "gurugram", "delhi", "new delhi",
        "faridabad", "ghaziabad", "hyderabad", "mumbai",
    )
    willing = cand.get("beh_willing_to_relocate", False)
    outside_preferred = location and not any(fragment in location.lower() for fragment in preferred_fragments)
    notice_days = cand.get("beh_notice_period_days")
    try:
        notice_float = float(notice_days) if notice_days is not None else None
    except (TypeError, ValueError):
        notice_float = None
    if (
        notice_float is not None
        and notice_float > 90
        and outside_preferred
        and not willing
    ):
        return f"{int(notice_float)}-day notice plus no indicated relocation outside the JD's preferred or welcome city list raises hiring friction."
    if notice_float is not None and notice_float > 90:
        return f"Long notice period ({int(notice_float)} days) raises hiring friction."
    if notice_float is not None and notice_float >= 90:
        return f"{int(notice_float)}-day notice raises the hiring bar for this role."
    if country and country != "india":
        return "Located outside India, which the JD treats as case-by-case."
    if outside_preferred and not willing:
        return "Location is outside the JD's preferred or welcome city list and relocation is not indicated."
    if outside_preferred:
        return "Location is outside the JD's preferred or welcome city list."
    target_contradictions = cand.get("target_skill_duration_contradiction", 0) or 0
    skill_contradictions = cand.get("contradiction_skill_duration", 0) or 0
    try:
        target_contradictions = int(target_contradictions)
    except (TypeError, ValueError):
        target_contradictions = 0
    try:
        skill_contradictions = int(skill_contradictions)
    except (TypeError, ValueError):
        skill_contradictions = 0
    if target_contradictions > 0 or skill_contradictions > 0:
        total = target_contradictions + skill_contradictions
        return f"Skill-duration metadata has {total} overclaim signal(s), so duration claims are not used in this explanation."
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
        key=lambda d: (cand.get(d[0], 0.0), _explanation_quality(_snippet_for(snippets, d[0]))),
        reverse=True
    )
    
    # Tone depends on rank and the actual domain score
    rank = cand.get("rank", 100)
    if rank <= 15:
        primary_key, primary_name = next(
            (
                (key, name)
                for key, name in ranked_domains
                if key in TOP_RANK_PRIMARY_DOMAINS
                and cand.get(key, 0.0) >= 2.0
                and _explanation_quality(_snippet_for(snippets, key)) >= 0
            ),
            ranked_domains[0],
        )
    else:
        primary_key, primary_name = ranked_domains[0]
    secondary_key, secondary_name = (None, None)
    if rank <= 60:
        primary_snippet_candidate = _snippet_for(snippets, primary_key)
        secondary_key, secondary_name = next(
            (
                (key, name)
                for key, name in STRENGTH_DOMAINS
                if key != primary_key
                and key in REVIEW_BAND_SECONDARY_DOMAINS
                and cand.get(key, 0.0) >= 2.0
                and _snippet_for(snippets, key)
                and _snippet_for(snippets, key) != primary_snippet_candidate
            ),
            (None, None),
        )
    if secondary_key is None:
        secondary_key, secondary_name = next(
            ((key, name) for key, name in ranked_domains if key != primary_key),
            (None, None),
        )
    primary_score = cand.get(primary_key, 0.0)
    primary_snippet = _snippet_for(snippets, primary_key)
    
    # 2. Lead sentence
    # A score >= 3.0 means they had career description evidence WITH production/scale context.
    # Score 2.0 means they just mentioned it in text. Score 1.0 means just in skills list.
    band_label = _rank_band_label(rank)
    profile_prefix = _profile_prefix(cand)
    identity = profile_prefix if profile_prefix else "Candidate"
    primary_short = DOMAIN_SHORT_NAMES.get(primary_key, primary_name)
    if rank <= 60 and primary_score >= 3.0 and primary_snippet:
        strong_template = LEAD_TEMPLATES_STRONG.get(primary_key, "")
        if primary_snippet and "{snippet}" in strong_template:
            lead = _finish_sentence(
                f"{band_label}: {identity} and "
                + strong_template.format(snippet=primary_snippet)
            )
        elif strong_template:
            lead = _finish_sentence(f"{band_label}: {identity} and {strong_template}")
        else:
            lead = f"{band_label}: {identity} has extracted {primary_short} evidence relevant to the JD."
    elif rank <= 60 and primary_score >= 2.0 and primary_snippet:
        if primary_snippet:
            lead = _finish_sentence(f"{band_label}: {identity} and clear {primary_short} evidence: {primary_snippet}")
        else:
            options = LEAD_TEMPLATES_MODERATE.get(primary_key, [f"Profile text shows relevant {primary_name} evidence."])
            lead = _finish_sentence(f"{band_label}: {identity}. {options[_variant(cand, len(options))]}")
    elif primary_score >= 2.0:
        options = LEAD_TEMPLATES_WEAK.get(primary_key, [f"Mentions {primary_name} in profile text, though production depth is less explicit."])
        lead = _finish_sentence(f"{band_label}: {identity}. {options[_variant(cand, len(options))]}")
    elif primary_score > 0.0:
        lead = f"{band_label}: {identity} has some extracted {primary_short} signal, but career-description evidence is limited."
    else:
        lead = f"{band_label}: {identity} has limited direct evidence for the core technical requirements."

    # 3. Support sentence
    support = ""
    # Only offer a supportive secondary capability if they are highly ranked and actually have the skill
    if rank <= 15 and cand.get("career_ir_density", 0.0) >= 0.60:
        secondary_snippet = _snippet_for(snippets, secondary_key) if secondary_key else ""
        if secondary_key and cand.get(secondary_key, 0.0) >= 2.0 and secondary_snippet:
            template = SUPPORT_TEMPLATES[_variant(cand, len(SUPPORT_TEMPLATES))]
            secondary_label = DOMAIN_SHORT_NAMES.get(secondary_key, secondary_name)
            support = template.format(secondary_name=secondary_label, snippet=secondary_snippet)
        else:
            support = "Career-history features show sustained search/ranking/evaluation density"
    elif rank <= 60 and secondary_key and cand.get(secondary_key, 0.0) >= 2.0:
        secondary_snippet = _snippet_for(snippets, secondary_key)
        if secondary_snippet:
            template = SUPPORT_TEMPLATES[_variant(cand, len(SUPPORT_TEMPLATES))]
            secondary_label = DOMAIN_SHORT_NAMES.get(secondary_key, secondary_name)
            support = template.format(secondary_name=secondary_label, snippet=secondary_snippet)
    elif rank <= 60 and secondary_key and cand.get(secondary_key, 0.0) >= 1.0:
        template = SUPPORT_TEMPLATES[_variant(cand, len(SUPPORT_TEMPLATES))]
        secondary_snippet = _snippet_for(snippets, secondary_key)
        if secondary_snippet:
            secondary_label = DOMAIN_SHORT_NAMES.get(secondary_key, secondary_name)
            support = template.format(secondary_name=secondary_label, snippet=secondary_snippet)
    elif rank <= 30:
        milestone = get_90day_milestone(primary_key)
        support = f"best positioned for {milestone}"

    # 4. Concern sentence
    caveat = ""
    concern_text = _band_limiter(cand, primary_key)
    
    # JD mandates: If rank > 30, concerns must be acknowledged if present
    # If rank > 70, gap acknowledgment is mandatory
    if concern_text and (
        rank > 25
        or "notice period" in concern_text
        or "Location" in concern_text
        or "outside India" in concern_text
        or "isolated" in concern_text
        or "career pattern" in concern_text
    ):
        caveat = _natural_caveat(concern_text)
    elif rank > 70 and not concern_text:
        # We must acknowledge a gap for low-ranked candidates
        # If score is very low, make it clear
        if cand.get("core_score", 0.0) < 40.0:
            caveat = _natural_caveat("failed to meet the technical depth required for the JD")
        else:
            caveat = _natural_caveat("overall evidence density is lower than top-tier candidates")

    behavioral = _behavioral_phrase(cand)

    second_parts = [p.rstrip(".") for p in [support, behavioral, caveat] if p]
    if second_parts:
        return f"{lead} {'; '.join(second_parts)}."
    return lead
