"""
Runtime JD calibration helpers.

These functions read only the final ranking candidates and add lightweight
signals from current/recent profile text. They are intentionally generic:
no candidate IDs, no final-rank overrides, and no hidden labels.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.weights import W


RETRIEVAL_RE = re.compile(
    r"\b(search|retrieval|bm25|semantic search|dense retrieval|hybrid retrieval|"
    r"hybrid search|candidate sourcing|candidate-jd matching|matching layer|"
    r"search and discovery|query understanding)\b",
    re.IGNORECASE,
)
VECTOR_RE = re.compile(
    r"\b(vector|embedding|embeddings|bge|e5|faiss|pinecone|qdrant|milvus|"
    r"weaviate|opensearch|elasticsearch|hnsw|pgvector|sentence-transformers?|"
    # Specific ANN library names — zero false-positive risk
    r"nmslib|scann|"
    # Neural retrieval architectures — highly domain-specific, multi-word for safety
    r"dual[\s\-]encoder|bi[\-]encoder|two[\s\-._]tower|"
    # Full phrase only — safe as multi-word, still gated by career_ship co-occurrence
    r"approximate nearest neighbor)\b",
    re.IGNORECASE,
)
CONCRETE_VECTOR_TOOL_RE = re.compile(
    r"\b(faiss|pinecone|qdrant|milvus|weaviate|opensearch|elasticsearch|"
    r"pgvector|hnsw|sentence-transformers?|bge|e5|openai embeddings|"
    r"cohere embeddings|jina embeddings|voyage|"
    # Additional specific ANN libraries added in Fix 2
    r"nmslib|scann)\b",
    re.IGNORECASE,
)
RANKING_RE = re.compile(
    r"\b(ranking layer|ranking pipeline|ranker|re-ranker|reranker|reranking|"
    r"learning-to-rank|learning to rank|ltr|xgboost|lambdamart|recommendation|"
    r"recommender|personalization|discovery feed|matching layer|scoring function)\b",
    re.IGNORECASE,
)
EVAL_RE = re.compile(
    r"\b(ndcg|mrr|map@|recall@k|precision@k|offline[- ]online|a/b|ab test|"
    r"evaluation framework|eval framework|evaluation harness|relevance labeling|"
    r"human relevance|human judgments|offline metrics|online metrics|"
    r"metric correlation|engagement metrics|simulated a/b)\b",
    re.IGNORECASE,
)
EVAL_ADJACENT_RE = re.compile(
    r"\b(engagement signals?|implicit-feedback|implicit feedback|click signals?|"
    r"conversion signals?|feedback signals?|gradient-boosted re-ranking|"
    r"gradient boosted re-ranking|quality regression|retrieval-quality regression)\b",
    re.IGNORECASE,
)
SHIP_RE = re.compile(
    r"\b(owned|built|shipped|deployed|production|serving|rolled out|migration|"
    r"real users|recruiter-feedback|feedback loop|qps|queries|latency|p95|"
    r"index refresh|drift|rollback|monitoring|dashboard)\b",
    re.IGNORECASE,
)
PYTHON_RE = re.compile(
    r"\b(python|pydata|pandas|numpy|scikit-learn|sklearn|pytorch|tensorflow|"
    r"fastapi|django|flask)\b",
    re.IGNORECASE,
)
PYTHON_NATIVE_ML_TOOL_RE = re.compile(
    r"\b(mlflow|kubeflow|hugging\s*face|transformers|bentoml|pytest|"
    r"jupyter|notebook)\b",
    re.IGNORECASE,
)
FAISS_RE = re.compile(r"\bfaiss\b", re.IGNORECASE)
HANDS_ON_ML_ROLE_RE = re.compile(
    r"\b(machine learning engineer|ml engineer|applied ml engineer|ai engineer|"
    r"search engineer|recommendation systems engineer|recommender engineer|"
    r"ranking engineer|nlp engineer|applied scientist|data scientist)\b",
    re.IGNORECASE,
)
SERVICES_RE = re.compile(
    r"\b(services|consulting|outsourcing|ai services|it services|professional services|"
    r"system integration|client delivery|staffing)\b",
    re.IGNORECASE,
)

# Part 4: Same-project full-system bonus patterns
# Group 1: Retrieval/Search/Matching
FSYS_RETRIEVAL_RE = re.compile(
    r"\b(search|retrieval|ranking pipeline|matching|recommendation system|"
    r"candidate search|recruiter search|candidate.jd matching|discovery feed|"
    r"marketplace ranking|semantic search|hybrid retrieval)\b",
    re.IGNORECASE,
)
# Group 2: Embeddings/Vector/Hybrid
FSYS_VECTOR_RE = re.compile(
    r"\b(embedding|sentence.transformer|bge|e5|openai embedding|dense retrieval|"
    r"sparse retrieval|hybrid retrieval|bm25.*dense|vector db|faiss|pinecone|"
    r"weaviate|qdrant|milvus|opensearch|elasticsearch|hnsw)\b",
    re.IGNORECASE,
)
# Group 3: Ranking/Reranking/LTR
FSYS_RANKING_RE = re.compile(
    r"\b(learning.to.rank|ltr|reranker|re.ranker|re.scoring|xgboost ranker|"
    r"lightgbm ranker|ranker variants|ranking model|behavioral re.ranking|top.k reranking)\b",
    re.IGNORECASE,
)
# Group 4: Evaluation
FSYS_EVAL_RE = re.compile(
    r"\b(ndcg|mrr|map@|recall@k|offline.online correlation|a/b test|a/b testing|"
    r"human relevance judgments?|relevance labels?|recruiter feedback loop|"
    r"eval harness|evaluation framework)\b",
    re.IGNORECASE,
)
# Group 5: Production/Ops/Impact
FSYS_PROD_RE = re.compile(
    r"\b(production|shipped|deployed|real users|serving|qps|p95 latency|corpus|"
    r"30m|35m|50m|engagement|time.to.shortlist|revenue.per.search|index refresh|"
    r"embedding drift|rollback|monitoring|migration)\b",
    re.IGNORECASE,
)

# Part 5: Recruiter/Candidate workflow bonus
RECRUITER_WORKFLOW_RE = re.compile(
    r"\b(recruiter.facing search|recruiter search|recruiter engagement|"
    r"recruiter feedback loop|time.to.shortlist|candidate corpus|"
    r"candidate profiles?|candidate.jd matching|jd matching|role matching|"
    r"talent marketplace|hiring marketplace|candidate search product|"
    r"matching candidates? to jobs?|matching jobs? to candidates?)\b",
    re.IGNORECASE,
)

# Part 6: Adjacent/internal-only evidence detection
# Strong markers that evidence is primarily internal/adjacent (not production product retrieval)
INTERNAL_ONLY_RE = re.compile(
    r"\b(internal knowledge base|internal kb|customer.support chatbot|"
    r"support bot|rag chatbot|churn prediction|generic mlflow|"
    r"internal document search|employee search|internal corpus)\b",
    re.IGNORECASE,
)
# Strong markers of production product retrieval (negates internal-only classification)
# Fix 4: Expanded to include e-commerce search, LTR, offline-online correlation, relevance labeling etc.
PRODUCT_RETRIEVAL_RE = re.compile(
    r"\b(recruiter.facing|candidate search|talent marketplace|"
    r"marketplace ranking|product search|recommendation system at scale|"
    r"real users|shipped.*search|shipped.*retrieval|production retrieval|"
    r"e.commerce search|product discovery|ranking layer|learning.to.rank|"
    r"relevance labeling|offline.online correlation|offline.online metric|"
    r"recommendation.*serving|discovery feed|serving.*users|"
    r"\d+m (candidates?|items?|queries?|users?))\b",
    re.IGNORECASE,
)

# Fix 2: Strict ranking/LTR group check — mandatory for full-system bonus
# (separate from FSYS_RANKING_RE which includes broader patterns)
FSYS_STRICT_RANKING_RE = re.compile(
    r"\b(learning.to.rank|ltr|reranker|re.ranker|re.scoring|xgboost ranker|"
    r"lightgbm ranker|ranker variants|ranking model|ranking layer|behavioral re.ranking|"
    r"top.k reranking|relevance labeling|ranker|ranked|re-ranked)\b",
    re.IGNORECASE,
)

# Fix 2: Strict recruiter/candidate workflow for full-system alternative path
FSYS_RECRUITER_RE = re.compile(
    r"\b(recruiter.facing|recruiter search|candidate search|candidate.jd matching|"
    r"jd matching|talent marketplace|hiring marketplace|candidate corpus|"
    r"time.to.shortlist|recruiter feedback)\b",
    re.IGNORECASE,
)


def load_candidate_records(path: str, wanted_ids: set[str]) -> dict[str, dict[str, Any]]:
    """Load complete JSON records for a small set of candidate IDs."""
    records: dict[str, dict[str, Any]] = {}
    if not wanted_ids:
        return records
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            for item in json.load(f):
                cid = str(item.get("candidate_id", ""))
                if cid in wanted_ids:
                    records[cid] = item
        else:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                cid = str(item.get("candidate_id", ""))
                if cid in wanted_ids:
                    records[cid] = item
                    if len(records) == len(wanted_ids):
                        break
    return records


def _role_text(role: dict[str, Any]) -> str:
    return " ".join(
        str(role.get(key, "") or "")
        for key in ("title", "company", "industry", "description")
    )


def _profile_context(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    return " ".join(
        str(profile.get(key, "") or "")
        for key in ("current_title", "headline", "summary", "current_industry")
    )


def _skills_text(candidate: dict[str, Any]) -> str:
    return " ".join(
        str(skill.get("name", "") or "")
        for skill in candidate.get("skills", [])
        if isinstance(skill, dict)
    )


def _count(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text or ""))


def calibrate_candidate(cand: dict[str, Any], profile_record: dict[str, Any] | None) -> dict[str, Any]:
    """
    Add recruiter-calibration signals to a candidate feature dict.

    The important distinction is current/recent evidence. The JD asks what this
    person will do in the first 90 days, so recent shipped ranking/eval work
    should break ties over older or skills-only evidence.
    """
    if not profile_record:
        cand["runtime_full_plan_signal"] = 0.0
        cand["runtime_current_services_signal"] = 0.0
        return cand

    profile = profile_record.get("profile", {})
    roles = profile_record.get("career_history", [])
    profile_context = _profile_context(profile_record)
    recent_text = " ".join([profile_context] + [_role_text(r) for r in roles[:2]])
    current_text = _role_text(roles[0]) if roles else _profile_context(profile_record)
    previous_text = _role_text(roles[1]) if len(roles) > 1 else ""
    career_text = " ".join([_profile_context(profile_record)] + [_role_text(r) for r in roles])
    skills_text = _skills_text(profile_record)
    career_plus_skills_text = " ".join([career_text, skills_text])

    recent_retrieval = bool(RETRIEVAL_RE.search(recent_text))
    recent_vector = bool(VECTOR_RE.search(recent_text))
    recent_ranking = bool(RANKING_RE.search(recent_text))
    recent_eval = bool(EVAL_RE.search(recent_text))
    recent_ship = bool(SHIP_RE.search(recent_text))

    current_retrieval = bool(RETRIEVAL_RE.search(current_text))
    current_vector = bool(VECTOR_RE.search(current_text))
    current_ranking = bool(RANKING_RE.search(current_text))
    current_eval = bool(EVAL_RE.search(current_text))
    current_ship = bool(SHIP_RE.search(current_text))
    previous_retrieval = bool(RETRIEVAL_RE.search(previous_text))
    previous_ranking = bool(RANKING_RE.search(previous_text))
    previous_eval = bool(EVAL_RE.search(previous_text))
    previous_ship = bool(SHIP_RE.search(previous_text))

    career_retrieval = bool(RETRIEVAL_RE.search(career_text))
    career_vector = bool(VECTOR_RE.search(career_text))
    career_ranking = bool(RANKING_RE.search(career_text))
    career_eval = bool(EVAL_RE.search(career_text))
    career_eval_adjacent = bool(EVAL_ADJACENT_RE.search(career_text))
    career_ship = bool(SHIP_RE.search(career_text))
    explicit_python = bool(PYTHON_RE.search(career_plus_skills_text))
    python_native_tool = bool(PYTHON_NATIVE_ML_TOOL_RE.search(career_plus_skills_text))
    faiss_with_hands_on_ml = (
        bool(FAISS_RE.search(career_plus_skills_text))
        and bool(HANDS_ON_ML_ROLE_RE.search(career_text))
        and career_ship
    )
    career_python = explicit_python or python_native_tool or faiss_with_hands_on_ml
    skill_vector = bool(VECTOR_RE.search(skills_text))
    concrete_vector_tool = bool(CONCRETE_VECTOR_TOOL_RE.search(skills_text))
    career_production_retrieval = career_ship and (career_retrieval or career_ranking)
    career_production_vector = career_ship and career_vector
    corroborated_vector = career_production_vector or (career_production_retrieval and concrete_vector_tool)
    career_ranking_eval_adjacent = (
        career_eval_adjacent
        and career_ship
        and (career_retrieval or career_ranking)
    )

    recent_hits = sum((recent_retrieval, recent_vector, recent_ranking, recent_eval, recent_ship))
    career_hits = sum((career_retrieval, career_vector, career_ranking, career_eval, career_ship))

    current_hits = sum((current_retrieval, current_vector, current_ranking, current_eval, current_ship))
    previous_full_plan_support = bool(
        previous_ship and (previous_retrieval or previous_ranking) and previous_eval
    )

    full_plan = 0.0
    if current_hits >= 5:
        full_plan = 1.0
    elif current_retrieval and current_ranking and current_eval and current_ship:
        full_plan = 0.85
    elif recent_hits >= 5 and (current_retrieval or current_ranking or current_eval):
        full_plan = W["behavioral.runtime_full_plan_current_role_cap"]
    elif recent_retrieval and recent_ranking and recent_eval and recent_ship and (
        current_retrieval or current_ranking or current_eval
    ):
        full_plan = W["behavioral.runtime_full_plan_current_role_cap"]
    elif career_hits >= 5:
        full_plan = 0.70
    elif career_retrieval and career_ranking and career_eval:
        full_plan = 0.50
    if previous_full_plan_support and not (current_retrieval or current_ranking or current_eval):
        full_plan = min(full_plan, W["behavioral.runtime_full_plan_current_role_cap"])

    current_industry = str(profile.get("current_industry", "") or "")
    current_company = str(profile.get("current_company", "") or "")
    current_services = bool(SERVICES_RE.search(current_industry) or SERVICES_RE.search(current_company))

    cand["runtime_full_plan_signal"] = full_plan
    cand["runtime_recent_retrieval_signal"] = float(recent_retrieval)
    cand["runtime_recent_vector_signal"] = float(recent_vector)
    cand["runtime_recent_ranking_signal"] = float(recent_ranking)
    cand["runtime_recent_eval_signal"] = float(recent_eval)
    cand["runtime_recent_ship_signal"] = float(recent_ship)
    cand["runtime_current_retrieval_signal"] = float(current_retrieval)
    cand["runtime_current_vector_signal"] = float(current_vector)
    cand["runtime_current_ranking_signal"] = float(current_ranking)
    cand["runtime_current_eval_signal"] = float(current_eval)
    cand["runtime_current_ship_signal"] = float(current_ship)
    cand["runtime_career_retrieval_signal"] = float(career_retrieval)
    cand["runtime_career_vector_signal"] = float(career_vector)
    cand["runtime_career_ranking_signal"] = float(career_ranking)
    cand["runtime_career_eval_signal"] = float(career_eval)
    cand["runtime_production_retrieval_signal"] = float(career_production_retrieval)
    cand["runtime_production_vector_signal"] = float(career_production_vector)
    cand["runtime_vector_skill_signal"] = float(skill_vector)
    cand["runtime_concrete_vector_tool_signal"] = float(concrete_vector_tool)
    cand["runtime_corroborated_vector_signal"] = float(corroborated_vector)
    cand["runtime_career_eval_adjacent_signal"] = float(career_ranking_eval_adjacent)
    cand["runtime_career_python_signal"] = float(career_python)
    cand["runtime_career_ship_signal"] = float(career_ship)
    cand["runtime_current_services_signal"] = float(current_services)
    cand["runtime_current_role_text"] = current_text[:280]

    # -----------------------------------------------------------------------
    # Part 4 (Fix 2): Same-project full-system bonus — STRICT version
    # Full bonus requires: production + eval + vector + (ranking/LTR OR recruiter/candidate workflow)
    # Partial bonus: 4+ groups matched but missing strict ranking/recruiter requirement
    # -----------------------------------------------------------------------
    full_system_bonus_applied = False
    full_system_partial_applied = False
    full_system_groups_found = []
    full_system_snippet = ""
    best_group_count = 0
    for role in roles:
        role_text = _role_text(role)
        has_retrieval = bool(FSYS_RETRIEVAL_RE.search(role_text))
        has_vector    = bool(FSYS_VECTOR_RE.search(role_text))
        has_ranking   = bool(FSYS_RANKING_RE.search(role_text))
        has_eval      = bool(FSYS_EVAL_RE.search(role_text))
        has_prod      = bool(FSYS_PROD_RE.search(role_text))
        # Strict checks for full bonus path
        has_strict_ranking   = bool(FSYS_STRICT_RANKING_RE.search(role_text))
        has_strict_recruiter = bool(FSYS_RECRUITER_RE.search(role_text))

        matched = [
            ("retrieval", has_retrieval),
            ("vector",    has_vector),
            ("ranking",   has_ranking),
            ("evaluation",has_eval),
            ("production",has_prod),
        ]
        matched_names = [name for name, found in matched if found]
        if len(matched_names) > best_group_count:
            best_group_count = len(matched_names)
            full_system_groups_found = matched_names
            full_system_snippet = role_text[:200]

        if len(matched_names) >= 4:
            # Check strict full-bonus criteria:
            # Must have production + eval + vector + (strict ranking OR recruiter workflow)
            strict_full = (
                has_prod and has_eval and has_vector
                and (has_strict_ranking or has_strict_recruiter)
            )
            if strict_full:
                full_system_bonus_applied = True
                break
            else:
                # 4+ groups but missing strict ranking/recruiter — partial bonus
                full_system_partial_applied = True

    # Partial bonus only if no full bonus
    if full_system_bonus_applied:
        full_system_partial_applied = False

    cand["runtime_same_project_full_system_bonus_applied"] = full_system_bonus_applied
    cand["runtime_same_project_partial_system_bonus_applied"] = full_system_partial_applied
    cand["runtime_same_project_bonus_type"] = (
        "full" if full_system_bonus_applied else
        "partial" if full_system_partial_applied else
        "none"
    )
    cand["runtime_same_project_full_system_evidence_groups"] = ";".join(full_system_groups_found)
    cand["runtime_same_project_full_system_evidence_snippet"] = full_system_snippet

    # -----------------------------------------------------------------------
    # Part 5: Recruiter/Candidate workflow bonus from career-history only
    # -----------------------------------------------------------------------
    recruiter_workflow_hit = False
    recruiter_workflow_snippet = ""
    for role in roles:
        role_text = _role_text(role)
        m = RECRUITER_WORKFLOW_RE.search(role_text)
        if m:
            recruiter_workflow_hit = True
            start = max(0, m.start() - 60)
            recruiter_workflow_snippet = role_text[start:m.end() + 100].strip()[:200]
            break
    cand["runtime_recruiter_workflow_bonus_applied"] = recruiter_workflow_hit
    cand["runtime_recruiter_workflow_evidence_snippet"] = recruiter_workflow_snippet

    # -----------------------------------------------------------------------
    # Part 6 (Fix 4): Adjacent/internal-only evidence detection
    # Expanded PRODUCT_RETRIEVAL_RE now catches e-commerce search, LTR, offline-online etc.
    # -----------------------------------------------------------------------
    has_internal_only_evidence = bool(INTERNAL_ONLY_RE.search(career_text))
    has_product_retrieval_evidence = bool(PRODUCT_RETRIEVAL_RE.search(career_text))
    adjacent_internal_only = (
        has_internal_only_evidence
        and not has_product_retrieval_evidence
        and not full_system_bonus_applied
        and not full_system_partial_applied  # Fix 4: also exempt partial bonus candidates
    )
    cand["runtime_adjacent_internal_only_flag"] = adjacent_internal_only

    # Part 3: Bonus evidence levels for retrieval/vector/LTR/eval/product
    # Distinguish career-history evidence from skills-only evidence
    # Level 2 = full (production career-history), 1 = medium (internal/adjacent), 0 = low/none
    cand["runtime_retrieval_evidence_level"] = (
        2 if (career_ship and (career_retrieval or career_ranking)) else
        1 if career_retrieval else
        0
    )
    cand["runtime_vector_evidence_level"] = (
        2 if career_production_vector else
        1 if (career_vector and not career_ship) else
        0 if not career_vector else 0
    )
    cand["runtime_ltr_evidence_level"] = (
        2 if (bool(FSYS_RANKING_RE.search(career_text)) and career_ship) else
        1 if bool(FSYS_RANKING_RE.search(career_text)) else
        0
    )
    cand["runtime_eval_evidence_level"] = (
        2 if (career_eval and career_ship) else
        1 if career_eval_adjacent else
        0
    )
    cand["runtime_product_evidence_level"] = (
        2 if (career_ship and bool(PRODUCT_RETRIEVAL_RE.search(career_text))) else
        1 if career_ship else
        0
    )
    # Fix 3: Split-career core coverage bonus
    # Career-level coverage of product ranking/LTR/eval in one role + vector/search/retrieval in another
    # without needing all signals in the same role.
    # Eligibility: career has ship + (ranking or eval) + (retrieval or vector)
    # and has evidence of product-level (not purely internal) work
    split_career_bonus_applied = False
    if (
        not full_system_bonus_applied  # full bonus is already better
        and career_ship
        and (career_ranking or career_eval)
        and (career_retrieval or career_vector)
        and has_product_retrieval_evidence
        and len(roles) >= 2  # must span multiple roles
    ):
        # Check that the signals come from at least 2 different roles
        role_ranking_eval = []
        role_vector_retrieval = []
        for i, role in enumerate(roles):
            rt = _role_text(role)
            if bool(FSYS_RANKING_RE.search(rt)) or bool(FSYS_EVAL_RE.search(rt)):
                role_ranking_eval.append(i)
            if bool(FSYS_RETRIEVAL_RE.search(rt)) or bool(FSYS_VECTOR_RE.search(rt)):
                role_vector_retrieval.append(i)
        # Split: at least one role has ranking/eval AND a different role has retrieval/vector
        if role_ranking_eval and role_vector_retrieval:
            overlap = set(role_ranking_eval) & set(role_vector_retrieval)
            if len(overlap) < len(role_ranking_eval) or len(overlap) < len(role_vector_retrieval):
                split_career_bonus_applied = True
    cand["runtime_split_career_core_coverage_bonus_applied"] = split_career_bonus_applied

    return cand
