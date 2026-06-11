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
    r"weaviate|opensearch|elasticsearch|hnsw|pgvector|sentence-transformers?)\b",
    re.IGNORECASE,
)
CONCRETE_VECTOR_TOOL_RE = re.compile(
    r"\b(faiss|pinecone|qdrant|milvus|weaviate|opensearch|elasticsearch|"
    r"pgvector|hnsw|sentence-transformers?|bge|e5|openai embeddings|"
    r"cohere embeddings|jina embeddings|voyage)\b",
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
    return cand
