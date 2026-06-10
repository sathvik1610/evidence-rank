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
SHIP_RE = re.compile(
    r"\b(owned|built|shipped|deployed|production|serving|rolled out|migration|"
    r"real users|recruiter-feedback|feedback loop|qps|queries|latency|p95|"
    r"index refresh|drift|rollback|monitoring|dashboard)\b",
    re.IGNORECASE,
)
SERVICES_RE = re.compile(r"\b(services|consulting|outsourcing|ai services|it services)\b", re.IGNORECASE)


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
    recent_text = " ".join([_profile_context(profile_record)] + [_role_text(r) for r in roles[:2]])
    current_text = _role_text(roles[0]) if roles else _profile_context(profile_record)
    career_text = " ".join([_profile_context(profile_record)] + [_role_text(r) for r in roles])

    recent_retrieval = bool(RETRIEVAL_RE.search(recent_text))
    recent_vector = bool(VECTOR_RE.search(recent_text))
    recent_ranking = bool(RANKING_RE.search(recent_text))
    recent_eval = bool(EVAL_RE.search(recent_text))
    recent_ship = bool(SHIP_RE.search(recent_text))

    career_retrieval = bool(RETRIEVAL_RE.search(career_text))
    career_vector = bool(VECTOR_RE.search(career_text))
    career_ranking = bool(RANKING_RE.search(career_text))
    career_eval = bool(EVAL_RE.search(career_text))
    career_ship = bool(SHIP_RE.search(career_text))

    recent_hits = sum((recent_retrieval, recent_vector, recent_ranking, recent_eval, recent_ship))
    career_hits = sum((career_retrieval, career_vector, career_ranking, career_eval, career_ship))

    full_plan = 0.0
    if recent_hits >= 5:
        full_plan = 1.0
    elif recent_retrieval and recent_ranking and recent_eval and recent_ship:
        full_plan = 0.85
    elif career_hits >= 5:
        full_plan = 0.70
    elif career_retrieval and career_ranking and career_eval:
        full_plan = 0.50

    current_industry = str(profile.get("current_industry", "") or "")
    current_company = str(profile.get("current_company", "") or "")
    current_services = bool(SERVICES_RE.search(current_industry) or SERVICES_RE.search(current_company))

    cand["runtime_full_plan_signal"] = full_plan
    cand["runtime_recent_retrieval_signal"] = float(recent_retrieval)
    cand["runtime_recent_vector_signal"] = float(recent_vector)
    cand["runtime_recent_ranking_signal"] = float(recent_ranking)
    cand["runtime_recent_eval_signal"] = float(recent_eval)
    cand["runtime_recent_ship_signal"] = float(recent_ship)
    cand["runtime_current_services_signal"] = float(current_services)
    cand["runtime_current_role_text"] = current_text[:280]
    return cand
