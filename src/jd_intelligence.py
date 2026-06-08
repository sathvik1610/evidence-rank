"""
src/jd_intelligence.py - Phase 0 JD contract expansion.

Builds retrieval and reranking query text from metadata/JD_contract.yaml and
Resources/job_description.txt. The goal is to keep the first-mile retrieval
inputs synchronized with the recruiter-intent contract instead of relying on
handwritten HyDE blocks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List
import re

import yaml

import constants


_MAX_TERMS_PER_RULE = 18

_BM25_NOISE_TERMS = {
    "conversion",
    "engagement",
    "retention",
    "business metrics",
    "lift",
    "kpi",
    "growth metric",
    "startup",
    "seed stage",
    "shipped",
    "launched",
    "real users",
    "user feedback",
    "feedback loop",
    "human evaluation",
    "online experiments",
    "experimentation platform",
    "hit rate",
    "hit-rate",
}

_RULE_LABELS = {
    "core_search_and_retrieval": "production retrieval, search, vector, and hybrid-search systems",
    "semantic_recsys_and_ranking": "ranking, recommendation, matching, and marketplace relevance systems",
    "evaluation_culture": "ranking evaluation, offline/online metrics, A/B interpretation, and feedback loops",
    "hands_on_engineering_python": "strong hands-on Python engineering and ML implementation",
    "llm_fine_tuning": "LLM fine-tuning and applied LLM integration",
    "distributed_systems_and_inference": "distributed systems, model serving, and inference optimization",
    "domain_exposure_hr_marketplace": "HR-tech, recruiting, job marketplace, or talent-platform exposure",
    "shipper_product_engineer": "scrappy product-engineering behavior and shipping to real users",
    "operational_pain_and_scale": "production operational pain such as drift, index refresh, regressions, and latency",
    "ownership_action_verbs": "end-to-end ownership of relevant production systems",
}


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_text(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def _norm_term(term: str) -> str:
    return re.sub(r"\s+", " ", str(term).strip()).lower()


def _patterns(rule: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for pat in rule.get("patterns", []) or []:
        if isinstance(pat, str):
            term = _norm_term(pat)
            if term:
                out.append(term)
    return _dedupe(out)


def _term_to_regex(term: str) -> str:
    """Convert a contract phrase to a forgiving regex while preserving intent."""
    escaped = re.escape(_norm_term(term))
    escaped = re.sub(r"(\\\s|\\-|_)+", r"[\\s._/-]+", escaped)
    # Keep common metric notation flexible: ndcg@ should match ndcg@10.
    if escaped.endswith("@"):
        return escaped
    return escaped


def _dedupe(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = _norm_term(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _rules_by_id(contract: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        rule.get("id"): rule
        for rule in contract.get("extraction_rules", []) or []
        if isinstance(rule, dict) and rule.get("id")
    }


def load_jd_contract(contract_path: str | Path = constants.JD_CONTRACT_YAML) -> Dict[str, Any]:
    """Public loader used by pipeline phases that must stay synchronized with YAML."""
    return _load_yaml(contract_path)


def get_rule_patterns(
    contract: Dict[str, Any],
    rule_id: str,
    as_regex: bool = False,
) -> List[str]:
    """Return de-duplicated patterns for one extraction rule."""
    rule = _rules_by_id(contract).get(rule_id, {})
    terms = _patterns(rule)
    return [_term_to_regex(t) for t in terms] if as_regex else terms


def get_condition_values(
    contract: Dict[str, Any],
    multiplier_id: str,
    field: str,
    operator: str | None = None,
) -> List[str]:
    """Return list values from multiplier conditions, e.g. title or skill penalty terms."""
    for mult in contract.get("multipliers", []) or []:
        if not isinstance(mult, dict) or mult.get("id") != multiplier_id:
            continue
        values: List[str] = []
        for cond in mult.get("conditions", []) or []:
            if cond.get("field") == field and (operator is None or cond.get("operator") == operator):
                value = cond.get("value")
                if isinstance(value, list):
                    values.extend(str(v).lower() for v in value)
                elif isinstance(value, str):
                    values.append(value.lower())
        return _dedupe(values)
    return []


def get_multiplier_value(contract: Dict[str, Any], multiplier_id: str, default: float = 1.0) -> float:
    for mult in contract.get("multipliers", []) or []:
        if isinstance(mult, dict) and mult.get("id") == multiplier_id:
            try:
                return float(mult.get("multiplier_value", default))
            except (TypeError, ValueError):
                return default
    return default


def get_band_multiplier(contract: Dict[str, Any], multiplier_id: str, value: float, default: float = 1.0) -> float:
    for mult in contract.get("multipliers", []) or []:
        if not isinstance(mult, dict) or mult.get("id") != multiplier_id:
            continue
        for band in mult.get("bands", []) or []:
            try:
                if float(band["min"]) <= value <= float(band["max"]):
                    return float(band["multiplier"])
            except (KeyError, TypeError, ValueError):
                continue
    return default


def get_location_bands(contract: Dict[str, Any]) -> Dict[str, List[str]]:
    """Parse location city lists from the YAML location multiplier condition strings."""
    bands = {"preferred": [], "welcome": []}
    for mult in contract.get("multipliers", []) or []:
        if not isinstance(mult, dict) or mult.get("id") != "location_tier_multiplier":
            continue
        for idx, band in enumerate(mult.get("bands", []) or []):
            condition = str(band.get("condition", ""))
            found = re.findall(r"'([^']+)'", condition)
            if idx == 0:
                bands["preferred"].extend(x.lower() for x in found)
            elif idx == 1:
                bands["welcome"].extend(x.lower() for x in found)
    return {k: _dedupe(v) for k, v in bands.items()}


def build_feature_contract(contract_path: str | Path = constants.JD_CONTRACT_YAML) -> Dict[str, Any]:
    """
    Build Phase 3/5 regex and policy config from the JD YAML.

    This keeps feature extraction and behavioral reranking aligned with the
    same recruiter-intent contract used by Phase 0 retrieval.
    """
    contract = load_jd_contract(contract_path)
    rule = lambda rid: get_rule_patterns(contract, rid, as_regex=True)
    raw_rule = lambda rid: get_rule_patterns(contract, rid, as_regex=False)

    retrieval = rule("core_search_and_retrieval")
    ranking = rule("semantic_recsys_and_ranking")
    evaluation = rule("evaluation_culture")
    operations = rule("operational_pain_and_scale")
    shipper = rule("shipper_product_engineer")
    ownership = rule("ownership_action_verbs")
    llm = rule("llm_fine_tuning")
    python = rule("hands_on_engineering_python")
    distributed = rule("distributed_systems_and_inference") + rule("devops_infrastructure")
    hr = rule("domain_exposure_hr_marketplace")

    vector_terms = [
        p for p in retrieval
        if any(anchor in p for anchor in (
            "vector", "hybrid", "dense", "sparse", "embedding", "faiss",
            "pinecone", "qdrant", "milvus", "weaviate", "opensearch",
            "elasticsearch", "hnsw", "pgvector"
        ))
    ]

    external_terms = [
        "published", "publication", "paper", "conference", "talk", "speaker",
        "blog", "maintainer", "contributor", "open source", "open-source",
        "github", "arxiv", "neurips", "icml", "iclr", "sigir", "acl",
        "emnlp", "naacl", "keynote",
    ]

    return {
        "target_skills": {
            "retrieval_search": retrieval,
            "vector_db_hybrid": vector_terms,
            "eval_framework": evaluation,
            "ltr_reranking": ranking,
            "llm_integration": llm + [r"retrieval[\s._/-]+augmented", r"\brag\b"],
            "python_coding": python,
            "distributed_systems": distributed,
            "hr_tech_exposure": hr,
        },
        "retrieval_patterns": retrieval,
        "ranking_patterns": ranking,
        "recommendation_patterns": ranking,
        "system_semantics_patterns": retrieval + ranking,
        "production_patterns": operations + shipper + [
            r"\bproduction\b", r"\breal[\s._/-]+users\b", r"\blive[\s._/-]+traffic\b",
            r"\bshipped[\s._/-]+model\b", r"\bmodel[\s._/-]+in[\s._/-]+production\b",
        ],
        "shipper_terms": shipper + ownership,
        "researcher_terms": [r"\bpaper\b", r"\bbenchmark\b", r"\bablation\b", r"\bnovel\b", r"\barxiv\b"],
        "ownership_patterns": ownership,
        "external_validation_terms": [_term_to_regex(t) for t in external_terms],
        "stopped_coding_titles": get_condition_values(
            contract,
            "code_stopped_architect",
            "profile.current_title",
            "matches_any_ignorecase",
        ),
        "hands_on_title_terms": get_condition_values(
            contract,
            "code_stopped_architect",
            "profile.current_title",
            "matches_none_ignorecase",
        ),
        "framework_demo_terms": get_condition_values(
            contract,
            "langchain_tourist_trap",
            "skills[].name",
            "matches_any_ignorecase",
        ),
        "pre_llm_production_terms": retrieval + ranking,
        "wrong_domain_terms": get_condition_values(
            contract,
            "computer_vision_trap",
            "skills[].name",
            "matches_any_ignorecase",
        ),
        "research_title_terms": get_condition_values(
            contract,
            "pure_research_penalty",
            "profile.current_title",
            "matches_any_ignorecase",
        ),
        "wrong_domain_escape_terms": raw_rule("core_search_and_retrieval") + raw_rule("semantic_recsys_and_ranking"),
        "seniority_bands": next(
            (m.get("bands", []) for m in contract.get("multipliers", []) or [] if m.get("id") == "seniority_sweet_spot"),
            [],
        ),
        "location_bands": get_location_bands(contract),
        "floor_exempt_multiplier_ids": contract.get("metadata", {})
            .get("multiplier_application", {})
            .get("floor_exempt_multiplier_ids", []),
        "multiplier_values": {
            "consulting_heavy_soft_penalty": get_multiplier_value(contract, "consulting_heavy_soft_penalty", 0.40),
            "pure_research_penalty": get_multiplier_value(contract, "pure_research_penalty", 0.20),
            "computer_vision_trap": get_multiplier_value(contract, "computer_vision_trap", 0.60),
            "keyword_stuffer_penalty": get_multiplier_value(contract, "keyword_stuffer_penalty", 0.55),
        },
    }


def _terms_for(rule_map: Dict[str, Dict[str, Any]], rule_id: str, limit: int = _MAX_TERMS_PER_RULE) -> List[str]:
    return _patterns(rule_map.get(rule_id, {}))[:limit]


def _phrase(label: str, terms: List[str]) -> str:
    if not terms:
        return label
    return f"{label}: {', '.join(terms)}"


def _contract_keywords(contract: Dict[str, Any]) -> List[str]:
    keywords: List[str] = []
    for rule in contract.get("extraction_rules", []) or []:
        if not isinstance(rule, dict):
            continue
        bucket = rule.get("feature_bucket", "")
        if bucket not in {"primary_fit", "must_have", "nice_to_have"}:
            continue
        for term in _patterns(rule):
            if term in _BM25_NOISE_TERMS:
                continue
            # Single generic words are poor BM25 anchors unless they are named systems/metrics.
            if " " not in term and term not in {
                "bm25", "faiss", "pinecone", "weaviate", "milvus", "qdrant",
                "elasticsearch", "opensearch", "lucene", "solr", "hnsw",
                "bge", "e5", "gte", "pgvector", "ndcg", "mrr@", "map@",
                "python", "pytorch", "numpy", "fastapi", "flask", "pytest",
                "lora", "qlora", "peft", "rlhf", "dpo", "sft", "vllm",
                "triton", "tensorrt", "mlops",
            }:
                continue
            keywords.append(term)
    return _dedupe(keywords)


def _jd_anchor_text(jd_text: str) -> str:
    anchors = []
    for line in jd_text.splitlines():
        low = line.lower()
        if any(
            key in low
            for key in (
                "ideal candidate",
                "things you absolutely need",
                "production experience",
                "what you'd actually be doing",
                "explicitly do not want",
                "location:",
                "active on redrob",
            )
        ):
            anchors.append(line.strip("* ").strip())
    return " ".join(anchors[:10])


def build_jd_intelligence(
    contract_path: str | Path = constants.JD_CONTRACT_YAML,
    jd_text_path: str | Path = constants.JD_TEXT,
) -> Dict[str, Any]:
    """
    Return all Phase 0 text inputs derived from the YAML contract and JD text.

    Output keys:
      - config: auditable structured summary saved to artifacts/jd_config.json
      - keywords: BM25 lexical query terms saved to artifacts/jd_keywords.json
      - queries: the three dense/sparse query texts encoded by BGE-M3
      - cross_encoder_query: richer JD text used for offline cross-encoder pairs
    """
    contract = _load_yaml(contract_path)
    jd_text = _load_text(jd_text_path)
    rule_map = _rules_by_id(contract)

    core_terms = _terms_for(rule_map, "core_search_and_retrieval", 24)
    ranking_terms = _terms_for(rule_map, "semantic_recsys_and_ranking", 22)
    eval_terms = _terms_for(rule_map, "evaluation_culture", 20)
    python_terms = _terms_for(rule_map, "hands_on_engineering_python", 12)
    llm_terms = _terms_for(rule_map, "llm_fine_tuning", 10)
    ops_terms = _terms_for(rule_map, "operational_pain_and_scale", 12)
    shipper_terms = _terms_for(rule_map, "shipper_product_engineer", 10)
    domain_terms = _terms_for(rule_map, "domain_exposure_hr_marketplace", 9)
    dist_terms = _terms_for(rule_map, "distributed_systems_and_inference", 8)

    config = {
        "source_contract": str(contract_path),
        "source_jd_text": str(jd_text_path),
        "contract_version": contract.get("metadata", {}).get("version"),
        "role_intent": contract.get("metadata", {}).get("description", ""),
        "must_have": [
            _phrase(_RULE_LABELS["core_search_and_retrieval"], core_terms),
            _phrase(_RULE_LABELS["semantic_recsys_and_ranking"], ranking_terms),
            _phrase(_RULE_LABELS["evaluation_culture"], eval_terms),
            _phrase(_RULE_LABELS["hands_on_engineering_python"], python_terms),
        ],
        "nice_to_have": [
            _phrase(_RULE_LABELS["llm_fine_tuning"], llm_terms),
            _phrase(_RULE_LABELS["distributed_systems_and_inference"], dist_terms),
            _phrase(_RULE_LABELS["domain_exposure_hr_marketplace"], domain_terms),
        ],
        "product_builder": [
            _phrase(_RULE_LABELS["shipper_product_engineer"], shipper_terms),
            _phrase(_RULE_LABELS["operational_pain_and_scale"], ops_terms),
            _phrase(_RULE_LABELS["ownership_action_verbs"], _terms_for(rule_map, "ownership_action_verbs", 10)),
        ],
        "disqualifier_multiplier_ids": [
            m.get("id")
            for m in contract.get("multipliers", []) or []
            if isinstance(m, dict) and float(m.get("multiplier_value", 1.0)) < 0.75
        ],
    }

    v1_skills = (
        "Senior AI Engineer candidate for Redrob. Must show career-history evidence of "
        f"{config['must_have'][0]}. {config['must_have'][1]}. "
        f"{config['must_have'][2]}. {config['must_have'][3]}. "
        "Prefer production systems deployed to real users over keyword-only skill lists."
    )

    hyde_recsys = (
        "Hypothetical ideal candidate profile: 6 to 8 years in applied ML or AI at product companies. "
        "Built and shipped an end-to-end ranking, search, recommendation, or candidate-job matching system. "
        f"Relevant system language includes {', '.join(ranking_terms + core_terms[:10])}. "
        "Owned retrieval quality, product impact, and real-user deployment rather than demos."
    )

    hyde_eval = (
        "Hypothetical ideal candidate profile: hands-on Python engineer who improves retrieval and ranking quality. "
        f"Evaluation evidence includes {', '.join(eval_terms)}. "
        f"Production-operational evidence includes {', '.join(ops_terms)}. "
        f"Useful LLM integration includes {', '.join(llm_terms)}, but only with strong pre-LLM retrieval or ranking foundations."
    )

    jd_anchor = _jd_anchor_text(jd_text)
    cross_encoder_query = (
        "Redrob Senior AI Engineer founding-team role. Recruiter intent: find a product-minded, hands-on "
        "AI engineer who has shipped production retrieval, ranking, search, recommendation, or matching systems "
        "to real users, can evaluate them rigorously, writes strong Python, and is reachable for hiring. "
        f"{v1_skills} {hyde_recsys} {hyde_eval} JD anchors: {jd_anchor}"
    )

    keywords = _contract_keywords(contract)
    # Add a few phrase anchors from the generated query in case the YAML is too terse.
    keywords = _dedupe(keywords + [
        "production retrieval",
        "production ranking",
        "candidate job matching",
        "offline online evaluation",
        "recruiter feedback",
        "product company",
        "real users",
    ])

    return {
        "config": config,
        "keywords": keywords,
        "queries": {
            "v1_skills": v1_skills,
            "hyde_recsys": hyde_recsys,
            "hyde_eval": hyde_eval,
        },
        "cross_encoder_query": cross_encoder_query,
    }
