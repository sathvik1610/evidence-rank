## 7. Phase 3 — Candidate Feature Extraction

### 7.1 What this phase does

Runs **entirely offline** in `preprocess.py`. No time constraint. Processes the 3,000–5,000 candidates retrieved by Phase 1d, computes three evidence buckets (A/B/C) plus the consistency score, and serializes all results — including verbatim career description snippets — to `artifacts/candidate_features.parquet`. At rank time, `rank.py` uses **Polars** to safely load this parquet and perform an inner join against the 3,000 Candidate IDs from Phase 2; no regex or feature computation happens at rank time.

### 7.2 Evidence pattern sets

```python
import re

RETRIEVAL_PATTERNS = [
    r"faiss", r"pinecone", r"qdrant", r"milvus", r"weaviate",
    r"opensearch", r"elasticsearch", r"dense retrieval",
    r"vector search", r"embedding", r"semantic search",
    r"ann\b", r"approximate nearest", r"sentence.transformer",
    r"bi.encoder", r"cross.encoder", r"dense.encoder",
    r"retrieval system", r"search system", r"information retrieval"
]

RANKING_PATTERNS = [
    r"learning.to.rank", r"xgboost.*rank", r"lambdamart",
    r"pairwise.*rank", r"listwise", r"ranking.*pipeline",
    r"relevance.*score", r"rerank", r"bm25", r"search ranking"
]

RECOMMENDATION_PATTERNS = [
    r"recommendation.system", r"recsys", r"collaborative.filtering",
    r"content.based.filtering", r"matching.engine", r"candidate.matching",
    r"personalization.engine", r"match.score", r"recommender"
]

EVALUATION_PATTERNS = [
    r"ndcg", r"mrr\b", r"mean.average.precision", r"a/b.test",
    r"online.*eval", r"offline.*eval", r"precision.at", r"recall.at",
    r"evaluation.framework", r"ranking.metric"
]

PRODUCTION_PATTERNS = [
    r"produc.*deploy", r"latency", r"inference.*serv",
    r"real.user", r"live.*system", r"million.*request",
    r"billion.*query", r"serving.*infrastructure",
    r"qps\b", r"p99", r"p95", r"shipped to production"
]

# Shipper vs Researcher vocabulary — the JD's most explicit culture signal
SHIPPER_TERMS = [
    r"\bshipped\b", r"\blaunched\b", r"\bdeployed\b", r"\bbuilt\b",
    r"\bproduction\b", r"\breal users\b", r"\bcustomers\b",
    r"\brevenue\b", r"\bgrowth\b", r"\blatency\b", r"\bscale\b"
]

RESEARCHER_TERMS = [
    r"\bpaper\b", r"\bbenchmark\b", r"\bablation\b", r"\bnovel\b",
    r"\bwe propose\b", r"\bstate.of.the.art\b", r"\bneurips\b",
    r"\bicml\b", r"\biclr\b", r"\barxiv\b", r"\bacademic\b"
]

# System Semantics Patterns — broad functional descriptions of IR/ranking systems
# Catches plain-language fits who built the right systems without the fashionable keywords.
# The JD explicitly warns: "A candidate who built a recommendation system at a product company
# is a fit even if they never say RAG, Pinecone, or FAISS."
SYSTEM_SEMANTICS_PATTERNS = [
    # Marketplace and matching
    r"matching engine", r"candidate.job match", r"marketplace.*rank",
    r"job matching", r"candidate matching", r"talent matching",
    r"two.sided.*platform", r"supply.*demand.*match",
    # Feed and personalization
    r"feed rank", r"content rank", r"personali[sz]ation", r"personali[sz]ed feed",
    r"home.*feed", r"news.*feed.*rank", r"relevance.*feed",
    # Recommendation systems (broad)
    r"recomm.*system", r"recomm.*engine", r"collaborative.*filter",
    r"content.based.*filter", r"item.*embed", r"user.*embed",
    r"matrix.*factori", r"item2vec", r"user2item",
    # Search and retrieval (plain language)
    r"search.*engine", r"search.*pipeline", r"search.*infra",
    r"document.*retriev", r"query.*retriev", r"result.*rank",
    r"relevance.*engin", r"relevance.*score", r"relevance.*model",
    # Ranking systems (plain language)
    r"ranking.*model", r"ranking.*system", r"ranking.*pipeline",
    r"sort.*results", r"order.*results", r"scored.*results",
    # Scoring systems
    r"scoring.*model", r"candidate.*score", r"match.*score",
    r"fit.*score", r"relevance.*score", r"quality.*score"
]
```

### 7.3 Bucket A — Skill Evidence

Per-skill score 0–3:
- **0** = skill absent from profile
- **1** = skill mentioned in skills section only
- **2** = skill mentioned in career description (project-level evidence)
- **3** = skill mentioned in career description with production/scale signals

```python
TARGET_SKILLS = {
    "retrieval_search": RETRIEVAL_PATTERNS + [r"bm25"],
    "vector_db_hybrid": [r"vector database", r"hybrid search", r"dense retrieval", r"sparse retrieval",
                         r"embedding search", r"ann\b", r"approximate nearest"],
    "eval_framework": EVALUATION_PATTERNS,
    "ltr_reranking": RANKING_PATTERNS + [r"cross.encoder", r"bi.encoder"],
    "llm_integration": [r"llm", r"fine.tuning", r"lora", r"qlora", r"peft", r"rag",
                        r"retrieval augmented", r"prompt engineering"],
    # JD Must-Have: "Strong Python. Yes really, we care about code quality."
    # Detect Python use in career descriptions, not just skills section listing.
    "python_coding": [r"python", r"fastapi", r"flask", r"django", r"pyspark", r"asyncio",
                      r"pytest", r"type hints", r"mypy", r"poetry", r"pyproject"],
    # JD Nice-to-Haves
    "distributed_systems": [r"distributed system", r"inference optimization", r"tensorrt", r"vllm", r"triton", r"high throughput", r"large scale inference"],
    "hr_tech_exposure": [r"hr tech", r"hr.tech", r"recruiting tech", r"talent acquisition", r"applicant tracking", r"marketplace"]
}

def score_skill_bucket(candidate, career_text):
    scores = {}
    snippets = {}
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})

    for bucket_name, keywords in TARGET_SKILLS.items():
        skill_mentioned = any(
            any(kw in s["name"].lower() for kw in keywords)
            for s in candidate.get("skills", [])
        )
        career_evidence = []
        for kw in keywords:
            matches = list(re.finditer(kw, career_text, re.IGNORECASE))
            if matches:
                # Find a 60-char snippet around the first match
                idx = matches[0].start()
                snippet = career_text[max(0, idx-30):idx+60].strip()
                career_evidence.append(snippet)

        # Check for production signals localized to the extracted snippets
        # (instead of anywhere in the 10-year career history)
        has_production = any(
            re.search(p, snippet, re.IGNORECASE)
            for p in PRODUCTION_PATTERNS for snippet in career_evidence
        )

        # Determine score
        if career_evidence and has_production:
            score = 3
        elif career_evidence:
            score = 2
        elif skill_mentioned:
            score = 1
        else:
            score = 0

        # Assessment score boost: if candidate has a high verified score for a target skill
        for s in candidate.get("skills", []):
            if any(kw in s["name"].lower() for kw in keywords):
                asc = assessment_scores.get(s["name"])
                if asc is not None and asc >= 70 and score >= 1:
                    score = min(score + 0.5, 3)  # Boost but don't exceed 3

        scores[bucket_name] = score
        snippets[bucket_name] = career_evidence[0] if career_evidence else ""

    return scores, snippets
```

### 7.4 Bucket B — Career Quality

```python
def score_career_quality(candidate, career_text, flags):
    # Product ratio (from Phase 1 flags)
    product_ratio = flags.get("product_ratio", 0.5)

    # Deploy signal: mentions of users, production, scale, launch
    deploy_count = sum(
        1 for p in PRODUCTION_PATTERNS
        if re.search(p, career_text, re.IGNORECASE)
    )
    deploy_signal = min(deploy_count / 5.0, 1.0)

    # Experience recency: is the most recent role in a relevant domain?
    career = candidate.get("career_history", [])
    # Ensure career history is sorted by recency (assuming descending date order)
    # The competition JSONL typically has the current role at index 0.
    recent_role = career[0] if career else {}
    recent_desc = recent_role.get("description", "").lower()
    recent_relevant = any(
        re.search(p, recent_desc, re.IGNORECASE)
        for p in RETRIEVAL_PATTERNS + RANKING_PATTERNS
    )
    experience_recency = 1.0 if recent_relevant else 0.5

    # Depth signal: multiple roles with IR/retrieval work, not just one mention
    roles_with_retrieval = sum(
        1 for role in career
        if any(re.search(p, role.get("description", ""), re.IGNORECASE)
               for p in RETRIEVAL_PATTERNS + RANKING_PATTERNS)
    )
    depth_signal = min(roles_with_retrieval / 2.0, 1.0)

    # Search/Ranking/Recommendation System Experience Score
    # Evaluates direct evidence of having built search, ranking, or recommendation systems
    # from core pattern lists (including broad system semantics) combined with production/scale signals.
    has_sys_evidence = any(
        re.search(p, career_text, re.IGNORECASE)
        for p in RETRIEVAL_PATTERNS + RANKING_PATTERNS + RECOMMENDATION_PATTERNS + SYSTEM_SEMANTICS_PATTERNS
    )
    has_sys_production = has_sys_evidence and any(
        re.search(p, career_text, re.IGNORECASE) for p in PRODUCTION_PATTERNS
    )
    sys_experience_score = 1.0 if has_sys_production else (0.5 if has_sys_evidence else 0.0)

    # Shipper ratio: shipper vocabulary vs researcher vocabulary
    shipper_count = sum(1 for p in SHIPPER_TERMS if re.search(p, career_text, re.IGNORECASE))
    researcher_count = sum(1 for p in RESEARCHER_TERMS if re.search(p, career_text, re.IGNORECASE))
    total_vocab = shipper_count + researcher_count
    shipper_ratio = shipper_count / total_vocab if total_vocab > 0 else 0.5

    # Writing signal: avg length of career descriptions
    descriptions = [r.get("description", "") for r in career]
    avg_desc_len = sum(len(d) for d in descriptions) / len(descriptions) if descriptions else 0
    writing_signal = 1.00 if avg_desc_len >= 150 else (0.95 if avg_desc_len >= 60 else 0.90)

    # Product Builder Score — explicit composite of founding-team and shipping signals.
    # Computed here (Bucket B) so Phase 4 can use it as a first-class 20% scoring component.
    # The JD emphasises product-company background, shipping velocity, and ownership
    # at least as much as specific ML tool keywords (§1.2: shipper vs researcher distinction).
    _OWNERSHIP_PATTERNS = [
        r"built from scratch", r"founded", r"co-founder", r"led.*team",
        r"ownership", r"end.to.end", r"greenfield", r"zero to one"
    ]
    ownership_signal = any(re.search(p, career_text, re.IGNORECASE) for p in _OWNERSHIP_PATTERNS)
    product_builder_score = (
        0.35 * product_ratio +                       # Time-weighted product-company career fraction
        0.30 * deploy_signal +                       # Production/scale deployment language density
        0.20 * shipper_ratio +                       # Shipper vs researcher vocabulary ratio
        0.15 * (1.0 if ownership_signal else 0.0)   # End-to-end ownership / startup language
    )
    # Disqualifier multipliers: consulting/research backgrounds cannot score high as product builders
    if flags.get("consulting_only"):
        product_builder_score *= 0.4
    if flags.get("research_only"):
        product_builder_score *= 0.5
    if flags.get("wrong_domain"):
        product_builder_score *= 0.3

    return {
        "product_ratio": product_ratio,
        "deploy_signal": deploy_signal,
        "experience_recency": experience_recency,
        "depth_signal": depth_signal,
        "shipper_ratio": shipper_ratio,
        "writing_signal": writing_signal,
        "sys_experience_score": sys_experience_score,
        "product_builder_score": product_builder_score,
        "ownership_signal": ownership_signal
    }
```


### 7.5 Bucket C — JD Fit Gaps

```python
def score_fit_gaps(candidate, career_text, flags):
    # Title velocity: avg tenure < 18 months across 3+ roles
    # Exclude current role from average tenure calculation (accumulating) per contract instructions
    career = candidate.get("career_history", [])
    past_roles = career[1:] if len(career) > 1 else []
    valid_durations = [r.get("duration_months") for r in past_roles if r.get("duration_months") is not None]
    
    if len(past_roles) > 0 and len(valid_durations) == len(past_roles):
        avg_tenure = sum(valid_durations) / len(valid_durations)
        title_velocity_flag = (avg_tenure < 18.0) and (len(career) >= 3)
    else:
        # Missing token guardrails: fail open if durations are missing or only 1 career role exists
        title_velocity_flag = False

    # Consulting flag (from Phase 1 flags)
    consulting_flag = flags.get("consulting_only", False)

    # External validation: GitHub, papers, talks, open-source
    EXTERNAL_VALIDATION_TERMS = [
        r"open.source", r"github", r"published", r"publication", r"paper",
        r"conference", r"talk", r"speaker", r"blog", r"maintainer", r"contributor"
    ]
    signals = candidate.get("redrob_signals", {})
    github_score = signals.get("github_activity_score", -1)
    has_external_text = any(re.search(p, career_text, re.IGNORECASE) for p in EXTERNAL_VALIDATION_TERMS)
    external_validation = github_score > 0 or has_external_text

    # Code stopped: architect/VP/Director with yoe > 8 (likely stopped coding)
    yoe = candidate["profile"].get("years_of_experience", -1)
    current_title = candidate["profile"].get("current_title", "UNKNOWN").lower()
    STOPPED_CODING_TITLES = {"architect", "vp", "vice president", "director", "cto", "head of"}
    code_stopped = yoe > 8 and any(t in current_title for t in STOPPED_CODING_TITLES)

    # Seniority score: continuous float bands aligned with JD_contract.yaml and without gaps
    if 5.0 <= yoe < 10.0:
        seniority_score = 1.00   # Sweet spot (5.0 - 9.9)
    elif 4.0 <= yoe < 5.0:
        seniority_score = 0.95   # Slightly junior (4.0 - 4.9)
    elif 10.0 <= yoe < 13.0:
        seniority_score = 0.95   # Mild over-seniority (10.0 - 12.9)
    elif 0.0 <= yoe < 4.0:
        seniority_score = 0.75   # Junior / significant gap (0.0 - 3.9)
    elif yoe >= 13.0:
        seniority_score = 0.90   # Over-senior (13.0 - 99.0)
    else:
        seniority_score = 1.00   # Default/fallback

    # LangChain-only flag: JD says "if your AI experience consists primarily of recent
    # (under 12 months) projects using LangChain to call OpenAI, we will probably not move forward
    # unless you can demonstrate substantial pre-LLM ML production experience."
    # Detection: heavy LangChain/OpenAI wrapper vocabulary + short total AI skill durations
    FRAMEWORK_DEMO_TERMS = [
        r"langchain", r"openai.*api", r"chatgpt.*api", r"gpt.*wrapper",
        r"llamaindex", r"llama.index"
    ]
    PRE_LLM_PRODUCTION_TERMS = [
        r"faiss", r"elasticsearch", r"opensearch", r"bm25", r"xgboost.*rank",
        r"tensorflow.*serving", r"pytorch.*production", r"recommendation.*system",
        r"retrieval.*system", r"search.*engine"
    ]
    has_framework_demo = sum(
        1 for p in FRAMEWORK_DEMO_TERMS if re.search(p, career_text, re.IGNORECASE)
    ) >= 2
    has_pre_llm_production = any(
        re.search(p, career_text, re.IGNORECASE) for p in PRE_LLM_PRODUCTION_TERMS
    )
    # AI skill duration: sum months of LLM/AI skills claimed
    ai_skill_months = sum(
        s.get("duration_months", 0) for s in candidate.get("skills", [])
        if any(kw in s["name"].lower() for kw in ["llm", "gpt", "langchain", "openai", "ai"])
    )
    langchain_only_flag = has_framework_demo and not has_pre_llm_production and ai_skill_months < 12

    # Closed-source flag: 5+ years total experience without external validation
    # (open-source contributions, publications, talks, or GitHub activity)
    closed_source_flag = yoe >= 5 and not external_validation

    return {
        "title_velocity_flag": title_velocity_flag,
        "consulting_flag": consulting_flag,
        "external_validation": external_validation,
        "code_stopped": code_stopped,
        "seniority_score": seniority_score,
        "langchain_only_flag": langchain_only_flag,
        "closed_source_flag": closed_source_flag
    }
```

### 7.6 90-Day Plan Alignment Score

```python
def compute_ninety_day_alignment(bucket_a, product_ratio) -> float:
    """
    Computes a score in [0, 1] representing the candidate's alignment with the JD's 90-day plan:
    - Weeks 1-3: Audit BM25 / Retrieval (retrieval_search)
    - Weeks 4-8: Ship v2 ranker (vector database / hybrid search or learning-to-rank/reranking)
    - Weeks 9-12: Evaluation framework (NDCG/MRR/MAP/A-B testing)
    """
    m1 = bucket_a.get("retrieval_search", 0) / 3.0
    m2 = max(bucket_a.get("vector_db_hybrid", 0), bucket_a.get("ltr_reranking", 0)) / 3.0
    m3 = bucket_a.get("eval_framework", 0) / 3.0

    readiness = (m1 + m2 + m3) / 3.0

    # Boost for complete plan coverage, penalize for missing milestones entirely
    coverage = sum(1 for m in [m1, m2, m3] if m > 0)
    if coverage == 3:
        readiness = min(readiness + 0.15, 1.0)
    elif coverage == 1:
        readiness = max(readiness - 0.10, 0.0)
    elif coverage == 0:
        readiness = 0.0

    # Product company exposure weights candidate's ability to execute a plan in a real startup environment
    alignment = 0.8 * readiness + 0.2 * product_ratio
    return round(alignment, 4)
```

### 7.7 Behavioral signal extraction

```python
from datetime import date

def extract_behavioral(candidate, reference_date) -> dict:
    signals = candidate.get("redrob_signals", {})

    # NOTE: days_inactive is NOT computed here.
    # The raw last_active_date string is passed through so that rank.py can
    # compute (reference_date - last_active_date).days dynamically at rank time,
    # using the guard: reference_date = max(stored_date, max(candidates_last_active_dates)).
    # This prevents negative inactivity values when the sandbox receives candidates
    # with dates newer than the precompute run.

    return {
        "last_active_date": signals.get("last_active_date", None),   # Raw string; computed to days_inactive at rank time
        "open_to_work": signals.get("open_to_work_flag", False),
        "recruiter_response_rate": signals.get("recruiter_response_rate", 0.5),
        "avg_response_time_hours": signals.get("avg_response_time_hours", 24.0),
        "notice_period_days": signals.get("notice_period_days", 60),
        "interview_completion_rate": signals.get("interview_completion_rate", 0.5),
        "offer_acceptance_rate": signals.get("offer_acceptance_rate", -1),
        "github_activity_score": signals.get("github_activity_score", -1),
        "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d", -1),
        "endorsements_received": signals.get("endorsements_received", -1),
        "applications_submitted_30d": signals.get("applications_submitted_30d", -1),
        "profile_completeness_score": signals.get("profile_completeness_score", -1.0),
        "verified_email": signals.get("verified_email", False),
        "verified_phone": signals.get("verified_phone", False),
        "linkedin_connected": signals.get("linkedin_connected", False),
        "willing_to_relocate": signals.get("willing_to_relocate", False),
        "preferred_work_mode": signals.get("preferred_work_mode", "UNKNOWN"),
        "location": candidate["profile"].get("location", "UNKNOWN").lower(),
        "country": candidate["profile"].get("country", "UNKNOWN").lower(),
    }
```

> **Runtime note:** `behavioral.py` resolves `days_inactive` at rank time:
> ```python
> from datetime import date
> ref = reference_date  # loaded from artifacts/run_metadata.json, then max'd against candidate pool
> last_active_str = behavioral.get("last_active_date")
> days_inactive = (ref - date.fromisoformat(last_active_str)).days if last_active_str else 180
> ```

### 7.8 Candidate Features Parquet Schema

`artifacts/candidate_features.parquet` is the central feature store containing all offline-computed signals.

| Column | Type | Notes |
|---|---|---|
| `candidate_id` | string | Primary key |
| `retrieval_search` | float | Bucket A score (0–3) |
| `vector_db_hybrid` | float | Bucket A score (0–3) |
| `eval_framework` | float | Bucket A score (0–3) |
| `ltr_reranking` | float | Bucket A score (0–3) |
| `llm_integration` | float | Bucket A score (0–3) |
| `python_coding` | float | Bucket A score (0–3) |
| `distributed_systems` | float | Bucket A score (0–3) |
| `hr_tech_exposure` | float | Bucket A score (0–3) |
| `experience_recency` | float | Bucket B recency signal |
| `depth_signal` | float | Bucket B depth signal |
| `sys_experience_score` | float | Bucket B system evidence |
| `product_builder_score`| float | Bucket B composite (normalized [0,1]) |
| `seniority_score` | float | Bucket C (from `score_fit_gaps`) |
| `snippets_json` | string | JSON dict of the best 60-char evidence snippets |

---

