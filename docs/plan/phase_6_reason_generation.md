## 10. Phase 6 — Reason Generation

### 10.1 Requirements

The Stage 4 review samples 10 random rows and checks for:
- Specific facts from the candidate's profile
- Connection to JD requirements
- Acknowledgment of gaps (mandatory for ranks 50+)
- No hallucinated claims
- Structural variation across entries
- Rank-appropriate tone
- **Downstream Independence:** Explanations are strictly computed after all scores and ranks have been finalized. The reason generation step must never feedback into or influence candidate scores or final rank ordering.

### 10.2 90-day plan milestone framing

The JD describes three milestones for the first 90 days. Map each candidate's strongest evidence to the milestone they are best positioned for:

- **Weeks 1-3** (Audit BM25/retrieval): Strong retrieval evidence or BM25/search infrastructure history
- **Weeks 4-8** (Ship v2 hybrid ranker): Strong vector DB + production deployment evidence
- **Weeks 9-12** (Build evaluation framework): Strong NDCG/MRR/A-B testing evidence

This framing shows Stage 4 reviewers that the system understood the JD at a human level, not a keyword level.

### 10.3 Generator function

### 10.3 Generator architecture: Evidence-driven lead selection

Fixed templates fail Stage 4 not because of wording but because of **information ordering** — template systems always present signals in the same sequence regardless of which signals are strongest for that specific candidate. Reviewers see through this.

The correct architecture: the **strongest domain signal per candidate leads the sentence**. A candidate whose top signal is marketplace ranking gets a completely different opening than one whose top signal is evaluation metrics infrastructure — not synonym swapping, but different facts leading.

```python
STRENGTH_DOMAINS = [
    "retrieval_systems",       # FAISS, vector DB, hybrid search, ANN
    "recommendation_systems",  # RecSys, collaborative filtering, matching engines
    "marketplace_ranking",     # two-sided marketplace, feed ranking, job/candidate matching
    "evaluation_metrics",      # NDCG, MRR, MAP, A/B testing, offline-online eval
    "product_ml",              # shipping to real users, production deployment, latency
    "startup_scale",           # founding team, 0-to-1 builds, startup product experience
    "vector_infrastructure",   # Pinecone, Qdrant, Milvus, Weaviate, embedding drift
]

def generate_reasoning(candidate_data: dict) -> str:
    """
    Evidence-driven lead selection with safe sentinel handling.
    Three rules:
    1. Lead varies because strongest domain varies per candidate.
    2. Every proper noun and number comes from the actual profile JSON (no hallucination).
    3. Missing data (sentinels: -1, "UNKNOWN") triggers safe fallback templates.
    """
    domain_scores = candidate_data["domain_scores"]  # dict[str, float] from Phase 3
    candidate_facts = candidate_data["candidate_facts"]  # extracted facts per domain
    flags = candidate_data["flags"]
    behavioral = candidate_data["behavioral"]

    # Safely filter facts to ignore sentinel values (-1, "UNKNOWN")
    def safe_facts(domain: str) -> dict:
        facts = candidate_facts.get(domain, {})
        return {k: v for k, v in facts.items() if v not in (-1, "UNKNOWN")}

    # Rank domains by evidence score
    ranked_domains = sorted(STRENGTH_DOMAINS, key=lambda d: domain_scores.get(d, 0.0), reverse=True)
    primary = ranked_domains[0]
    secondary = ranked_domains[1] if len(ranked_domains) > 1 else None

    # Lead sentence: safe formatting
    primary_facts = safe_facts(primary)
    try:
        lead = LEAD_TEMPLATES[primary].format(**primary_facts)
    except KeyError:
        lead = f"Candidate shows strong evidence in {primary.replace('_', ' ')}."

    # Support sentence: secondary strength or 90-day milestone framing
    support = ""
    if secondary and domain_scores.get(secondary, 0.0) > 0.2:
        secondary_facts = safe_facts(secondary)
        try:
            support = SUPPORT_TEMPLATES[secondary].format(**secondary_facts)
        except KeyError:
            support = ""
            
    if not support:
        # Fall back to 90-day milestone framing if no strong secondary signal
        milestone = get_90day_milestone(primary)
        support = f"Best positioned for {milestone} mandate."

    # Concern sentence: only when genuinely present
    concern = get_largest_concern(candidate_data)
    if concern:
        concern_facts = safe_facts(concern)
        try:
            caveat = CONCERN_TEMPLATES[concern].format(**concern_facts)
        except KeyError:
            caveat = f"Note: Potential gap identified in {concern.replace('_', ' ')}."
    else:
        caveat = ""

    parts = [p for p in [lead, support, caveat] if p]
    return " ".join(parts[:2])  # Spec: 1-2 sentences


def get_largest_concern(candidate_data: dict) -> str | None:
    """
    Returns the name of the largest gap/concern, or None if no real concern exists.
    DO NOT emit a concern just to hit a target percentage.
    Reviewers care about honest concerns, not equal concern distribution.
    """
    CONCERN_THRESHOLD = 0.4  # Minimum gap score to surface as a concern
    concern_scores = candidate_data.get("concern_scores", {})
    if not concern_scores:
        return None
    best = max(concern_scores, key=lambda k: concern_scores[k])
    return best if concern_scores[best] > CONCERN_THRESHOLD else None
```

**What creates genuine variation:** Different candidates have different primary domains. A RecSys engineer leads with marketplace matching facts. An IR specialist leads with retrieval infrastructure facts. The sentence structure may share a template, but the facts are different, the domain is different, and the ordering is different.

**Hallucination prevention:** Every value passed to `.format(**candidate_facts[domain])` must be extracted directly from the candidate JSON in Phase 3 — company names, system names, duration numbers, endorsement counts. Never infer or invent.

### 10.4 Rank-dependent tone

| Rank Range | Template Tone | Mandatory Elements |
|---|---|---|
| 1–30 | Strong positive; lead with best evidence | Snippet + company/scale signal; positive behavioral if strong |
| 31–70 | Neutral; evidence + one concern if present | Snippet; concern if notice/consulting/inactive |
| 71–100 | Honest gap acknowledgment mandatory | At least one concern; "limited evidence" if no snippet |

### 10.5 Debugging Outputs (Offline only)

For troubleshooting and manual verification, the ranking engine will save a separate debug file (`artifacts/ranking_debug.csv`) alongside the official submission CSV. This file will contain:
- `candidate_id`
- `rank`
- `score`
- `reasoning`
- `primary_strength`: The primary domain identifier leading the explanation (e.g. `retrieval_systems`)
- `secondary_strength`: The secondary domain identifier (or `None` / `milestone`)
- `concern`: The identifier of the largest concern (or `None`)

This is exclusively for developer inspection and manual audits of the top 100 candidates; the official submission CSV (`BuriBuri.csv`) will remain strictly formatted with only the required columns (`candidate_id,rank,score,reasoning`).

---

