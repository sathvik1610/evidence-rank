# Evidence-Rank: Intelligent Candidate Discovery & Ranking Engine

**Hackathon:** Redrob × Hack2skill — Intelligent Candidate Discovery & Ranking Challenge  
**Role:** Senior AI Engineer, Founding Team, Redrob AI  
**Task:** Rank 100,000 candidates against a job description. Output the top 100, ranked best-fit first, with a 1–2 sentence explanation for each.

---

## The Core Philosophy

A great recruiter does not pick the candidate with the most AI keywords. They balance many dimensions simultaneously — technical fit, career quality, practical availability, logistical fit, culture signals — and weigh each one according to how much it actually matters for this specific role.

This system replicates that judgment. Every JD dimension is considered at a different pipeline stage, with weights calibrated to the JD's actual language.

The dimensions, in rough priority order:

1. **Technical Fit** — Evidence (not claims) of must-have skills in career descriptions
2. **Career Quality** — Product company background, production deployment, shipper attitude
3. **JD Disqualifiers** — Hard negatives: pure research, consulting-only, wrong domain, LangChain-only
4. **Seniority** — 5-9 year sweet spot, soft taper outside the band
5. **Behavioral Availability** — Can the platform actually reach and hire this person?
6. **Logistics** — Location, notice period, work mode preference
7. **Culture Fit** — Writing habit, startup/ownership mindset, async communication
8. **Delighters** — Open-source, LTR experience, HR-tech exposure, recruiter saves

No single dimension wins alone. A technically perfect candidate with a 3% recruiter response rate is not a real hire. A highly available candidate in Pune with zero retrieval evidence is not a fit. The system balances all of them.

**Where the 90-Day Plan comes in:**
The JD's first 90 days — Weeks 1-3 (Audit BM25/retrieval), Weeks 4-8 (Ship v2 hybrid ranker), Weeks 9-12 (Build evaluation framework) — are encoded into the **90-Day Plan Alignment Score** ($M_{\text{90day}}$). This score directly ranks candidates by their evidence for executing these exact milestones. It is also used to map each candidate's profile to their best-suited milestone in Phase 6 (Reason Generation), showing Stage 4 reviewers that the system understood the JD at a human level.

**Honoring "Let's be honest about this role":**
The JD explicitly states: *"This role writes code. Senior engineers who haven't written production code in the last 18 months because they've moved into 'architecture' or 'tech lead' roles — we will probably not move forward."* and *"We work async-first and write a lot."* 
Our system directly operationalizes these constraints:
1. **Writing Signal:** Calculates the average length of career descriptions. Candidates with minimal or blank roles get a culture mismatch penalty (up to ×0.70).
2. **Code-Stopped Penalty:** High YoE candidates with pure architecture/VP titles who likely stopped coding get a ×0.75 penalty.
3. **Python Quality:** Bucket A searches for active coding evidence (`fastapi`, `flask`, `pytest`, `asyncio`, type hints, etc.), not just listed skills.

---

## The Hard Constraints

The ranking step must run in under 5 minutes on a standard CPU with 16 GB RAM, no internet, no GPU. This forces a clean two-phase design:

| Phase | When | Time Limit |
|---|---|---|
| **Offline Precompute** (Phases 0–1) | Run once before submission | No limit |
| **Runtime Ranking** (Phases 2–6) | Run at evaluation time | ≤ 5 minutes |

---

## System Overview

```
JD → [Phase 0: JD Intelligence] → Structured Config + Two JD Vectors + BM25 Keywords
Corpus → [Phase 1: Preprocessing] → FAISS Index + BM25 Index + Honeypot/Ghost Flags
                                              ↓ (runtime starts here)
                              [Phase 2: Retrieval] → 3,000–5,000 candidates
                                              ↓
                         [Phase 3: Feature Extraction] → Evidence Buckets A/B/C + Consistency Score
                                              ↓
                          [Phase 4: Core Scoring + Rerank] → Top 200–300
                                              ↓
                    [Phase 5: Behavioral & Logistic Adjustment] → Top 100
                                              ↓
                         [Phase 6: Reason Generation] → Final CSV
                                              ↓
                    [Phase 7: Manual Validation] → Submit
```

---

## Phase 0 — JD Intelligence
*Offline. Runs once.*

The JD is decoded into five typed signal groups:

| Signal Group | Content |
|---|---|
| **Must-Haves** | Production retrieval systems, vector DBs/hybrid search, strong Python, evaluation frameworks (NDCG/MRR/MAP/A-B) |
| **Nice-to-Haves** | LLM fine-tuning (LoRA/QLoRA), learning-to-rank, HR-tech/marketplace exposure, distributed systems, open-source |
| **Hard Disqualifiers** | Pure research (no production deployment), LangChain-only AI (<12 months), no code written in 18 months, CV/speech without NLP |
| **Soft Negatives** | Consulting-only career, title-chasers, framework demo enthusiasts, closed-source-only 5+ years |
| **Logistics & Culture** | Location (Pune/Noida preferred), notice period (sub-30 ideal), async writing culture, startup/ownership mindset |

The JD is embedded into **two separate semantic vectors** using BGE-M3:
1. **Skills-Focused Vector** — dense on technical vocabulary (FAISS, NDCG, hybrid search, retrieval, etc.)
2. **Synthetic Ideal Candidate Vector** — embeds the JD's own plain-language description of the ideal hire: *"6-8 years, 4-5 at product companies, shipped one end-to-end ranking/search/recsys to real users, strong opinions about retrieval and evaluation..."*

Two vectors catch two different types of candidates: technically-worded profiles and plain-language career narratives.

A BM25 keyword list covers exact technical terms for lexical retrieval.

**Tools:** `BAAI/bge-m3`

---

## Phase 1 — Corpus Preprocessing + Honeypot Detection
*Offline. Runs once on all 100,000 candidates.*

**1. Embedding all 100K profiles**
Every candidate is flattened into a text string (title + headline + summary + career history + skills) and embedded with BGE-M3. Stored in a FAISS `IndexFlatIP`.

**2. BM25 index**
A BM25 (Okapi BM25) index is built over all 100K candidate texts for lexical retrieval.

**3. Honeypot Detection**
~80 deliberately fabricated profiles exist in the dataset with impossible timelines. Four checks flag them:
- Skill claimed for more months than total career length
- Expert/advanced skill with zero months of usage
- Expert/advanced skill with a platform assessment score below 40
- Single job role lasting longer than the candidate's total career

Honeypots are **hard-zeroed** (score = 0.0). They never appear in the top 100.

**4. Ghost Profile Pre-filter**
Profiles that are completely unreachable are flagged: inactive >365 days AND recruiter response rate <5% AND not open to work AND zero applications in 30 days. These are scored 0.0 — they won't be hired regardless of fit.

**5. Disqualifier Tagging**
Lightweight flags saved per candidate:
- `consulting_only` — entire career at IT services firms
- `research_only` — only academic/research titles, no engineering roles
- `wrong_domain` — CV/speech/robotics expertise without any NLP or search background

Used as soft multipliers in Phase 5.

**Tools:** `BAAI/bge-m3`, `faiss-cpu`, `rank_bm25`, `pandas`, `pyarrow`

---

## Phase 2 — Multi-Signal Retrieval
*Runtime. Target: under 30 seconds.*

Three retrievers run and merge to produce 3,000–5,000 candidates.

**Retriever 1: Dense Semantic Search (×2)**
FAISS cosine-similarity against both JD vectors. Top 2,000 from skills-focused + top 2,000 from ideal-candidate vector. Catches both technically-worded and plain-language profiles.

**Retriever 2: Lexical BM25 Search**
Exact-match retrieval for specific rare technical terms — "NDCG", "Pinecone", "A/B test" — that embeddings can dilute.

**Merging: Reciprocal Rank Fusion (RRF)**

$$\text{RRF Score}(d) = \sum_i \frac{1}{k + \text{rank}_i(d)}, \quad k = 60$$

A candidate appearing across all three retrievers outscores one appearing in only one. This rewards multi-dimensional evidence.

Active job-seekers (`open_to_work` or recently active) get a small RRF boost so strong active candidates aren't buried behind stale passive ones.

**Tools:** `faiss-cpu`, `rank_bm25`, `BAAI/bge-m3`

---

## Phase 3 — Candidate Feature Extraction
*Runtime. Target: under 1 minute on 3,000–5,000 candidates.*

Rule-based extraction produces four evidence buckets. All verbatim evidence snippets from career descriptions are stored here for use in Phase 6 reasoning.

---

### Bucket A — Technical Fit (The Must-Haves)

Each JD-critical skill domain gets an evidence score from 0 to 3:

| Score | Meaning |
|---|---|
| **0** | Not present anywhere |
| **1** | Mentioned in skills section only |
| **2** | Appears in career description (project evidence) |
| **3** | Career description + production/scale signals |

Platform assessment scores (verified by Redrob) boost a skill toward score 3 when the candidate passed a test on that skill.

**Domains scored:**

| Domain | JD Basis |
|---|---|
| Retrieval / Search systems | Must-have: production retrieval |
| Vector DB / Hybrid search | Must-have: vector DB operational experience |
| Evaluation frameworks (NDCG/MRR/MAP/A-B) | Must-have: rigorous ranking evaluation |
| Python code quality | Must-have: "strong Python, we care about code quality" |
| Learning-to-rank & reranking | Nice-to-have |
| LLM integration & fine-tuning | Nice-to-have |

**System Semantics Patterns** extend the retrieval/ranking detection to plain-language descriptions of the right kind of work: "matching engine", "personalization system", "feed ranking", "relevance model", "candidate-job matching" — catching candidates who built the right systems without fashionable keywords.

---

### Bucket B — Career Quality

| Signal | What it measures |
|---|---|
| `product_ratio` | Time-weighted fraction of career at product companies vs. consulting |
| `deploy_signal` | Density of production/scale language in career descriptions |
| `experience_recency` | Is the most recent role in a relevant technical domain? |
| `depth_signal` | IR/retrieval evidence across multiple roles, not just one mention |
| `shipper_ratio` | Ratio of shipper vocabulary ("shipped", "launched", "deployed", "real users") to researcher vocabulary ("paper", "benchmark", "ablation", "NeurIPS") — directly operationalizes the JD's preference for shippers over researchers |
| `writing_signal` | Average career description length — the JD says "we write a lot; writing-averse candidates will struggle" |
| `ownership_signal` | Language suggesting building from scratch, end-to-end ownership, startup-style delivery |

---

### Bucket C — JD Fit Gaps

| Signal | What it detects |
|---|---|
| `title_velocity_flag` | Average tenure <18 months across 3+ roles (title-chaser pattern) |
| `code_stopped` | Architect/VP/Director title with 8+ years YoE (stopped writing code) |
| `external_validation` | GitHub activity, publications, talks, open-source contributions |
| `seniority_score` | Soft window: 5-9 years = 1.0; 4 years = 0.80; 10-12 years = 0.85; outside these = soft taper. YoE is not a hard cutoff per JD. |

---

### Consistency Score

Detects the most common form of keyword stuffing: claiming advanced/expert proficiency in JD-relevant skills with no career evidence to support it.

$$\text{Consistency} = 1.0 - 0.7 \times \frac{\text{contradicted expert/advanced claims}}{\text{total expert/advanced target-skill claims}}$$

A contradiction is flagged when a candidate claims expert/advanced on a JD-relevant skill but:
1. Their career titles suggest a completely different domain (e.g., all titles are "BI Analyst" but claims "NLP: Expert")
2. The skill duration is under 12 months
3. The skill term never appears in any career description

Examples this catches:
- Claims "NLP: Expert" → entire career as BI Analyst
- Claims "LLM fine-tuning: Expert" → skill duration is 3 months
- Claims "Retrieval systems: Advanced" → no retrieval language in any job description

Score of 1.0 = fully consistent. Score of 0.30 = most claims are contradicted.

**Tools:** `regex`, phrase dictionaries (pure Python).

---

## Phase 4 — Core Scoring + Cross-Encoder Rerank
*Runtime. Target: 2.5–3 minutes.*

### Weighted Formula

$$\text{Core Score} = 0.60 \times S_{\text{must-have}} + 0.20 \times S_{\text{nice-to-have}} + 0.20 \times S_{\text{career quality}}$$

**Must-Have Score (60%):**

$$S_{\text{must-have}} = 0.25 \times S_{\text{retrieval}} + 0.20 \times S_{\text{vectordb}} + 0.15 \times S_{\text{eval}}$$

Hard rule: if retrieval or vector DB evidence is zero, the must-have score is capped at 0.5. These are non-negotiable per the JD.

**Nice-to-Have Score (20%):**
LLM fine-tuning, learning-to-rank, HR-tech/marketplace exposure.

**Career Quality Score (20%):**
Product ratio, deployment signal, search/ranking/recommendation production experience score, shipper ratio. Multiplied down by consulting-only (×0.4), research-only (×0.5), or wrong-domain (×0.3) penalty.

### Cross-Encoder Rerank

The top 300 candidates by core score are re-scored by `BAAI/bge-reranker-v2-m3`. Unlike the bi-encoder in FAISS, a cross-encoder sees both the JD and the candidate simultaneously, capturing relevance nuances cosine similarity misses.

$$\text{Phase 4 Score} = 0.80 \times \text{Core Score} + 0.20 \times \text{Cross-Encoder Score}$$

The 80/20 split keeps handcrafted features dominant — they encode behavioral, career, and logistical signals the cross-encoder cannot see. The cross-encoder acts as a semantic tie-breaker.

**Tools:** `BAAI/bge-reranker-v2-m3`

---

## Phase 5 — Behavioral & Logistic Adjustment
*Runtime. Target: under 5 seconds.*

Phase 4 produces a technical fit score. Phase 5 converts it into a practical hiring score by multiplying in behavioral, logistical, and culture signals.

$$\text{Final Score} = \text{Phase4Score} \times M_{\text{availability}} \times M_{\text{notice}} \times M_{\text{location}} \times M_{\text{seniority}} \times M_{\text{penalties}} \times M_{\text{consistency}} \times M_{\text{writing}} \times M_{\text{90day}} + B_{\text{social}}$$

---

**Availability Multiplier**

Built from `last_active_date`, `recruiter_response_rate`, and `open_to_work_flag`.

| Condition | Multiplier |
|---|---|
| Active ≤30d + response ≥70% + open to work | ×1.15 |
| Active ≤90d + response ≥50% | ×1.05 |
| Active >180d or response <15% | ×0.50 |
| All others | ×0.90 |

---

**Notice Period Modifier**

From JD: *"sub-30 preferred; we can buy out up to 30 days; bar gets higher after 30"*

| Notice | Multiplier |
|---|---|
| 0–30 days | ×1.00 |
| 31–60 days | ×0.95 |
| 61–90 days | ×0.90 |
| 91–120 days | ×0.85 |
| 120+ days | ×0.75 |

---

**Location Modifier**

From JD: *"Pune/Noida preferred. Hyderabad, Mumbai, Delhi NCR welcome. Outside India: case-by-case."*

| Candidate Location | Willing to Relocate? | Multiplier |
|---|---|---|
| Pune / Noida / Delhi NCR (Local) | Any / N/A | ×1.00 |
| Welcome Cities (Hyderabad, Mumbai) | Yes | ×1.00 |
| Welcome Cities (Hyderabad, Mumbai) | No | ×0.98 |
| India Adjacent (Bangalore, Chennai, etc.) | Yes | ×0.98 |
| India Adjacent (Bangalore, Chennai, etc.) | No | ×0.95 |
| Other India | Yes | ×0.95 |
| Other India | No | ×0.92 |
| Outside India | Yes | ×0.90 |
| Outside India | No | ×0.85 |

---

**Seniority Modifier**

From JD: *"5-9 years is a range, not a requirement. We'll seriously consider candidates outside the band if other signals are strong."*

| YoE | Multiplier |
|---|---|
| 5–9 years | ×1.00 (sweet spot) |
| 4 years | ×0.85 (acceptable if other signals strong) |
| 10–12 years | ×0.90 (mild over-seniority) |
| < 4 years | ×0.65 (significant gap) |
| > 12 years | ×0.80 (likely over-senior for founding-team dynamic) |

---

**90-Day Plan Alignment Modifier ($M_{\text{90day}}$)**

From JD: *"who has the evidence that they can actually do this job in the first 90 days?"*
Scores readiness across all three milestones (retrieval search, hybrid vector database/ranking, and evaluation frameworks) combined with product company exposure:

$$\text{Alignment Score} = 0.8 \times \text{Milestone Readiness} + 0.2 \times \text{Product Company Ratio}$$
$$M_{\text{90day}} = 0.85 + 0.25 \times \text{Alignment Score}$$

This ranges from **×0.85** (low alignment) to **×1.10** (perfect coverage across all milestones + product background).

---

**Writing Culture Modifier ($M_{\text{writing}}$)**

From JD: *"We work async-first and write a lot. If you find writing painful, you'll find this role painful."*
Computed based on the average length of the candidate's career role descriptions to penalize empty or minimal job listings:

| Avg description length | Multiplier |
|---|---|
| ≥ 150 characters | ×1.00 (Healthy writing habit) |
| 60–149 characters | ×0.95 (Concise descriptions) |
| < 60 characters | ×0.90 (High culture risk / blank descriptions) |

---

**Soft Penalty Multipliers**

All soft — the JD uses *"probably not move forward"*, not *"auto-reject"*. Only honeypots and ghosts are zeroed.

| Signal | Multiplier | JD Language |
|---|---|---|
| Title-chaser (avg tenure <18mo, 3+ jobs) | ×0.70 | "not a fit" |
| Code-stopped (Architect/VP + YoE >8) | ×0.75 | "probably not move forward" (writes code) |
| LangChain-only AI (<12 months, no pre-LLM) | ×0.45 | "probably not move forward" |
| Closed-source only (5+ years, no validation) | ×0.80 | "do NOT want... without external validation" |
| Research-only background | ×0.40 | "will not move forward" — strongest language |
| Wrong domain (CV/speech, no NLP) | ×0.50 | "re-learning fundamentals" |
| Remote-only preference (hybrid role) | ×0.85 | Role is hybrid Tue/Thu |
| Consistency Score | ×[0.30–1.00] | Keyword stuffer detection |
| Consulting-only career | ×0.40 applied to career quality score | "bad fit in both directions" |

---

**Social Proof Boost** *(additive)*

| Signal | Boost | Rationale |
|---|---|---|
| `saved_by_recruiters_30d` > 5 | +0.04 | Human-curated market validation — other recruiters already shortlisted them |
| `github_activity_score` > 60 | +0.03 | JD values external validation for the role |
| `endorsements_received` > 20 | +0.01 | Peer credibility signal |
| `linkedin_connected` | +0.01 | Basic platform legitimacy |

---

## Phase 6 — Reason Generation
*Runtime. Under 5 seconds. No LLM calls.*

Reasons are assembled deterministically from verbatim evidence snippets stored in Phase 3. No language model is called at runtime.

**Rank-dependent tone:**

| Rank | Tone | Required |
|---|---|---|
| 1–30 | Strong positive | Best snippet + which 90-day milestone they cover + positive behavioral if strong |
| 31–70 | Neutral | Evidence + one concern if present |
| 71–100 | Honest gaps mandatory | At least one concern; "limited verifiable evidence" if no snippet |

**90-Day Milestone Framing:**
Each candidate's strongest evidence maps to the milestone they are best positioned for:
- Strong retrieval/BM25 experience → *"suited for Weeks 1-3 retrieval audit"*
- Vector DB + production deployment → *"suited for Weeks 4-8 hybrid ranker mandate"*
- NDCG/A-B evaluation evidence → *"suited for Weeks 9-12 evaluation framework mandate"*

This framing shows Stage 4 reviewers that the system understood the JD at a human level.

**Tools:** Pure Python string formatting.

---

## Phase 7 — Manual Validation
*Pre-submission.*

1. **Top-20 manual review** — Would a recruiter agree this person fits?
2. **Obvious-fit audit** — Find clear fits (ML Engineer + FAISS + evaluation + product company). If not in top 10, trace the failure and fix it. NDCG@10 = 50% of score.
3. **Honeypot audit** — No honeypots in top 10.
4. **Reasoning sample** — 10 rows across ranks; all claims must exist in the profile.
5. **Format validation** — `validate_submission.py` must print `Submission is valid.`

---

## Scoring Formula (Hackathon Evaluation)

$$\text{Composite} = 0.50 \times \text{NDCG@10} + 0.30 \times \text{NDCG@50} + 0.15 \times \text{MAP} + 0.05 \times \text{P@10}$$

NDCG@10 dominates at 50%. The top 10 picks matter more than the rest of the list combined.

---

## Models & Libraries

| Component | Model / Library | Size | Purpose |
|---|---|---|---|
| Embedding | `BAAI/bge-m3` | 570 MB | Dense + sparse profile and JD embedding |
| Vector Index | `faiss-cpu` | ~300 MB | Fast cosine similarity on 100K vectors |
| Lexical Retrieval | `rank_bm25` | — | Exact-term BM25 matching |
| Cross-Encoder | `BAAI/bge-reranker-v2-m3` | 130 MB | Query-document reranking on top 300 |
| Feature Extraction | `regex`, phrase dictionaries | — | Pure Python pattern matching. No NLP library dependency. |
| Data | `pandas`, `pyarrow` | — | Feature storage |
| Sandbox | `gradio` | — | HuggingFace Spaces demo |

No LLM is called at ranking time. All model inference happens in offline precompute. Runtime is fully deterministic, CPU-bound, and network-free.

---

## Reproducibility

```bash
# Step 1: Precompute (run once, no time limit)
python preprocess.py --candidates ./candidates.jsonl

# Step 2: Rank (must complete in under 5 minutes)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Step 3: Validate
python validate_submission.py submission.csv
```

---

## What Makes This Different

| Common Approach | This System |
|---|---|
| Keyword match on skills section | Evidence from career descriptions (0-3 per domain) |
| Single JD vector | Two vectors: technical vocabulary + ideal-candidate narrative |
| One-size-fits-all scoring | Phased: technical → behavioral → logistic → culture |
| Hard disqualifier rejections | Soft multipliers; only honeypots/ghosts zeroed |
| Treats all YoE equally | Soft seniority window with taper, not hard cutoff |
| Generic reasoning templates | Verbatim career evidence + 90-day milestone framing |
| Keyword stuffers pass | Consistency Score: expert claims × career evidence cross-check |
| Treats availability as optional | Multiplicative: unreachable candidates can't rank high regardless of fit |
