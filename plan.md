# Intelligent Candidate Discovery & Ranking Engine
## Production System Specification — Version 3.3.0

**Project:** Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge  
**Role Being Ranked For:** Senior AI Engineer, Founding Team, Redrob AI  
**Dataset:** 100,000 candidates in `candidates.jsonl` or `candidates.jsonl.gz`  
**Output:** `submission.csv` — top 100 candidates, ranked best-fit first  
**Hard Constraints:** 5-minute wall-clock execution (ranking step only), CPU-only, 16 GB RAM, no network calls during ranking

---

## Table of Contents

1. Problem Understanding & Design Philosophy
2. System Architecture Overview
3. Repository Layout
4. Phase 0 — JD Intelligence
5. Phase 1 — Corpus Preprocessing + Honeypot Detection
6. Phase 2 — Multi-Signal Retrieval
7. Phase 3 — Candidate Feature Extraction
8. Phase 4 — Core Scoring + Cross-Encoder Rerank
9. Phase 5 — Behavioral Re-ranking + Penalization
10. Phase 6 — Reason Generation
11. Phase 7 — Manual Validation + Weight Tuning
12. Data Field Reference
13. Model & Library Choices
14. Dependency Declarations

---

## 1. Problem Understanding & Design Philosophy

### 1.1 What the hackathon is actually testing

The challenge is not asking you to find candidates who contain the most AI keywords. It is asking you to reason about candidates the way a skilled recruiter would. The JD for Senior AI Engineer at Redrob AI is unusually honest about what it wants and — more importantly — what it does not want.

The scoring weights confirm this: NDCG@10 carries 50% of the total score. Your top 10 picks matter more than everything else combined.

### 1.2 What the JD actually means

The JD explicitly tells participants: *"The right answer involves reasoning about the gap between what the JD says and what the JD means."*

A Tier 5 candidate may not use the words "RAG" or "Pinecone" in their profile, but if their career history shows they built a recommendation system at a product company, they are a fit. A candidate who has all the AI keywords listed as skills but whose title is "Marketing Manager" is not a fit.

**The ideal candidate (JD's own words):**
- 6-8 years total experience, 4-5 at product companies in applied ML roles
- Has shipped at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale
- Has strong opinions about retrieval (hybrid vs dense), evaluation (offline vs online), LLM integration — and can defend them with reference to systems they actually built
- Located in or willing to relocate to Noida or Pune
- Active on the Redrob platform

**Hard disqualifiers (from JD text):**
- Pure research background with zero production deployment
- "AI experience" consisting primarily of LangChain/OpenAI wrappers with under 12 months of use and no pre-LLM ML background
- Senior engineer who has not written production code in the last 18 months
- Entire career at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, HCL, etc.) with zero product company exposure
- Computer vision, speech, or robotics specialists without significant NLP/IR exposure
- 5+ years entirely on closed-source proprietary systems without papers, talks, or open-source

**The JD's critical culture signal:**
The JD says: *"We work async-first and write a lot. If you find writing painful, you'll find this role painful."* Candidates with blank or minimal job descriptions are a cultural mismatch regardless of their skill set.

**The shipper vs researcher distinction:**
The JD explicitly says it tilts toward shippers over researchers. Candidates whose career language is full of "shipped," "launched," "deployed," "real users," "production" rank above equally skilled candidates whose language is "proposed," "benchmark," "ablation study," "novel architecture."

### 1.3 What the behavioral signals actually mean

The 23 `redrob_signals` fields are not profile quality metrics. They are **availability and reachability metrics**. A candidate with a perfect skills profile who last logged in 8 months ago and has a 5% recruiter response rate is, for practical hiring purposes, not a real option. Behavioral signals should be used as multiplicative modifiers on top of technical fit scores, not as primary ranking signals. The ground truth is scored on fit, not hireability — so behavioral signals should never override strong technical evidence.

**The ghost exception:** A candidate who is inactive for over 365 days AND has a response rate below 5% AND is not open to work AND has submitted zero applications in 30 days is a dead profile. They will never be hired regardless of fit. These can be safely pre-filtered.

---

## 2. System Architecture Overview

The system executes in two strictly separated environments:

**Offline Precompute (`preprocess.py`, no time limit, run once before submission):**
- Phase 0: Parse and embed the JD, build BM25 keyword list, save query vectors
- Phase 1: Embed all 100K profiles, build FAISS index, flag honeypots and disqualifiers, pre-filter ghost profiles
- Phase 1b: Run cross-encoder on all retrieved candidates offline; save scores to `cross_encoder_scores.parquet`
- Phase 1c: Extract all candidate features offline; save to `candidate_features.parquet`
- Phase 1d: Compute retrieval RRF scores offline; save to `retrieval_scores.parquet`

**Runtime Ranking Engine (`rank.py`, must complete in under 5 minutes on CPU, no network):**
- Phase 2: Load precomputed retrieval scores → filter to top 3,000 candidates (configurable via weights.yaml) (< 5s)
- Phase 3: Load precomputed candidate features from parquet (< 5s)
- Phase 4: Load precomputed cross-encoder scores + compute weighted final score → top 200–300 (< 30s)
- Phase 5: Behavioral re-ranking + penalization → top 100 (< 10s)
- Phase 6: Reason generation → final CSV (< 30s)
- Phase 7: Manual validation (pre-submission, not part of 5-min constraint)

**Demo Sandbox (`app.py`, HuggingFace Spaces, no precomputed artifacts):**
- Heuristic-only BM25 path; no FAISS or cross-encoder at inference time
- Accepts a small candidate JSON payload; returns ranked results in under 10s
- Used for interactive demonstration only — not the competition submission path

### 2.1 Dataset Assumption

This system assumes the official competition dataset (`candidates.jsonl` as distributed in the hackathon bundle). Runtime embedding generation is **intentionally unsupported** — the competition explicitly permits offline precomputation, and adding fallback runtime ML inference would force heavy dependencies into `rank.py`.

**Small-file fallback (sandbox path):** If `artifacts/retrieval_scores.parquet` is not found **and** the input file contains ≤ 100 candidates, `rank.py` runs Phase 3 regex feature extraction and Phase 4 scoring entirely in-memory — no FAISS, no BM25 index, no cross-encoder. Because only 100 candidates are processed, this takes milliseconds on CPU. This means the Gradio `app.py` sandbox invokes `rank.py` directly (not a separate demo codebase), preserving a single source of truth.

If artifacts are missing **and** the file contains > 100 candidates, `rank.py` logs an error to stderr and exits gracefully.

**Weight Overfitting Protection:**
Validation must be performed on the 150-200 manually-labeled candidates without aggressively tuning weights to prevent validation set overfitting. Keep weights conservative and generalized.

---

## 3. Repository Layout

```
├── rank.py                        # Runtime entry point
├── preprocess.py                  # Offline pipeline runner
├── weights.yaml                   # Dynamic weights, multipliers, boosts, and thresholds config file
├── requirements.txt
├── submission_metadata.yaml
├── README.md
├── app.py                         # HuggingFace Spaces Gradio sandbox
├── src/
│   ├── __init__.py
│   ├── retriever.py               # Phase 2: FAISS + BM25 + RRF
│   ├── features.py                # Phase 3: Bucket A/B/C extraction
│   ├── scorer.py                  # Phase 4: Weighted scoring formula
│   ├── reranker.py                # Phase 4: Cross-encoder reranking
│   ├── behavioral.py              # Phase 5: Behavioral multipliers, seniority, and penalties
│   └── explainer.py              # Phase 6: Reason generation
├── metadata/
│   └── validation_set.json        # Hand-labeled validation set (150-200 candidates)
└── artifacts/
    ├── jd_skills_vector.npy        # Phase 0: Skills-focused JD embedding
    ├── jd_ideal_vector.npy         # Phase 0: Synthetic ideal candidate embedding
    ├── jd_keywords.json            # Phase 0: BM25 keyword list
    ├── jd_config.json              # Phase 0: Structured JD rule set
    ├── faiss_index.bin             # Phase 1: FAISS IndexFlatIP
    ├── candidate_ids.json          # Phase 1: Ordered candidate ID list (matches FAISS index)
    ├── candidate_texts.pkl         # Phase 1: Serialized candidate texts for BM25
    ├── bm25_index.pkl              # Phase 1: Serialized BM25 index
    ├── candidate_flags.parquet     # Phase 1: Lightweight flags: is_honeypot, is_ghost, disqualifier tags
    ├── retrieval_scores.parquet    # Phase 1d: Precomputed RRF scores for all retrieved candidates
    ├── candidate_features.parquet  # Phase 1c: Precomputed Bucket A/B/C features for all retrieved candidates
    └── cross_encoder_scores.parquet # Phase 1b: Precomputed bge-reranker-v2-m3 scores for top 500 candidates
```

**What each key file does:**

`rank.py` — Competition entry point. Takes `--candidates` and `--out` as CLI arguments. Loads all precomputed artifacts from `artifacts/`, runs Phases 2–6 as pure in-memory operations, writes the output CSV. Must complete in under 5 minutes. **Does not import `flagembedding`, `sentence-transformers`, `faiss`, or `torch` at import time.**

`preprocess.py` — Offline pipeline runner. JD intelligence, corpus embedding, FAISS index building, honeypot flagging, ghost pre-filtering, cross-encoder inference, feature extraction, retrieval scoring. No time constraint. May take 1–4 hours on CPU.

`weights.yaml` — Dynamic configuration file containing all scoring weights, multipliers, boosts, and thresholds. Loaded at startup by both pipelines to avoid hardcoded parameter logic.

`app.py` — HuggingFace Spaces demo sandbox. Heuristic BM25-only path; no transformer model loads. Accepts small JSON payloads for interactive demonstration.

`src/retriever.py` — At preprocess time: FAISS dense search + BM25 sparse search + RRF fusion → saves `retrieval_scores.parquet`. At rank time: loads parquet, filters to top N.

`src/features.py` — At preprocess time: regex-based Bucket A/B/C extraction on all retrieved candidates → saves `candidate_features.parquet`. At rank time: loads parquet.

`src/scorer.py` — Weighted formula: must-have (55%), nice-to-have (10%), career quality (15%), product builder (20%). Pure pandas/numpy; no model calls.

`src/reranker.py` — At preprocess time only: cross-encoder scoring using `bge-reranker-v2-m3` on top 500 → saves `cross_encoder_scores.parquet`. At rank time: loads parquet, merges scores.

`src/behavioral.py` — Multiplicative modifiers: availability, notice, location, social proof, seniority, writing signal, soft penalties, and 90-day bonus.

`src/explainer.py` — Reason generation with 90-day plan framing and evidence injection.

---

## 4. Phase 0 — JD Intelligence

### 4.1 What this phase does

Runs once offline. Parses the job description into a structured config, creates two semantic embedding variants for ensemble retrieval, and builds the BM25 keyword list.

### 4.2 Structured JD config

Parse the JD into typed buckets and save as `artifacts/jd_config.json`:

```python
JD_CONFIG = {
    "must_have": [
        "production embeddings retrieval",
        "vector database hybrid search",
        "ranking system evaluation NDCG MRR MAP",
        "Python production code quality"
    ],
    "nice_to_have": [
        "LLM fine-tuning LoRA QLoRA",
        "learning to rank XGBoost LambdaMART",
        "HR-tech recruiting marketplace",
        "distributed inference optimization"
    ],
    "hard_disqualifiers": [
        "pure research no production",
        "consulting only no product company",
        "computer vision speech robotics no NLP",
        "LangChain only no pre-LLM experience",
        "no code written 18 months architecture only"
    ],
    "soft_negatives": [
        "title chaser switching every 1.5 years",
        "framework demo LangChain tutorial",
        "closed source only no external validation"
    ],
    "location_prefs": ["pune", "noida", "hyderabad", "mumbai", "delhi", "ncr"],
    "behavior_prefs": {
        "notice_period_days_max": 30,
        "open_to_work": True,
        "min_recruiter_response_rate": 0.50
    }
}
```

### 4.3 JD Embeddings — Two Variants

Use `BAAI/bge-m3` (dense + sparse, 570MB). Load once and reuse.

**Variant 1: Skills-focused** (dense on technical vocabulary):

```python
JD_SKILLS_TEXT = """
Senior AI Engineer production embeddings retrieval systems sentence-transformers BGE E5
vector databases Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS
hybrid search semantic search dense retrieval Python evaluation frameworks NDCG MRR MAP
A/B testing ranking systems recommendation systems learning to rank XGBoost neural reranking
product company applied ML deployment real users inference optimization latency
"""
```

**Variant 2: Synthetic Ideal Candidate** (directly from JD "How to read between the lines"):

```python
JD_IDEAL_TEXT = """
6 to 8 years total experience of which 4 to 5 are in applied ML and AI roles at product companies
not pure services. Has shipped at least one end-to-end ranking search or recommendation system
to real users at meaningful scale. Has strong opinions about retrieval hybrid versus dense
evaluation offline versus online and LLM integration when to fine-tune versus prompt and can
defend them with reference to systems they actually built. Located in or willing to relocate to
Noida or Pune. Active on platform recently logged in responds to recruiters.
"""
```

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

jd_skills_output = model.encode(JD_SKILLS_TEXT, return_dense=True, return_sparse=True)
jd_ideal_output = model.encode(JD_IDEAL_TEXT, return_dense=True, return_sparse=True)

import numpy as np
np.save("artifacts/jd_skills_vector.npy", jd_skills_output["dense_vecs"])
np.save("artifacts/jd_ideal_vector.npy", jd_ideal_output["dense_vecs"])
```

The two embeddings represent two perspectives on the ideal candidate: technical vocabulary match and narrative/career-pattern match. Using both in retrieval catches candidates that one vector alone would miss.

### 4.4 BM25 Keyword List

```python
JD_KEYWORDS = [
    "embeddings", "FAISS", "Pinecone", "Weaviate", "Qdrant", "Milvus",
    "sentence-transformers", "BGE", "E5", "NDCG", "MRR", "MAP",
    "A/B test", "A/B testing", "vector search", "hybrid retrieval",
    "hybrid search", "BM25", "ranking", "reranking", "learning to rank",
    "LTR", "dense retrieval", "sparse retrieval", "retrieval system",
    "recommendation system", "search ranking", "relevance", "inverted index",
    "approximate nearest neighbor", "ANN", "evaluation framework",
    "offline evaluation", "online evaluation", "production ML", "inference",
    "semantic search", "bi-encoder", "cross-encoder", "reranker"
]
```

Save as `artifacts/jd_keywords.json`.

---

## 5. Phase 1 — Corpus Preprocessing + Honeypot Detection

### 5.1 What this phase does

Streams all 100K candidates, serializes profile text, builds the FAISS vector index, flags honeypots and disqualifiers, and identifies ghost profiles. All outputs saved to `artifacts/`. No time constraint.

### 5.2 Profile text serialization

For each candidate, build a single text string for embedding:

```python
def build_profile_text(candidate):
    profile = candidate["profile"]
    parts = [
        profile.get("current_title", ""),
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_industry", ""),
    ]
    for role in candidate.get("career_history", []):
        parts.append(role.get("title", ""))
        parts.append(role.get("company", ""))
        parts.append(role.get("description", ""))
    skill_names = " ".join(s["name"] for s in candidate.get("skills", []))
    parts.append(skill_names)
    return " ".join(p for p in parts if p).strip()
```

### 5.3 Corpus embedding and FAISS index

```python
import faiss
import numpy as np
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)

# Stream candidates, encode in batches, build FAISS index
all_texts = []
all_ids = []

for candidate in stream_candidates("candidates.jsonl"):
    all_texts.append(build_profile_text(candidate))
    all_ids.append(candidate["candidate_id"])

# Encode in batches of 512
embeddings = model.encode(all_texts, batch_size=512, return_dense=True)["dense_vecs"]
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # L2-normalize

# Build FAISS IndexFlatIP (inner product = cosine on L2-normalized vectors)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings.astype(np.float32))

faiss.write_index(index, "artifacts/faiss_index.bin")
import json
json.dump(all_ids, open("artifacts/candidate_ids.json", "w"))
```

### 5.4 BM25 index

```python
import pickle
from rank_bm25 import BM25Okapi

tokenized_corpus = [text.lower().split() for text in all_texts]
bm25 = BM25Okapi(tokenized_corpus)

with open("artifacts/bm25_index.pkl", "wb") as f:
    pickle.dump(bm25, f)
with open("artifacts/candidate_texts.pkl", "wb") as f:
    pickle.dump(all_texts, f)
```

### 5.5 Honeypot detection

```python
def is_honeypot(candidate) -> bool:
    """
    Returns True if the candidate has ANY structural impossibility.
    These are forced to relevance tier 0 in the ground truth.
    A True return means final_score = 0.0 — excluded from top 100.
    """
    yoe = candidate.get("profile", {}).get("years_of_experience")
    if yoe is None:
        # If missing, derive from career history so we don't accidentally honeypot everyone
        total_months = sum(r.get("duration_months", 0) for r in candidate.get("career_history", []))
        yoe = total_months / 12.0
    yoe_months = yoe * 12
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    assessment_scores = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})

    # Check 1: Skill duration > total career length (+6 month grace)
    for skill in skills:
        duration = skill.get("duration_months", 0)
        if duration > yoe_months + 6:
            return True

    # Check 2: Expert/advanced skill with ZERO duration
    for skill in skills:
        if skill.get("proficiency") in ("expert", "advanced"):
            duration = skill.get("duration_months", None)
            if duration is not None and duration == 0:
                return True

    # Check 3: Expert/advanced skill with low assessment score
    for skill in skills:
        if skill.get("proficiency") in ("expert", "advanced"):
            score = assessment_scores.get(skill["name"])
            if score is not None and score < 40:
                return True

    # Check 4: Per-role tenure > total YoE
    for role in career:
        role_duration = role.get("duration_months", 0)
        if role_duration > yoe_months + 12:
            return True

    # Check 5: The "LLM Repetition" Trap
    # 10+ skills and every single one has the exact same duration_months
    if len(skills) >= 10:
        durations = {s.get("duration_months") for s in skills if "duration_months" in s}
        if len(durations) == 1 and list(durations)[0] is not None:
            return True

    # Check 6: The "Empty Shell" Trap
    # Claiming > 5 years of experience but having zero job history
    if yoe > 5 and len(career) == 0:
        return True

    # Check 7: The "Impossible Overlap" Trap
    # Sum of all job durations is > 1.5x their stated YoE (hallucinated timelines)
    if yoe_months > 0:
        total_career_months = sum(r.get("duration_months", 0) for r in career)
        # Give a grace period for small overlaps, but flag massive >1.5x concurrency
        if total_career_months > (yoe_months * 1.5) + 12:
            return True

    # Check 8: Company Founding Time Travel
    # JD warning: "8 years of experience at a company founded 3 years ago"
    KNOWN_COMPANIES = {
        "anthropic": 2021, "openai": 2015, "hugging face": 2016, "huggingface": 2016,
        "pinecone": 2019, "qdrant": 2021, "weaviate": 2019, "langchain": 2022,
        "llamaindex": 2022, "redrob": 2023, "cohere": 2019, "mistral": 2023
    }
    for role in career:
        company = role.get("company", "").lower()
        duration_years = role.get("duration_months", 0) / 12.0
        for comp_name, founded_year in KNOWN_COMPANIES.items():
            if comp_name in company:
                # Assuming current year is 2026 per hackathon timeline
                max_possible_years = 2026 - founded_year + 1 
                if duration_years > max_possible_years:
                    return True

    return False
```

### 5.6 Ghost profile pre-filter

A ghost profile is one that is effectively unreachable regardless of fit. Pre-filtering removes them from retrieval entirely, improving precision without meaningful recall loss. Approximately 1–3% of the corpus will be flagged.

**Ghost criteria — ALL four conditions must be true:**
- `last_active_date` > 365 days before the reference date
- `recruiter_response_rate` < 0.05
- `open_to_work_flag` = False
- `applications_submitted_30d` = 0

```python
from datetime import date

def is_ghost(candidate, reference_date) -> bool:
    # Note: ghost pre-filtering is irreversible. The reference_date used here is locked
    # at preprocess time. If the evaluation payload contains newer dates, pre-filtered
    # candidates cannot be recovered.
    signals = candidate.get("redrob_signals", {})
    last_active_str = signals.get("last_active_date")
    if last_active_str is None:
        # Default to False (active). Missing dates could represent newly-imported profiles.
        return False
    days_inactive = (reference_date - date.fromisoformat(last_active_str)).days
    return (
        days_inactive > 365
        and signals.get("recruiter_response_rate", 1.0) < 0.05
        and not signals.get("open_to_work_flag", True)
        and signals.get("applications_submitted_30d", 1) == 0
    )
```

### 5.7 Disqualifier tagging

```python
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
    "ltimindtree", "l&t infotech", "niit technologies", "zensar", "mastech",
    "syntel", "kpit", "cyient", "birlasoft", "persistent systems"
}

def tag_disqualifiers(candidate) -> dict:
    career = candidate.get("career_history", [])
    titles_lower = [r.get("title", "").lower() for r in career]
    desc_text = " ".join(r.get("description", "").lower() for r in career)
    skills_lower = [s["name"].lower() for s in candidate.get("skills", [])]

    # consulting_only: entire career at consulting firms, zero product company tenure
    total_months = sum(r.get("duration_months", 0) for r in career)
    consulting_months = sum(
        r.get("duration_months", 0) for r in career
        if any(firm in r.get("company", "").lower() for firm in CONSULTING_FIRMS)
        or r.get("industry", "").lower() in ("it services", "consulting", "outsourcing")
    )
    product_ratio = 1.0 - (consulting_months / total_months) if total_months > 0 else 0.0
    consulting_only = product_ratio == 0.0

    # research_only: only academic/research titles, no engineer/developer roles
    engineering_titles = {"engineer", "developer", "data scientist", "applied scientist", "architect", "lead", "head"}
    research_titles = {"researcher", "research scientist", "phd", "postdoc", "intern"}
    has_engineering = any(t in " ".join(titles_lower) for t in engineering_titles)
    has_only_research = not has_engineering and any(t in " ".join(titles_lower) for t in research_titles)
    research_only = has_only_research

    # wrong_domain: CV/speech/robotics without NLP/IR
    cv_speech_terms = {"computer vision", "opencv", "yolo", "object detection", "speech recognition", "tts", "asr", "robotics"}
    nlp_ir_terms = {"nlp", "retrieval", "ranking", "recommendation", "search", "embedding", "information retrieval"}
    has_cv_speech = any(t in desc_text or any(t in s for s in skills_lower) for t in cv_speech_terms)
    has_nlp_ir = any(t in desc_text or any(t in s for s in skills_lower) for t in nlp_ir_terms)
    wrong_domain = has_cv_speech and not has_nlp_ir

    return {
        "product_ratio": round(product_ratio, 4),
        "consulting_only": consulting_only,
        "research_only": research_only,
        "wrong_domain": wrong_domain
    }
```

### 5.8 Flags Parquet

Save `artifacts/candidate_flags.parquet` with one row per candidate:

| Column | Type | Notes |
|---|---|---|
| candidate_id | string | Primary key |
| is_honeypot | bool | True = score forced to 0.0 |
| is_ghost | bool | True = pre-filtered, score forced to 0.0 |
| product_ratio | float | 0.0–1.0 time-weighted product company ratio |
| consulting_only | bool | Entire career at consulting firms |
| research_only | bool | Only academic/research roles |
| wrong_domain | bool | CV/speech/robotics, no NLP/IR |

---

## 6. Phase 2 — Multi-Signal Retrieval

### 6.1 What this phase does

**At preprocess time (`preprocess.py`):** Runs FAISS dense search, BM25 sparse search, and RRF fusion against all 100K candidates. Saves ranked candidate IDs and RRF scores to `artifacts/retrieval_scores.parquet`. This is a one-time operation.

**At rank time (`rank.py`):** Loads `retrieval_scores.parquet` (pure pandas read), filters to top N by RRF score, optionally applies soft activity boost from `candidate_flags.parquet`. **No FAISS or BM25 calls at rank time.**

**Retrieval pool size tuning:** Precompute saves the top 5,000 candidates to `retrieval_scores.parquet`. At runtime, we filter this down to the top N = 3,000 (controlled by `pool_size` in `weights.yaml`). Experiment with 2,000–3,000; the RRF-ranked pool drops off steeply in quality after the top 2,000, and a smaller pool reduces Phase 4 scoring time. Only increase beyond 3,000 if validation shows recall loss (Tier 3+ candidates missing from the Phase 4 input — check against `metadata/validation_set.json`).

Target rank-time cost: under 5 seconds on CPU.

### 6.2 Dense retrieval via FAISS

```python
import faiss
import numpy as np
import json

index = faiss.read_index("artifacts/faiss_index.bin")
all_ids = json.load(open("artifacts/candidate_ids.json"))

jd_skills_vec = np.load("artifacts/jd_skills_vector.npy").astype(np.float32).reshape(1, -1)
jd_ideal_vec = np.load("artifacts/jd_ideal_vector.npy").astype(np.float32).reshape(1, -1)

# Search both JD vectors
scores_skills, idx_skills = index.search(jd_skills_vec, k=2000)
scores_ideal, idx_ideal = index.search(jd_ideal_vec, k=2000)

dense_ids_skills = [all_ids[i] for i in idx_skills[0]]
dense_ids_ideal = [all_ids[i] for i in idx_ideal[0]]
```

### 6.3 Sparse retrieval via BM25

```python
import pickle
from rank_bm25 import BM25Okapi

with open("artifacts/bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)

keywords = json.load(open("artifacts/jd_keywords.json"))
query_tokens = " ".join(keywords).lower().split()

scores_bm25 = bm25.get_scores(query_tokens)
top_bm25_idx = np.argsort(scores_bm25)[::-1][:2000]
sparse_ids = [all_ids[i] for i in top_bm25_idx]
```

### 6.4 Soft activity boost

Before RRF, lightly boost candidates who are actively seeking. This prevents active candidates from being buried behind stale ones when ranks are equal.

```python
# Load lightweight signals from flags parquet
# For candidates in the top pools: if open_to_work OR last_active < 90d,
# artificially boost their retrieval rank by 200 positions (within the retrieval step only).
# This is a soft nudge, not a hard filter.
```

### 6.5 Reciprocal Rank Fusion

```python
def rrf(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores = {}
    for ranked in ranked_lists:
        for rank_idx, cand_id in enumerate(ranked):
            scores[cand_id] = scores.get(cand_id, 0.0) + 1.0 / (k + rank_idx + 1)
    return scores

rrf_scores = rrf([dense_ids_skills, dense_ids_ideal, sparse_ids])

# Sort by RRF score, take top 5000
retrieved = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:5000]
retrieved_ids = [r[0] for r in retrieved]
```

The union of three retrievers typically produces 3,000–5,000 unique candidates. These go to Phase 3 for feature extraction.

**Recall verification (do this once during development):** After building the retrieval pool, manually confirm that 10+ Tier 3 candidates from your validation set are present. A missing Tier 3 candidate at this stage is unrecoverable. Fix retriever thresholds before proceeding.

---

## 7. Phase 3 — Candidate Feature Extraction

### 7.1 What this phase does

Runs **entirely offline** in `preprocess.py`. No time constraint. Processes the 3,000–5,000 candidates retrieved by Phase 1d, computes three evidence buckets (A/B/C) plus the consistency score, and serializes all results — including verbatim career description snippets — to `artifacts/candidate_features.parquet`. At rank time, `rank.py` simply reads this parquet; no regex or feature computation happens at rank time.

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

### 7.5 Consistency Score

Detects candidates who claim high-proficiency skills that are not supported by their actual career trajectory. This is distinct from honeypot detection (which catches impossible timelines). Consistency Score catches the **skill-career identity mismatch** — the most common form of keyword stuffing the JD warns about.

**Examples of inconsistency this catches:**
- Claims `NLP: Expert` → career is entirely BI Analyst / Data Analyst roles
- Claims `LLM fine-tuning: Expert` → skill duration is 3 months
- Claims `Retrieval systems: Advanced` → no retrieval or search terms in any career description

```python
# Domains mapped to career title signals that would NOT support the claim
DOMAIN_TITLE_CONTRADICTION = {
    "nlp": ["bi analyst", "business analyst", "data analyst", "excel", "tableau",
            "reporting", "bi developer", "sql analyst"],
    "retrieval": ["bi analyst", "business analyst", "data analyst", "frontend",
                  "mobile developer", "android", "ios"],
    "llm": ["bi analyst", "business analyst", "data analyst", "manual qa",
            "support engineer", "it support"],
    "ranking": ["bi analyst", "data analyst", "frontend developer",
                "mobile developer", "devops"],
}

# JD-target skill keywords to check for claim vs evidence mismatch
CONSISTENCY_TARGET_SKILLS = [
    "nlp", "retrieval", "ranking", "recommendation", "embeddings",
    "llm", "vector", "search", "faiss", "transformer"
]

def compute_consistency_score(candidate, career_text) -> float:
    """
    Returns a score in [0, 1] where 1 = fully consistent, 0 = maximally inconsistent.
    Detects: expert/advanced skill claims unsupported by career title or skill duration.
    """
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    all_titles = " ".join(r.get("title", "").lower() for r in career)

    total_checks = 0
    contradiction_count = 0

    for skill in skills:
        name = skill["name"].lower()
        proficiency = skill.get("proficiency", "beginner")
        duration = skill.get("duration_months", 0)

        # Only check claimed advanced/expert skills — beginner/intermediate are low risk
        if proficiency not in ("advanced", "expert"):
            continue

        is_target = any(kw in name for kw in CONSISTENCY_TARGET_SKILLS)
        if not is_target:
            continue

        total_checks += 1
        contradictions = 0

        # Check 1: Career title contradiction — claimed domain not reflected in any role title
        # Checked against current title only to avoid penalizing legitimate career progression
        current_title = candidate.get("profile", {}).get("current_title", "").lower()
        for domain, bad_titles in DOMAIN_TITLE_CONTRADICTION.items():
            if domain in name:
                if any(bt in current_title for bt in bad_titles):
                    contradictions += 1
                break

        # Check 2: Duration contradiction — expert claim with < 12 months of usage
        if duration > 0 and duration < 12:
            contradictions += 1

        # Check 3: Career text contradiction — skill not mentioned anywhere in career descriptions
        if name not in career_text.lower():
            contradictions += 0.5  # Partial penalty — may use synonym

        if contradictions >= 1:
            contradiction_count += 1

    if total_checks == 0:
        return 1.0  # No advanced/expert target-skill claims to check — neutral

    contradiction_ratio = contradiction_count / total_checks
    # Score: 1.0 = fully consistent, 0.5 = half claims are contradicted, 0.0 = all contradicted
    return round(1.0 - (contradiction_ratio * 0.7), 4)  # Scale: max penalty is 0.70 drop
```

Store `consistency_score` alongside Bucket A/B/C outputs. Use it as a multiplier in Phase 5.

### 7.5 Bucket C — JD Fit Gaps

```python
def score_fit_gaps(candidate, career_text, flags):
    # Title velocity: avg tenure < 18 months across 3+ roles
    career = candidate.get("career_history", [])
    if len(career) >= 3:
        avg_tenure = sum(r.get("duration_months", 0) for r in career) / len(career)
        title_velocity_flag = avg_tenure < 18
    else:
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
    yoe = candidate["profile"].get("years_of_experience", 0)
    current_title = candidate["profile"].get("current_title", "").lower()
    STOPPED_CODING_TITLES = {"architect", "vp", "vice president", "director", "cto", "head of"}
    code_stopped = yoe > 8 and any(t in current_title for t in STOPPED_CODING_TITLES)

    # Seniority score: JD says 5-9 years is a range, not a hard requirement.
    # "We'll seriously consider candidates outside the band if other signals are strong."
    # Model as a soft multiplier — sweet spot = 1.0, outside band tapers gradually.
    if 5 <= yoe <= 9:
        seniority_score = 1.00   # Sweet spot
    elif 4 <= yoe < 5:
        seniority_score = 0.85   # Slightly junior; acceptable if signals strong
    elif 10 <= yoe <= 12:
        seniority_score = 0.90   # Mild over-seniority; founding team dynamic
    elif 3 <= yoe < 4:
        seniority_score = 0.65   # Significant gap; needs exceptional other signals
    elif yoe > 12:
        seniority_score = 0.80   # Likely over-senior for fast-moving founding role
    else:
        seniority_score = 0.40   # <3 years: very unlikely fit

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
        "saved_by_recruiters_30d": signals.get("saved_by_recruiters_30d", 0),
        "endorsements_received": signals.get("endorsements_received", 0),
        "applications_submitted_30d": signals.get("applications_submitted_30d", 0),
        "profile_completeness_score": signals.get("profile_completeness_score", 50.0),
        "verified_email": signals.get("verified_email", False),
        "verified_phone": signals.get("verified_phone", False),
        "linkedin_connected": signals.get("linkedin_connected", False),
        "willing_to_relocate": signals.get("willing_to_relocate", False),
        "preferred_work_mode": signals.get("preferred_work_mode", "flexible"),
        "location": candidate["profile"].get("location", "").lower(),
        "country": candidate["profile"].get("country", "").lower(),
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
| `consistency_score` | float | Skill claim contradiction penalty |
| `snippets_json` | string | JSON dict of the best 60-char evidence snippets |

---

## 8. Phase 4 — Core Scoring + Cross-Encoder Rerank

### 8.1 Weighted formula

```python
def compute_core_score(bucket_a, bucket_b, bucket_c, behavioral, flags) -> float:
    # --- Must-Have Score (55% of total) ---
    # Source: Bucket A skill evidence scores
    retrieval_ev = bucket_a.get("retrieval_search", 0.0) / 3.0
    vectordb_ev = bucket_a.get("vector_db_hybrid", 0.0) / 3.0
    eval_ev = bucket_a.get("eval_framework", 0.0) / 3.0
    python_ev = bucket_a.get("python_coding", 0.0) / 3.0

    must_have_raw = (
        0.25 * retrieval_ev +
        0.20 * vectordb_ev +
        0.10 * eval_ev +
        0.05 * python_ev
    )

    # Get system experience score from Bucket B
    sys_experience_score = bucket_b.get("sys_experience_score", 0.0)

    # Softened Hard cap: if candidate has zero retrieval/search evidence AND zero vector DB evidence AND zero
    # broad system/recommendation evidence, cap must-have score at 0.5.
    # Recsys/matching engine experience implies semantic search capability per JD.
    has_any_retrieval_or_recsys = (
        bucket_a["retrieval_search"] > 0 or
        bucket_a["vector_db_hybrid"] > 0 or
        sys_experience_score > 0.0
    )
    if not has_any_retrieval_or_recsys:
        must_have_raw = min(must_have_raw, 0.5)

    must_have_score = must_have_raw / 0.60  # Normalize to [0,1]
    must_have_score = min(must_have_score, 1.0)  # Cap at 1.0 in case of assessment bonuses

    # --- Nice-to-Have Score (10%) ---
    # Weight reduced from 20% → 10%; headroom reallocated to Product Builder Score.
    ltr_ev = bucket_a.get("ltr_reranking", 0.0) / 3.0
    llm_ev = bucket_a.get("llm_integration", 0.0) / 3.0
    dist_ev = bucket_a.get("distributed_systems", 0.0) / 3.0
    hr_ev = bucket_a.get("hr_tech_exposure", 0.0) / 3.0

    nice_to_have_score = (
        0.04 * ltr_ev +
        0.03 * llm_ev +
        0.02 * dist_ev +
        0.01 * hr_ev
    ) / 0.10  # Normalize to [0,1]

    # --- Career Quality Score (15%) ---
    # Focused on career-trajectory quality: domain relevance, recency, depth.
    # Product company + deployment + shipper signals moved to Product Builder Score below
    # so they have an explicit, first-class presence in the formula.
    experience_recency = bucket_b["experience_recency"]
    depth_signal = bucket_b["depth_signal"]

    career_quality_raw = (
        0.08 * sys_experience_score +  # Built the right kind of systems
        0.04 * experience_recency +    # Most recent role in a relevant domain
        0.03 * depth_signal            # IR/retrieval evidence sustained across multiple roles
    )

    # Consulting/research/wrong-domain multipliers
    if flags.get("consulting_only"):
        career_quality_raw *= 0.4
    if flags.get("research_only"):
        career_quality_raw *= 0.5
    if flags.get("wrong_domain"):
        career_quality_raw *= 0.3

    career_quality_score = career_quality_raw / 0.15  # Normalize to [0,1]

    # --- Product Builder Score (20%) ---
    # Explicit composite of founding-team and shipping signals.
    # The JD emphasises product-company background, shipping velocity, and ownership
    # at least as much as specific ML tool keywords (§1.2). Computed in Bucket B
    # (score_career_quality) so career-description evidence is available at extraction time.
    # Note: product_builder_score is already normalized to [0,1] within Bucket B.
    product_builder_score = bucket_b.get("product_builder_score", 0.0)

    # --- Combined Weighted Score ---
    core_score = (
        0.55 * must_have_score +
        0.10 * nice_to_have_score +
        0.15 * career_quality_score +
        0.20 * product_builder_score
    )

    return float(core_score)
```

### 8.2 Cross-encoder score merge (rank time — precomputed)

The cross-encoder (`bge-reranker-v2-m3`) runs **entirely offline** in `preprocess.py`. It scores the top **500** candidates (by preliminary core score) against the JD, normalizes outputs, and saves them to `artifacts/cross_encoder_scores.parquet`. At rank time, `rank.py` simply joins this parquet on `candidate_id`.

**Why 500?** NDCG@50 and MAP both matter — a candidate at rank ~150 after handcrafted scoring may deserve top 20 after semantic reranking. 300 was too conservative for those metrics. 500 is still tiny at precompute time (offline, no time limit).

**Why offline?** Cross-encoder inference on 500 candidates takes ~2.5 min on CPU (~300ms/doc × 500 docs). Moving it offline eliminates the largest single runtime cost entirely.

**CE weight validation (do this before submission):** The 0.20 CE weight is an assumption. Before finalising, compare NDCG@10 with CE weight = 0 vs 0.10 vs 0.20 on the validation set. If the gain is <2–3 NDCG points, reduce to 0.10 or remove the merge entirely (set `cross_encoder_score` contribution to 0). The handcrafted evidence scoring already captures much of the semantic signal; the CE is most valuable when it separates candidates with near-identical Bucket A scores.

**Preprocess-time code (in `preprocess.py`):**
```python
from FlagEmbedding import FlagReranker
import pandas as pd

reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)

# top_500_candidates: list of dicts with candidate_id and profile_text
# (top 500 by preliminary core score computed during offline feature extraction)
pairs = [[JD_SKILLS_TEXT.strip(), c["profile_text"]] for c in top_500_candidates]
cross_enc_scores = reranker.compute_score(pairs, normalize=True)

ce_df = pd.DataFrame({
    "candidate_id": [c["candidate_id"] for c in top_500_candidates],
    "cross_encoder_score": cross_enc_scores
})
ce_df.to_parquet("artifacts/cross_encoder_scores.parquet", index=False)
```

**Rank-time code (in `rank.py` / `src/scorer.py`):**
```python
import pandas as pd

ce_df = pd.read_parquet("artifacts/cross_encoder_scores.parquet")
    # Left-join onto scored_candidates; candidates not in top 500 fall back to their core score
    scored_df = scored_df.merge(ce_df, on="candidate_id", how="left")
    scored_df["cross_encoder_score"] = scored_df["cross_encoder_score"].fillna(scored_df["core_score"])

# Merge: 0.8 handcrafted + 0.2 cross-encoder
# Rationale: handcrafted features encode behavioral signals (hireability, notice period,
# consulting history, product ratio) that the cross-encoder cannot see — it only scores
# semantic query-document relevance. 0.2 weight keeps CE as a tiebreaker, not an override.
scored_df["final_phase4_score"] = (
    0.8 * scored_df["core_score"] +
    0.2 * scored_df["cross_encoder_score"]
)
```

---

## 9. Phase 5 — Behavioral Re-ranking + Penalization

### 9.1 Availability multiplier

```python
def availability_multiplier(behavioral) -> float:
    days_inactive = behavioral["days_inactive"]
    response_rate = behavioral["recruiter_response_rate"]
    open_to_work = behavioral["open_to_work"]
    interview_rate = behavioral["interview_completion_rate"]

    if days_inactive <= 30 and response_rate >= 0.70 and (open_to_work or interview_rate >= 0.80):
        return 1.15  # Actively seeking, highly reachable
    elif days_inactive <= 90 and response_rate >= 0.50:
        return 1.05  # Recently active and responsive
    elif days_inactive <= 180 and response_rate >= 0.30:
        return 0.80  # Moderately passive
    elif days_inactive > 180 or response_rate < 0.15:
        return 0.70  # Practically unreachable; softened from 0.50 — competition scores relevance, not hireability
    else:
        return 0.90  # Mildly passive
```

### 9.2 Notice period modifier

```python
def notice_modifier(days) -> float:
    if days <= 30:
        return 1.00  # JD: ideal, can buy out up to 30 days
    elif days <= 60:
        return 0.95
    elif days <= 90:
        return 0.90  # JD: "bar gets higher" at 30+
    elif days <= 120:
        return 0.85
    else:
        return 0.75  # JD: "significant concern"
```

### 9.3 Location modifier

```python
PUNE_NOIDA_CITIES = {"pune", "noida", "greater noida", "delhi", "new delhi",
                      "gurugram", "gurgaon", "faridabad", "ghaziabad"}
JD_WELCOME_CITIES = {"hyderabad", "mumbai"}
INDIA_ADJACENT = {"bangalore", "bengaluru", "chennai", "kolkata",
                   "ahmedabad", "indore", "jaipur", "chandigarh", "kochi"}

def location_modifier(behavioral) -> float:
    location = behavioral["location"]
    country = behavioral["country"]
    willing = behavioral["willing_to_relocate"]

    if any(city in location for city in PUNE_NOIDA_CITIES):
        return 1.0
    if any(city in location for city in JD_WELCOME_CITIES):
        return 1.00 if willing else 0.98
    if any(city in location for city in INDIA_ADJACENT):
        return 0.98 if willing else 0.95
    if country == "india" and willing:
        return 0.95
    if country == "india":
        return 0.92
    if willing:
        return 0.90
    return 0.85
```

### 9.4 Social proof boost

```python
def social_proof_boost(behavioral) -> float:
    """
    Additive boost from Redrob platform signals not already captured by multipliers.
    Uses 9 of the 23 redrob_signals fields. Cap at 0.12 so no single cluster dominates.
    """
    boost = 0.0

    # --- Market validation (other recruiters already found this person valuable) ---
    if behavioral["github_activity_score"] > 60:
        boost += 0.03  # JD: external validation valued; open-source contributions
    if behavioral["saved_by_recruiters_30d"] > 5:
        boost += 0.04  # Human-curated: other recruiters are already shortlisting them
    if behavioral.get("profile_views_received_30d", 0) > 20:
        boost += 0.01  # Passive market interest — searched for and clicked on

    # --- Engagement quality (serious about the job search) ---
    if behavioral["endorsements_received"] > 20:
        boost += 0.01  # Peer credibility signal
    if behavioral.get("interview_completion_rate", 0) > 0.80:
        boost += 0.02  # Shows up and follows through — predictive of offer conversion
    if behavioral.get("offer_acceptance_rate", -1) > 0.70:
        boost += 0.01  # When they receive offers they accept them — not just browsing

    # --- Profile credibility ---
    if behavioral.get("profile_completeness_score", 0) > 80:
        boost += 0.01  # Actively managing profile = genuinely in the market
    if behavioral["linkedin_connected"]:
        boost += 0.01  # Basic platform legitimacy

    # --- Response speed (availability complement) ---
    avg_rt = behavioral.get("avg_response_time_hours", 24.0)
    if avg_rt <= 4.0 and behavioral["recruiter_response_rate"] >= 0.60:
        boost += 0.01  # Fast AND responsive — highest-reachability signal

    return min(boost, 0.12)  # Cap: no single signal cluster should dominate final score
```

### 9.5 Seniority modifier

```python
def seniority_modifier(bucket_c) -> float:
    """
    Applies the seniority soft window from Bucket C.
    JD: "5-9 years is a range, not a requirement. We'll seriously consider candidates
    outside the band if other signals are strong."
    Returns the seniority_score computed in Phase 3 — no additional logic needed.
    """
    return bucket_c.get("seniority_score", 1.0)
```

### 9.6 Soft penalties

All penalties are **soft multipliers**. The JD uses language like "probably not move forward" and "the bar gets higher" — not "automatic reject". Only honeypots and ghosts are zeroed out. Everything else is a downward nudge that strong technical evidence can overcome.

> **Validate harsh penalties before trusting them.** The multipliers below (`consulting_only ×0.4`, `research_only ×0.40`, `langchain_only_flag ×0.45`) are calibrated from JD language but have not yet been tested against the actual labeled set. If Redrob intentionally inserted edge-case candidates — e.g. a strong researcher with genuine product exposure, or a LangChain practitioner with a deep pre-LLM ML background — these will over-penalise real fits. Run Phase 7 validation against `metadata/validation_set.json` first. In particular: confirm that no Tier 3+ labeled candidate is being hard-penalised by a flag that misread their profile.

```python
def soft_penalties(bucket_c, flags, behavioral, consistency_score) -> float:
    multiplier = 1.0

    # Title velocity: switched every ~1.5 years across 3+ jobs
    # JD: "not a fit" — strong signal but softened to 0.80;
    # many startup engineers legitimately switch frequently.
    if bucket_c["title_velocity_flag"]:
        multiplier *= 0.80

    # Code stopped: architect/VP/Director with yoe > 8
    # JD: "probably not move forward" — soft, not hard.
    if bucket_c["code_stopped"]:
        multiplier *= 0.75

    # LangChain-only AI experience under 12 months, no pre-LLM ML background
    # JD: "probably not move forward, unless substantial pre-LLM ML production experience"
    if bucket_c.get("langchain_only_flag"):
        multiplier *= 0.45

    # Remote-only preference for a hybrid role
    pref_mode = behavioral.get("preferred_work_mode", "").lower().strip()
    if pref_mode in ("remote", "wfh", "work from home"):
        multiplier *= 0.85

    # Research-only background — JD: "will not move forward" — strongest language
    if flags.get("research_only"):
        multiplier *= 0.40

    # Wrong domain (CV/speech without NLP/IR)
    if flags.get("wrong_domain"):
        multiplier *= 0.50

    # Consistency score: skill-career mismatch multiplier.
    # Score 1.0 = no penalty. Score 0.30 = heavy mismatch (keyword stuffer).
    multiplier *= consistency_score

    # Closed-source only for 5+ years without external validation (GitHub, papers, talks)
    if bucket_c.get("closed_source_flag"):
        multiplier *= 0.80

    return multiplier
```

### 9.7 Final score assembly

```python
def compute_final_score(candidate_data) -> float:
    phase4_score = candidate_data["final_phase4_score"]
    behavioral = candidate_data["behavioral"]
    bucket_a = candidate_data.get("bucket_a", {})
    bucket_b = candidate_data["bucket_b"]
    bucket_c = candidate_data["bucket_c"]
    flags = candidate_data["flags"]
    consistency_score = candidate_data.get("consistency_score", 1.0)

    # Ghost and honeypot hard exclusions
    if flags.get("is_honeypot") or flags.get("is_ghost"):
        return 0.0

    avail_mult = availability_multiplier(behavioral)
    penalty_mult = soft_penalties(bucket_c, flags, behavioral, consistency_score)

    # Logistical signals: notice period, location, seniority, writing culture.
    # Grouped and floor-capped so no single operational signal collapses the score.
    notice_mult = notice_modifier(behavioral["notice_period_days"])
    loc_mult = location_modifier(behavioral)
    seniority_mult = seniority_modifier(bucket_c)
    writing_mult = bucket_b.get("writing_signal", 1.0)

    logistical_mult = notice_mult * loc_mult * seniority_mult * writing_mult
    logistical_mult = max(logistical_mult, 0.75)  # Floor: logistics cannot reduce score by >25%

    # Combined multiplier floor: prevents the full chain (availability × penalties × logistics)
    # from collapsing a strong technical score into near-zero. Strong technical fit must
    # always be able to show through — no variable cluster should dominate alone.
    combined_mult = avail_mult * penalty_mult * logistical_mult
    combined_mult = max(combined_mult, 0.25)  # Floor: maximum total reduction is 75%

    # Additive bonuses — reward without gating.
    # 90-day alignment: JD describes 3 milestones (audit retrieval, ship v2 ranker, build eval
    # framework). Moved from multiplicative to additive: being able to execute all 3 on day 1
    # is a bonus signal, not a disqualifier. Missing milestone evidence ≠ wrong hire.
    product_ratio = bucket_b.get("product_ratio", 0.5)
    ninety_day_alignment = compute_ninety_day_alignment(bucket_a, product_ratio)
    ninety_day_bonus = 0.08 * ninety_day_alignment  # Range: 0.0 to +0.08

    # Platform signals: market validation, engagement quality, profile credibility.
    # Uses 9 of the 23 redrob_signals fields not already captured in multipliers above.
    social_boost = social_proof_boost(behavioral)  # Range: 0.0 to +0.12 (capped)

    # Final formula:
    # Multiplicative chain: technical fit × strong behavioral gates × logistical group (capped)
    # Additive bonuses: 90-day milestone readiness + Redrob platform signal cluster
    # No single variable can move the score by more than ~75% on its own — mix of signals.
    final = (
        phase4_score
        * combined_mult
        + ninety_day_bonus
        + social_boost
    )

    # Floor protection: if penalties drop the score near zero, ensure it hits 0.0 to drop out of ranking
    if penalty_mult < 0.20:
        return 0.0

    # Score range bounding (max 2.0)
    final = min(final, 2.0)

    return round(float(final), 6)


# --- Rank assignment with deterministic tie-breaking ---
# Spec §3: "If two candidates have the same score, you must still assign unique ranks.
# Break score ties deterministically using a secondary signal from your model,
# or by candidate_id ascending."
#
# Sort key: (-final_score, candidate_id)
# candidate_id is a string (CAND_XXXXXXX); ascending lexicographic order is deterministic.
def assign_ranks(scored_candidates: list[dict]) -> list[dict]:
    sorted_cands = sorted(
        scored_candidates,
        key=lambda c: (-c["final_score"], c["candidate_id"])
    )
    for rank, c in enumerate(sorted_cands, start=1):
        c["rank"] = rank
        
    # Validation constraint check (non-increasing score)
    for i in range(1, len(sorted_cands)):
        assert sorted_cands[i]["final_score"] <= sorted_cands[i-1]["final_score"], "Score sorting failed"
        
    return sorted_cands
```

---

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

```python
def generate_reasoning(row: dict) -> str:
    """
    row: candidate dict with all Phase 3 features and Phase 5 final score.
    Returns: 1-2 sentence reasoning string.
    """
    yoe = row.get("yoe", 0)
    title = row.get("current_title", "engineer")
    rank = row.get("rank", 100)
    product_ratio = row.get("product_ratio", 0.5)
    bucket_a_snippets = row.get("bucket_a_snippets", {})
    behavioral = row.get("behavioral", {})

    # Pick strongest evidence snippet
    priority_order = ["retrieval_search", "vector_db_hybrid", "eval_framework",
                      "ltr_reranking", "llm_integration"]
    best_snippet = ""
    best_bucket = ""
    for bucket in priority_order:
        snippet = bucket_a_snippets.get(bucket, "")
        if snippet:
            best_snippet = snippet
            best_bucket = bucket
            break

    # Primary sentence: evidence + 90-day milestone
    if best_snippet:
        if best_bucket in ("retrieval_search", "vector_db_hybrid"):
            milestone = "Weeks 1-3 retrieval audit mandate"
        elif best_bucket in ("ltr_reranking", "llm_integration"):
            milestone = "Weeks 4-8 hybrid ranker mandate"
        elif best_bucket == "eval_framework":
            milestone = "Weeks 9-12 evaluation framework mandate"
        else:
            milestone = "Weeks 1-3 retrieval audit mandate"

        primary = (
            f"{int(yoe)}-year {title}; "
            f"evidence: '{best_snippet[:80]}'; "
            f"suited for {milestone}."
        )
    else:
        primary = (
            f"{int(yoe)}-year {title}; "
            f"no explicit retrieval or ranking deployment evidence found in career descriptions."
        )

    # Secondary sentence: context or concern based on rank tier
    secondary = ""
    notice = behavioral.get("notice_period_days", 60)
    response_rate = behavioral.get("recruiter_response_rate", 0.5)

    if rank <= 30:
        # Strong context: product company background or availability
        if product_ratio >= 0.8:
            secondary = "Predominantly product-company background aligns with JD requirements."
        elif notice <= 30 and response_rate >= 0.70:
            secondary = f"Strong availability: {notice}-day notice, {int(response_rate*100)}% recruiter response rate."
    elif rank <= 70:
        # Neutral: note any concern
        if product_ratio <= 0.2:
            secondary = "Predominantly consulting/services background — noted gap per JD."
        elif notice > 90:
            secondary = f"Risk: {notice}-day notice period exceeds preferred threshold."
    else:
        # Mandatory concern for ranks 71-100
        concerns = []
        if not best_snippet:
            concerns.append("limited verifiable retrieval/ranking evidence")
        if product_ratio <= 0.2:
            concerns.append("consulting-heavy career")
        if notice > 90:
            concerns.append(f"{notice}-day notice period")
        if behavioral.get("days_inactive", 0) > 180:
            concerns.append("platform inactive >180 days")
        if concerns:
            secondary = f"Risks: {'; '.join(concerns)}."
        else:
            # Generate concern only from actual signals, not rank position
            if behavioral.get("days_inactive", 0) > 90:
                secondary = f"Platform inactive {behavioral['days_inactive']} days — reachability uncertain."
            elif notice > 60:
                secondary = f"{notice}-day notice period is above preferred threshold."
            else:
                secondary = "Rank impacted primarily by behavioral or availability signals."

    parts = [p for p in [primary, secondary] if p]
    return " ".join(parts[:2])
```

### 10.4 Rank-dependent tone

| Rank Range | Template Tone | Mandatory Elements |
|---|---|---|
| 1–30 | Strong positive; lead with best evidence | Snippet + company/scale signal; positive behavioral if strong |
| 31–70 | Neutral; evidence + one concern if present | Snippet; concern if notice/consulting/inactive |
| 71–100 | Honest gap acknowledgment mandatory | At least one concern; "limited evidence" if no snippet |

---

## 11. Phase 7 — Manual Validation + Weight Tuning

### 11.1 Top-20 manual review

Read each of the top 20 profiles manually. Ask: would a senior recruiter agree this person fits a Senior AI Engineer role at a Series A? Flag:
- Keyword stuffers (lots of skills, no deployment evidence)
- Consultants who somehow ranked high despite the multiplier
- Wrong-domain candidates (CV/speech specialists)

Adjust weights in Phase 4 formula if patterns emerge.

### 11.2 Obvious-fit audit

Manually search the corpus for candidates who fit this description:
- Current title includes "ML Engineer," "AI Engineer," "Search Engineer," or "Relevance Engineer"
- Skills include FAISS, BGE, embedding, or NDCG
- Career history shows a product company

If these candidates are not in the top 10, trace which phase dropped them. The JD says "we'd rather see 10 great matches than 1000 maybes" and NDCG@10 is 50% of the score. A single misplaced obvious-fit in rank 15 instead of rank 5 costs significantly.

### 11.3 Honeypot audit

Scan top 100 for candidates with impossible timelines. If any known honeypot is in the top 10, the dense retrieval is over-indexing on keywords. Lower the dense weight in RRF or strengthen the honeypot gate.

The submission spec requires honeypot rate < 10% in top 100. Verify this before submitting.

### 11.4 Reasoning quality audit

Sample 10 rows from across ranks (not just top). Check:
- Every claim exists in the candidate's profile (no hallucination)
- Snippets reference specific tech names, companies, or numbers
- Tone is rank-appropriate (rank 90 should not have glowing reasoning)
- No two reasoning strings are identical

### 11.5 Pre-submission checklist

**File format:**
- [ ] Submission filename is `BuriBuri.csv` (registered team participant ID + `.csv` extension per spec §2)
- [ ] Encoding is UTF-8
- [ ] Exactly 100 rows of data plus 1 header row
- [ ] Column order is exactly: `candidate_id,rank,score,reasoning`
- [ ] Ranks 1–100 each appear exactly once
- [ ] Score non-increasing from rank 1 to rank 100 (ties allowed; ties broken by `candidate_id` ascending)
- [ ] Scores are not all identical (system is differentiating)
- [ ] No duplicate `candidate_id`s
- [ ] All `candidate_id`s match `^CAND_[0-9]{7}$` and exist in `candidates.jsonl`

**Quality:**
- [ ] Verify score is non-increasing from rank 1 to rank 100 and no duplicate IDs exist
- [ ] Honeypot rate in top 100 < 10% (spec §7: auto-disqualified at Stage 3 if exceeded)
- [ ] Top-20 manual review passed
- [ ] Obvious-fit candidates verified in top 10 (NDCG@10 = 50% of score)
- [ ] **Top 5 are pristine** — spec tiebreak §4: P@5 is the first tiebreaker between equal composites
- [ ] Reasoning variation check: 10 sampled rows are specific, honest, and varied
- [ ] No reasoning hallucinations (every claim exists in the candidate's profile)
- [ ] Rank-appropriate tone: rank 5 is not critical, rank 95 is not glowing

**Repo & sandbox:**
- [ ] Sandbox (HF Spaces / Gradio) accepts ≤100 candidates and ranks them end-to-end under 5 min on CPU
- [ ] `submission_metadata.yaml` complete at repo root
- [ ] `README.md` has exact reproduce command: `python rank.py --candidates ./candidates.jsonl --out ./submission.csv`
- [ ] Git history shows real iteration (at least 8 commits across phases; no single "Add everything" commit)

**Portal metadata (ready before upload — spec §10.2):**
- [ ] Team name: `BuriBuri`
- [ ] Primary contact name
- [ ] Primary contact email
- [ ] Primary contact phone
- [ ] GitHub repository URL (format: `https://github.com/USERNAME/REPO`)
- [ ] Sandbox / demo link (working HF Spaces URL)
- [ ] AI tools declared (multi-select: Claude / ChatGPT / Copilot / Cursor / Gemini / Other / None)
- [ ] Compute environment summary (one line: e.g. `Windows 11, Intel i7, 16GB RAM, Python 3.11`)
- [ ] Team member list (name + email for each member)
- [ ] Methodology summary (≤200 words — strongly recommended for Stage 4)

### 11.6 Dynamic Weight Tuning & Config Injection (weights.yaml)

To prevent code modification during final validation and parameter tuning, the implementation uses **Option A (Dynamic Configuration Injection)**:
- **Conceptual Simplicity in Plan:** The python code snippets within this `plan.md` specify hardcoded default numerical values (e.g., `0.55 * must_have_score` or `multiplier *= 0.80`) to remain clean, self-contained, and highly readable.
- **Dynamic Override in Execution:** In the actual codebase, all scoring weights, multipliers, boosts, and thresholds are loaded dynamically from `weights.yaml` (defined in detail at the bottom of [variables.md](file:///d:/GitHub/evidence-rank/variables.md)) into a central runtime configuration dictionary.
- **No-Code Iteration:** During manual review and validation (Phases 7.1–7.4), tuning parameter thresholds or weight distributions only requires editing `weights.yaml` and re-running the scripts. No Python files need to be opened or modified.

---

## 12. Data Field Reference

| Feature | JSON Path | Type | Notes |
|---|---|---|---|
| years_of_experience | `profile.years_of_experience` | float | 0–50 |
| current_title | `profile.current_title` | string | |
| headline | `profile.headline` | string | |
| summary | `profile.summary` | string | |
| current_company | `profile.current_company` | string | |
| current_industry | `profile.current_industry` | string | |
| location | `profile.location` | string | City, region |
| country | `profile.country` | string | |
| skill name | `skills[i].name` | string | |
| skill proficiency | `skills[i].proficiency` | string | beginner/intermediate/advanced/expert |
| skill endorsements | `skills[i].endorsements` | int | ≥0 |
| skill duration_months | `skills[i].duration_months` | int | ≥0 |
| career title | `career_history[i].title` | string | |
| career description | `career_history[i].description` | string | Primary evidence source |
| career company | `career_history[i].company` | string | |
| career industry | `career_history[i].industry` | string | |
| career duration | `career_history[i].duration_months` | int | |
| education tier | `education[i].tier` | string | tier_1 to tier_4 |
| assessment scores | `redrob_signals.skill_assessment_scores` | dict | skill_name → 0–100 |
| last active | `redrob_signals.last_active_date` | date | YYYY-MM-DD |
| open to work | `redrob_signals.open_to_work_flag` | bool | |
| profile views | `redrob_signals.profile_views_received_30d` | int | ≥0 |
| applications submitted | `redrob_signals.applications_submitted_30d` | int | ≥0 |
| response rate | `redrob_signals.recruiter_response_rate` | float | 0.0–1.0 |
| avg response time | `redrob_signals.avg_response_time_hours` | float | ≥0 |
| notice period | `redrob_signals.notice_period_days` | int | 0–180 |
| github score | `redrob_signals.github_activity_score` | float | -1 (no GitHub) to 100 |
| saved by recruiters | `redrob_signals.saved_by_recruiters_30d` | int | ≥0 |
| interview completion | `redrob_signals.interview_completion_rate` | float | 0.0–1.0 |
| offer acceptance | `redrob_signals.offer_acceptance_rate` | float | -1 (no history) to 1.0 |
| willing to relocate | `redrob_signals.willing_to_relocate` | bool | |
| preferred work mode | `redrob_signals.preferred_work_mode` | string | remote/hybrid/onsite/flexible |
| profile completeness | `redrob_signals.profile_completeness_score` | float | 0–100 |
| endorsements received | `redrob_signals.endorsements_received` | int | ≥0 |
| verified email | `redrob_signals.verified_email` | bool | |
| verified phone | `redrob_signals.verified_phone` | bool | |
| linkedin connected | `redrob_signals.linkedin_connected` | bool | |

---

## 13. Model & Library Choices

| Component | Choice | Size | Notes |
|---|---|---|---|
| Dense embedding | `BAAI/bge-m3` | 570 MB | Dense + sparse in one model. CPU ~2ms/doc. Offline only. |
| BM25 | `rank_bm25` | — | Pure Python. Built at precompute time, pickled to disk. Offline only. |
| Vector index | `faiss-cpu` `IndexFlatIP` | ~300 MB | Exact search on 100K. Built offline; not loaded at rank time. |
| Cross-encoder | `BAAI/bge-reranker-v2-m3` | 130 MB | Applied to top 300 offline. Scores saved to parquet. Offline only. |
| Feature extraction | `regex` + phrase dictionaries | — | Pure Python pattern matching. No NLP library dependency. Both offline and rank time. |
| Reason generation | Pure Python | — | String templates + evidence dict loaded from parquet. Rank time only. |

---

## 14. Dependency Declarations

### 14.1 Offline Dependencies (`preprocess.py` only)

These packages are required to build the precomputed artifacts. They are **not** imported by `rank.py` and do not need to be available at rank time.

```
flagembedding==1.2.5          # BGE-M3 embedder + bge-reranker-v2-m3
faiss-cpu==1.8.0              # FAISS IndexFlatIP build
torch>=2.0.0                  # Required by FlagEmbedding
sentence-transformers>=2.6.0  # Required by FlagEmbedding
```

### 14.2 Runtime Dependencies (`rank.py` only)

These are the only packages that `rank.py` imports. Total cold-import time should be under 2 seconds.

```
pandas==2.2.2
numpy==1.26.4
pyarrow==16.1.0
scipy==1.13.0
rank-bm25==0.2.2              # Required by app.py sandbox BM25 fallback path
```

### 14.3 Demo Dependencies (`app.py` only)

```
gradio==4.36.1
```

Total installed dependency size (offline + runtime + model artifacts) must stay within the 5 GB disk constraint.

> **No SpaCy.** Feature extraction uses pure Python `re` (regex) and phrase dictionaries only. SpaCy is not a dependency at any stage.

---

## 15. Reproducibility Flow

```
preprocess.py  (run once offline, no time limit)
    │
    ├── Phase 0: JD parse → artifacts/jd_*.json / jd_*_vector.npy
    ├── Phase 1: Embed 100K candidates → artifacts/faiss_index.bin
    │                                 → artifacts/bm25_index.pkl
    │                                 → artifacts/candidate_ids.json
    │                                 → artifacts/candidate_flags.parquet
    │                                 → artifacts/run_metadata.json  (reference_date = max last_active_date)
    ├── Phase 1d: RRF retrieval (FAISS + BM25 → RRF) → artifacts/retrieval_scores.parquet  [top 5,000]
    ├── Phase 1c: Feature extraction (Bucket A/B/C on top 5,000) → artifacts/candidate_features.parquet
    │            Preliminary core score computed here to rank candidates for CE selection
    └── Phase 1b: Cross-encoder top 500 (by preliminary core score) → artifacts/cross_encoder_scores.parquet

rank.py  (evaluation machine, ≤ 5 minutes wall clock)
    │
    ├── Load artifacts/retrieval_scores.parquet      → top 5000 candidates
    ├── Load artifacts/candidate_features.parquet    → Bucket A/B/C ready
    ├── Load artifacts/cross_encoder_scores.parquet  → CE scores ready (top 500 precomputed)
    ├── Load artifacts/run_metadata.json             → reference_date
    │   reference_date = max(stored_date, max(candidates last_active_date))
    │   (guards against time-drift if sandbox receives newer candidates)
    ├── Phase 4: Weighted score + CE merge → top 200–300
    ├── Phase 5: Behavioral multipliers → top 100
    └── Phase 6: Reason generation → submission.csv
```

**Dataset assumption:** `rank.py` expects precomputed artifacts for the official competition dataset. If `artifacts/retrieval_scores.parquet` is not found, it logs an error to stderr and exits. Runtime embedding generation is not supported.

**Reference date guard:** `rank.py` reads `reference_date` from `artifacts/run_metadata.json` but recalculates it as `max(stored_reference_date, max(new_candidates_last_active_date))` to prevent negative inactivity values if a sandbox payload contains candidates with dates newer than the precompute run.

---

### 15.1 Runtime Performance Logging

To ensure rapid debugging and timing verification on the evaluation machine, `rank.py` must track and log the following metrics to standard error (stderr):
- **Phase 2 (Retrieval) Time:** Parquet load + filter time.
- **Phase 3 (Feature Load) Time:** Parquet join time.
- **Phase 4 (Scoring + CE Merge) Time:** Weighted formula + parquet join.
- **Phase 5 & 6 (Behavioral + Reasoning) Time:** Wall-clock duration.
- **Total Runtime:** Overall ranking process runtime.
- **Retrieved Count:** Number of candidates after Phase 2 filter.
- **Peak Memory Usage:** Process peak RSS memory footprint.

The offline precompute (`preprocess.py`) has no time constraint and may use any compute needed.

---

### 15.2 Run Metadata Schema

`artifacts/run_metadata.json` is a simple dictionary generated by `preprocess.py`:

```json
{
  "reference_date": "2026-06-05",
  "total_candidates_processed": 100000,
  "faiss_index_size": 100000
}
```
`rank.py` uses `reference_date` to compute days inactive.

---

*End of specification. Version 3.3.0.*
