# Evidence-Rank: Intelligent Candidate Discovery & Ranking Engine

> **⚠️ TENTATIVE WORKING DOCUMENT**
> This README reflects the *current proposed architecture* for the Evidence-Rank pipeline. As we actively develop, test, and tune the codebase, this document is subject to change. It serves as our living blueprint.

**Hackathon:** Redrob × Hack2skill — Intelligent Candidate Discovery & Ranking Challenge  
**Role:** Senior AI Engineer, Founding Team, Redrob AI  
**Task:** Rank 100,000 candidates against a job description. Output the top 100, ranked best-fit first, with a 1–2 sentence explanation for each.

---

## 🏗 System Overview

The ranking step must run in under 5 minutes on a standard CPU with 16 GB RAM, no internet, no GPU. This forces a clean two-phase design: Heavy AI inference runs offline during precompute, while ranking logic runs memory-safe and lightning-fast at runtime using Polars.

```text
JD → [Phase 0: JD Intelligence] → Structured Config + Vectors + CSR
Corpus → [Phase 1: Preprocessing] → FAISS + BM25 + Offline Cross-Encoder
                                              ↓ (runtime starts here, ≤ 5 min)
                              [Phase 2: Retrieval] → Top 3,000 (Polars)
                                              ↓
                         [Phase 3: Feature Extraction] → Evidence Buckets (Sentinel Value Safe)
                                              ↓
                          [Phase 4: Core Scoring + CE Rerank] → Normalized 100.0 Scale
                                              ↓
                    [Phase 5: Behavioral & Logistic Adjustment] → Top 100 (Safety Floored)
                                              ↓
                         [Phase 6: Reason Generation] → Final CSV (No Hallucinations)
```

---

## 🚀 The Pipeline Phases

### Phase 0 — JD Intelligence (Offline)
The JD is parsed into actionable signals and embedded into **three semantic query vectors** (Skills, RecSys Persona, Eval Persona) plus a Sparse Lexical query. This captures both the exact technical terminology and the plain-language descriptions of an ideal hire.

### Phase 1 — Corpus Preprocessing & Honeypot Detection (Offline)
All 100,000 profiles are embedded (Dense and Sparse). We pre-calculate all Cross-Encoder (`bge-reranker-v2-m3`) scores for the top 5,000 candidates offline to bypass the massive runtime bottleneck.
**Security:** Deliberate "Honeypot" and "Ghost" candidates are flagged here.

### Phase 2 — Multi-Signal Retrieval (Runtime)
We use a **5-Way Reciprocal Rank Fusion (RRF)** to retrieve the best candidates across orthogonal dimensions:
1. Dense FAISS v1 (Skills)
2. Dense FAISS HyDE (RecSys Persona)
3. Dense FAISS HyDE (Eval Persona)
4. Learned Sparse (CSR Dot-Product)
5. Lexical BM25
The pre-computed Top 5,000 RRF pool is loaded lightning-fast into memory via `Polars`.

### Phase 3 — Candidate Feature Extraction (Runtime)
Rule-based extraction (using Regex, no slow NLP libraries) produces four evidence buckets:
- **Bucket A (Must-Haves):** Vector DBs, Hybrid Search, NDCG eval.
- **Bucket B (Career Quality):** Shipper attitude, writing culture, product company exposure.
- **Bucket C (Gaps):** Title-chasers, stopped coding, wrong domains.
- **Consistency Score:** Detects keyword stuffers who claim "Expert" without career proof.
*Resilience:* Missing metadata is handled gracefully using Sentinel Values (`-1`, `"UNKNOWN"`) to prevent null-crashes.

### Phase 4 — Core Scoring & Reranking (Runtime)
Candidates receive a strict **0–100.0 Core Score** combining the must-have, nice-to-have, and career quality buckets. 
The offline Cross-Encoder scores (also scaled 0-100.0) are left-joined using `Polars`. A final 80/20 blend creates the definitive technical rank.

### Phase 5 — Behavioral & Logistic Adjustment (Runtime)
Technical fit is translated into a *hiring* fit using soft multipliers:
- **Availability:** Inactive candidates get compound penalties (down to `0.60x`).
- **Logistics:** Location and notice-period mismatches nudge scores down.
- **Bonus:** 90-day milestone readiness and social proof add additive points.
- **The Kill Switch:** Identified Honeypots are hit with a `0.01x` multiplier. Thanks to the strict 100.0 scale, this mathematically forces them down to a maximum score of `1.0`, permanently dropping them out of the Top 100.

### Phase 6 — Reason Generation (Runtime)
Reasons are assembled deterministically. To pass human review, the generator dynamically selects the candidate's strongest domain (e.g., Evaluation vs. Infrastructure) to lead the sentence. Missing data gracefully falls back to generic templates via try-except blocks, ensuring no runtime crashes and zero hallucinations.

---

## 🛠 Reproducibility

```bash
# Step 1: Precompute (run once, no time limit)
python preprocess.py --candidates ./candidates.jsonl

# Step 2: Rank (must complete in under 5 minutes)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# Step 3: Validate
python validate_submission.py submission.csv
```
